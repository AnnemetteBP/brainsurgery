#!/usr/bin/env python3
"""Create the mixed-precision, size-bounded OLMo safetensors export."""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from contextlib import ExitStack
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


INPUT = Path("inputs/base")
OUTPUT = Path("out/T3")
INDEX_NAME = "model.safetensors.index.json"
MAX_SHARD_BYTES = 256 * 1024 * 1024
SHARD_RE = re.compile(r"model-\d{5}-of-\d{5}\.safetensors$")


def projection_names() -> set[str]:
    names: set[str] = set()
    for layer in range(16):
        prefix = f"model.layers.{layer}"
        names.update(
            {
                f"{prefix}.self_attn.{projection}_proj.weight"
                for projection in ("q", "k", "v", "o")
            }
        )
        names.update(
            {
                f"{prefix}.mlp.{projection}_proj.weight"
                for projection in ("gate", "up", "down")
            }
        )
    return names


def tensor_bytes(shape: list[int], dtype: torch.dtype) -> int:
    elements = 1
    for dimension in shape:
        elements *= dimension
    return elements * torch.empty((), dtype=dtype).element_size()


def main() -> None:
    input_index = json.loads((INPUT / INDEX_NAME).read_text())
    input_map: dict[str, str] = input_index["weight_map"]
    target_names = projection_names()

    # Inspect only headers first. This establishes the complete output plan and
    # enforces all required checks before any output checkpoint data is written.
    headers: dict[str, tuple[list[int], str]] = {}
    by_file: dict[str, set[str]] = {}
    for name, filename in input_map.items():
        by_file.setdefault(filename, set()).add(name)
    for filename, expected_names in by_file.items():
        with safe_open(INPUT / filename, framework="pt", device="cpu") as source:
            actual_names = set(source.keys())
            if actual_names != expected_names:
                raise AssertionError(
                    f"index/header mismatch in {filename}: "
                    f"missing={sorted(expected_names - actual_names)}, "
                    f"extra={sorted(actual_names - expected_names)}"
                )
            for name in actual_names:
                view = source.get_slice(name)
                headers[name] = (view.get_shape(), view.get_dtype())

    all_names = set(headers)
    if len(all_names) != 114:
        raise AssertionError(f"output plan has {len(all_names)} tensors, expected 114")
    if len(target_names) != 112:
        raise AssertionError(f"projection allowlist has {len(target_names)} names, expected 112")
    if not target_names <= all_names:
        raise AssertionError(f"missing projections: {sorted(target_names - all_names)}")
    non_targets = all_names - target_names
    if non_targets != {"model.embed_tokens.weight", "lm_head.weight"}:
        raise AssertionError(f"unexpected non-projection tensors: {sorted(non_targets)}")
    non_f32 = sorted(name for name, (_, dtype) in headers.items() if dtype != "F32")
    if non_f32:
        raise AssertionError(f"input tensors are not all float32: {non_f32}")

    planned_dtype = {
        name: torch.bfloat16 if name in target_names else torch.float32
        for name in all_names
    }
    if sum(dtype == torch.bfloat16 for dtype in planned_dtype.values()) != 112:
        raise AssertionError("output plan does not contain exactly 112 bfloat16 tensors")
    if planned_dtype["model.layers.0.self_attn.q_proj.weight"] != torch.bfloat16:
        raise AssertionError("layer 0 q_proj is not planned as bfloat16")
    if planned_dtype["model.embed_tokens.weight"] != torch.float32:
        raise AssertionError("embedding is not planned as float32")

    sizes = {
        name: tensor_bytes(headers[name][0], planned_dtype[name]) for name in all_names
    }
    groups: list[list[str]] = []
    current: list[str] = []
    current_size = 0
    for name in sorted(all_names):
        size = sizes[name]
        if size > MAX_SHARD_BYTES:
            if current:
                groups.append(current)
                current, current_size = [], 0
            groups.append([name])
        elif current and current_size + size > MAX_SHARD_BYTES:
            groups.append(current)
            current, current_size = [name], size
        else:
            current.append(name)
            current_size += size
    if current:
        groups.append(current)

    for group in groups:
        group_size = sum(sizes[name] for name in group)
        if group_size > MAX_SHARD_BYTES and len(group) != 1:
            raise AssertionError(f"oversized multi-tensor shard plan: {group_size} bytes")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".staging-", dir=OUTPUT))
    weight_map: dict[str, str] = {}
    try:
        with ExitStack() as stack:
            sources = {
                filename: stack.enter_context(
                    safe_open(INPUT / filename, framework="pt", device="cpu")
                )
                for filename in by_file
            }
            shard_count = len(groups)
            for number, group in enumerate(groups, start=1):
                filename = f"model-{number:05d}-of-{shard_count:05d}.safetensors"
                tensors: dict[str, torch.Tensor] = {}
                for name in group:
                    tensor = sources[input_map[name]].get_tensor(name)
                    tensors[name] = tensor.to(planned_dtype[name])
                save_file(tensors, staging / filename)
                weight_map.update({name: filename for name in group})
                del tensors

        # Validate the staged checkpoint itself before publishing its index.
        observed: dict[str, tuple[str, list[int]]] = {}
        for filename in sorted(set(weight_map.values())):
            shard_names = [name for name, shard in weight_map.items() if shard == filename]
            shard_size = sum(sizes[name] for name in shard_names)
            if shard_size > MAX_SHARD_BYTES and len(shard_names) != 1:
                raise AssertionError(f"{filename} contains {shard_size} bytes")
            with safe_open(staging / filename, framework="pt", device="cpu") as shard:
                if set(shard.keys()) != set(shard_names):
                    raise AssertionError(f"incorrect key set in {filename}")
                for name in shard.keys():
                    view = shard.get_slice(name)
                    observed[name] = (view.get_dtype(), view.get_shape())

        if set(observed) != all_names or len(observed) != 114:
            raise AssertionError(f"staged output has {len(observed)} tensors, expected 114")
        bf16_count = sum(dtype == "BF16" for dtype, _ in observed.values())
        if bf16_count != 112:
            raise AssertionError(f"staged output has {bf16_count} bfloat16 tensors")
        if observed["model.layers.0.self_attn.q_proj.weight"][0] != "BF16":
            raise AssertionError("staged layer 0 q_proj is not bfloat16")
        if observed["model.embed_tokens.weight"][0] != "F32":
            raise AssertionError("staged embedding is not float32")

        output_index = {
            "metadata": {"total_size": sum(sizes.values())},
            "weight_map": {name: weight_map[name] for name in sorted(weight_map)},
        }
        (staging / INDEX_NAME).write_text(json.dumps(output_index, indent=2) + "\n")

        new_files = set(weight_map.values())
        for path in OUTPUT.iterdir():
            if path.is_file() and SHARD_RE.fullmatch(path.name) and path.name not in new_files:
                path.unlink()
        for filename in sorted(new_files):
            os.replace(staging / filename, OUTPUT / filename)
        os.replace(staging / INDEX_NAME, OUTPUT / INDEX_NAME)
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    print(
        f"Wrote {len(all_names)} tensors ({len(target_names)} bfloat16) "
        f"across {len(groups)} shards; total tensor data={sum(sizes.values())} bytes"
    )


if __name__ == "__main__":
    main()
