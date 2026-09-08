#!/usr/bin/env python3
"""Export the OLMo checkpoint with projection-only BF16 and <=256 MiB shards."""

from __future__ import annotations

import json
import re
from collections import OrderedDict
from contextlib import ExitStack
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


INPUT_DIR = Path(__file__).resolve().parents[2] / "inputs" / "base"
OUTPUT_DIR = Path(__file__).resolve().parent
INDEX_NAME = "model.safetensors.index.json"
MAX_SHARD_BYTES = 256 * 1024 * 1024

PROJECTION_RE = re.compile(
    r"^model\.layers\.(\d+)\."
    r"(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|"
    r"mlp\.(?:gate_proj|up_proj|down_proj))\.weight$"
)


def tensor_nbytes(shape: list[int], dtype: torch.dtype) -> int:
    elements = 1
    for dimension in shape:
        elements *= dimension
    return elements * torch.empty((), dtype=dtype).element_size()


def main() -> None:
    source_index = json.loads((INPUT_DIR / INDEX_NAME).read_text())
    source_map: dict[str, str] = source_index["weight_map"]
    names = list(source_map)

    # Read only headers for preflight. No destination checkpoint is touched until
    # the complete output schema, dtype assignment, and shard plan are validated.
    metadata: dict[str, tuple[list[int], str]] = {}
    for shard_name in sorted(set(source_map.values())):
        expected = {name for name, shard in source_map.items() if shard == shard_name}
        with safe_open(INPUT_DIR / shard_name, framework="pt", device="cpu") as handle:
            actual = set(handle.keys())
            if actual != expected:
                raise AssertionError(
                    f"source index mismatch in {shard_name}: "
                    f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
                )
            for name in actual:
                view = handle.get_slice(name)
                metadata[name] = (view.get_shape(), view.get_dtype())

    expected_projections = {
        f"model.layers.{layer}.{group}.{projection}.weight"
        for layer in range(16)
        for group, projections in (
            ("self_attn", ("q_proj", "k_proj", "v_proj", "o_proj")),
            ("mlp", ("gate_proj", "up_proj", "down_proj")),
        )
        for projection in projections
    }
    matched_projections = {name for name in names if PROJECTION_RE.fullmatch(name)}
    if matched_projections != expected_projections:
        raise AssertionError(
            "projection key mismatch: "
            f"missing={sorted(expected_projections - matched_projections)}, "
            f"extra={sorted(matched_projections - expected_projections)}"
        )
    if set(metadata) != set(names):
        raise AssertionError("source metadata does not cover the indexed tensor set")
    non_f32 = sorted(name for name, (_, dtype) in metadata.items() if dtype != "F32")
    if non_f32:
        raise AssertionError(f"source tensors must all be float32, found: {non_f32}")

    output_dtype = {
        name: torch.bfloat16 if name in matched_projections else torch.float32
        for name in names
    }

    # Mandatory pre-write checks from the task specification.
    bf16_count = sum(dtype == torch.bfloat16 for dtype in output_dtype.values())
    if bf16_count != 112:
        raise AssertionError(f"expected exactly 112 bfloat16 tensors, got {bf16_count}")
    if output_dtype.get("model.layers.0.self_attn.q_proj.weight") != torch.bfloat16:
        raise AssertionError("layer 0 q_proj was not selected for bfloat16")
    if output_dtype.get("model.embed_tokens.weight") != torch.float32:
        raise AssertionError("model.embed_tokens.weight was not kept float32")
    if len(output_dtype) != 114:
        raise AssertionError(f"expected exactly 114 output tensors, got {len(output_dtype)}")

    # Greedy deterministic packing. Oversized tensors are valid only as singleton
    # shards; ordinary shards never exceed the tensor-data byte limit.
    shard_groups: list[list[str]] = []
    current: list[str] = []
    current_bytes = 0
    for name in names:
        size = tensor_nbytes(metadata[name][0], output_dtype[name])
        if size > MAX_SHARD_BYTES:
            if current:
                shard_groups.append(current)
                current, current_bytes = [], 0
            shard_groups.append([name])
        else:
            if current and current_bytes + size > MAX_SHARD_BYTES:
                shard_groups.append(current)
                current, current_bytes = [], 0
            current.append(name)
            current_bytes += size
    if current:
        shard_groups.append(current)

    for group in shard_groups:
        group_size = sum(tensor_nbytes(metadata[name][0], output_dtype[name]) for name in group)
        if group_size > MAX_SHARD_BYTES and len(group) != 1:
            raise AssertionError(f"oversized non-singleton shard plan: {group_size} bytes")
    planned_names = [name for group in shard_groups for name in group]
    if len(planned_names) != 114 or set(planned_names) != set(names):
        raise AssertionError("shard plan does not contain exactly the 114 source tensors")

    existing = sorted(OUTPUT_DIR.glob("model-*-of-*.safetensors"))
    if existing or (OUTPUT_DIR / INDEX_NAME).exists():
        raise FileExistsError("destination checkpoint already exists; refusing to overwrite it")

    shard_count = len(shard_groups)
    weight_map: dict[str, str] = {}
    total_size = 0
    with ExitStack() as stack:
        handles = {
            shard_name: stack.enter_context(
                safe_open(INPUT_DIR / shard_name, framework="pt", device="cpu")
            )
            for shard_name in sorted(set(source_map.values()))
        }
        for shard_number, group in enumerate(shard_groups, start=1):
            filename = f"model-{shard_number:05d}-of-{shard_count:05d}.safetensors"
            tensors: OrderedDict[str, torch.Tensor] = OrderedDict()
            for name in group:
                tensor = handles[source_map[name]].get_tensor(name)
                tensor = tensor.to(dtype=output_dtype[name])
                tensors[name] = tensor.contiguous()
                weight_map[name] = filename
                total_size += tensors[name].numel() * tensors[name].element_size()
            save_file(tensors, OUTPUT_DIR / filename)

    if len(weight_map) != 114 or set(weight_map) != set(names):
        raise AssertionError("internal error: written weight map is incomplete")
    destination_index = {
        "metadata": {"total_size": total_size},
        "weight_map": weight_map,
    }
    (OUTPUT_DIR / INDEX_NAME).write_text(json.dumps(destination_index, indent=2) + "\n")
    print(
        f"Wrote {len(weight_map)} tensors ({bf16_count} BF16) "
        f"across {shard_count} shards; tensor data={total_size:,} bytes"
    )


if __name__ == "__main__":
    main()
