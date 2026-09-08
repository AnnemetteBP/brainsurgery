#!/usr/bin/env python3
"""Merge the supplied PEFT LoRA into the base checkpoint and export shards."""

from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "inputs" / "base"
LORA = ROOT / "inputs" / "lora"
OUT = Path(__file__).resolve().parent
INDEX_NAME = "model.safetensors.index.json"
MAX_SHARD_BYTES = 512 * 1024 * 1024
LARGE_TENSORS = {"model.embed_tokens.weight", "lm_head.weight"}


def tensor_info(path: Path, name: str) -> tuple[list[int], torch.dtype, int]:
    """Return shape, torch dtype, and tensor-data byte count without loading."""
    with safe_open(path, framework="pt", device="cpu") as handle:
        view = handle.get_slice(name)
        shape = list(view.get_shape())
        dtype_name = view.get_dtype()
    dtype = {
        "F32": torch.float32,
        "F16": torch.float16,
        "BF16": torch.bfloat16,
    }.get(dtype_name)
    if dtype is None:
        raise AssertionError(f"unsupported dtype {dtype_name} for {name}")
    elements = 1
    for extent in shape:
        elements *= extent
    return shape, dtype, elements * torch.empty((), dtype=dtype).element_size()


def make_groups(names: list[str], sizes: dict[str, int]) -> list[list[str]]:
    """Pack tensors in deterministic order, keeping the two large matrices alone."""
    groups: list[list[str]] = []
    current: list[str] = []
    current_bytes = 0
    for name in names:
        size = sizes[name]
        if size > MAX_SHARD_BYTES:
            if current:
                groups.append(current)
                current, current_bytes = [], 0
            groups.append([name])
        elif name in LARGE_TENSORS:
            if current:
                groups.append(current)
                current, current_bytes = [], 0
            groups.append([name])
        else:
            if current and current_bytes + size > MAX_SHARD_BYTES:
                groups.append(current)
                current, current_bytes = [], 0
            current.append(name)
            current_bytes += size
    if current:
        groups.append(current)
    return groups


def main() -> None:
    config = json.loads((LORA / "adapter_config.json").read_text())
    rank = config["r"]
    alpha = config["lora_alpha"]
    fan_in_fan_out = config["fan_in_fan_out"]
    assert rank == 16, f"unexpected LoRA rank: {rank}"
    assert alpha == 32, f"unexpected LoRA alpha: {alpha}"
    assert fan_in_fan_out is False, "this merge expects nn.Linear [out, in] layout"
    scale = alpha / rank

    base_index = json.loads((BASE / INDEX_NAME).read_text())
    base_map: dict[str, str] = base_index["weight_map"]
    assert len(base_map) == 114, f"expected 114 base tensors, found {len(base_map)}"

    # Validate that the index and shard contents describe exactly one copy of each tensor.
    shard_keys: dict[str, set[str]] = {}
    for shard_name in sorted(set(base_map.values())):
        with safe_open(BASE / shard_name, framework="pt", device="cpu") as handle:
            shard_keys[shard_name] = set(handle.keys())
    assert set(base_map) == set().union(*shard_keys.values()), \
        "base index key set does not match shard contents"
    assert all(name in shard_keys[file] for name, file in base_map.items()), \
        "base index maps at least one tensor to the wrong shard"

    adapter = load_file(LORA / "adapter_model.safetensors", device="cpu")
    adapter_names = set(adapter)
    pairs: dict[str, tuple[str, str]] = {}
    prefix = "base_model.model."
    suffix_a = ".lora_A.weight"
    for a_name in sorted(name for name in adapter_names if name.endswith(suffix_a)):
        assert a_name.startswith(prefix), f"unexpected adapter prefix: {a_name}"
        stem = a_name[: -len(suffix_a)]
        b_name = stem + ".lora_B.weight"
        assert b_name in adapter_names, f"missing B factor for {a_name}"
        base_name = stem.removeprefix(prefix) + ".weight"
        assert base_name in base_map, f"adapter target absent from base: {base_name}"
        assert base_name not in pairs, f"duplicate adapter target: {base_name}"
        a, b = adapter[a_name], adapter[b_name]
        shape, dtype, _ = tensor_info(BASE / base_map[base_name], base_name)
        assert a.dtype == b.dtype == dtype == torch.float32, \
            f"non-float32 inputs for {base_name}"
        expected = [b.shape[0], a.shape[1]]
        assert a.shape[0] == b.shape[1] == rank and shape == expected, \
            f"incompatible factor/base shapes for {base_name}: {a.shape}, {b.shape}, {shape}"
        pairs[base_name] = (a_name, b_name)

    paired_adapter_names = {name for pair in pairs.values() for name in pair}
    assert paired_adapter_names == adapter_names, "unpaired or unexpected adapter tensors found"

    # Mandatory pre-write checks.
    output_names = set(base_map)
    assert len(pairs) == 32, f"expected exactly 32 adapter pairs, found {len(pairs)}"
    assert not any("lora_" in name for name in output_names), "LoRA name would leak into output"
    q0 = "model.layers.0.self_attn.q_proj.weight"
    q0_shape, _, _ = tensor_info(BASE / base_map[q0], q0)
    assert q0_shape == [2048, 2048], f"unexpected q_proj shape: {q0_shape}"
    assert len(output_names) == 114, f"expected exactly 114 output tensors, found {len(output_names)}"

    sizes = {
        name: tensor_info(BASE / base_map[name], name)[2]
        for name in sorted(output_names)
    }
    groups = make_groups(sorted(output_names), sizes)
    assert all(sum(sizes[name] for name in group) <= MAX_SHARD_BYTES for group in groups), \
        "planned shard exceeds 512 MiB"

    # Remove only checkpoint products from a previous invocation; keep source/report files.
    for old in OUT.glob("model-*-of-*.safetensors"):
        old.unlink()
    (OUT / INDEX_NAME).unlink(missing_ok=True)

    weight_map: dict[str, str] = {}
    shard_count = len(groups)
    for number, group in enumerate(groups, 1):
        shard_name = f"model-{number:05d}-of-{shard_count:05d}.safetensors"
        tensors: dict[str, torch.Tensor] = {}
        by_source: dict[str, list[str]] = defaultdict(list)
        for name in group:
            by_source[base_map[name]].append(name)
        for source_name, source_tensor_names in by_source.items():
            with safe_open(BASE / source_name, framework="pt", device="cpu") as handle:
                for name in source_tensor_names:
                    base_tensor = handle.get_tensor(name)
                    if name in pairs:
                        a_name, b_name = pairs[name]
                        # addmm performs the specified float32 B @ A update.
                        base_tensor = torch.addmm(
                            base_tensor, adapter[b_name], adapter[a_name],
                            beta=1.0, alpha=scale,
                        )
                    tensors[name] = base_tensor
        temp = OUT / f".{shard_name}.tmp"
        save_file(tensors, temp, metadata={"format": "pt"})
        os.replace(temp, OUT / shard_name)
        weight_map.update({name: shard_name for name in group})
        print(f"wrote {shard_name}: {len(group)} tensors, "
              f"{sum(sizes[name] for name in group):,} data bytes")
        del tensors

    output_index = {
        "metadata": {"total_size": sum(sizes.values())},
        "weight_map": weight_map,
    }
    temp_index = OUT / f".{INDEX_NAME}.tmp"
    temp_index.write_text(json.dumps(output_index, indent=2, sort_keys=True) + "\n")
    os.replace(temp_index, OUT / INDEX_NAME)
    print(f"merged {len(pairs)} LoRA pairs; exported {len(weight_map)} tensors in {shard_count} shards")


if __name__ == "__main__":
    main()
