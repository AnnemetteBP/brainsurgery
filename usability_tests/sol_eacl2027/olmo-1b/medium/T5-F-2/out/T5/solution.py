#!/usr/bin/env python3
"""Merge the supplied PEFT LoRA directly into a sharded safetensors model."""

from __future__ import annotations

import json
from contextlib import ExitStack
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


ROOT = Path(__file__).resolve().parents[2]
BASE_DIR = ROOT / "inputs" / "base"
LORA_DIR = ROOT / "inputs" / "lora"
OUT_DIR = ROOT / "out" / "T5"
INDEX_NAME = "model.safetensors.index.json"
MAX_SHARD_BYTES = 512 * 1024 * 1024

DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "F64": 8,
    "I64": 8,
    "U64": 8,
}


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def tensor_specs(weight_map: dict[str, str]) -> dict[str, tuple[list[int], str, int]]:
    """Read shapes/dtypes without materializing the checkpoint tensors."""
    specs: dict[str, tuple[list[int], str, int]] = {}
    by_file: dict[str, list[str]] = {}
    for name, filename in weight_map.items():
        by_file.setdefault(filename, []).append(name)
    for filename, names in by_file.items():
        with safe_open(BASE_DIR / filename, framework="pt", device="cpu") as reader:
            for name in names:
                view = reader.get_slice(name)
                shape = list(view.get_shape())
                dtype = str(view.get_dtype())
                assert dtype in DTYPE_BYTES, f"Unsupported safetensors dtype {dtype} for {name}"
                elements = 1
                for dim in shape:
                    elements *= dim
                specs[name] = (shape, dtype, elements * DTYPE_BYTES[dtype])
    return specs


def adapter_pairs(adapter_keys: set[str]) -> dict[str, tuple[str, str]]:
    suffix_a = ".lora_A.weight"
    suffix_b = ".lora_B.weight"
    prefix = "base_model.model."
    a_keys = {key for key in adapter_keys if key.endswith(suffix_a)}
    b_keys = {key for key in adapter_keys if key.endswith(suffix_b)}
    other = adapter_keys - a_keys - b_keys
    assert not other, f"Unexpected non-LoRA adapter tensors: {sorted(other)}"

    pairs: dict[str, tuple[str, str]] = {}
    for a_key in sorted(a_keys):
        stem = a_key[: -len(suffix_a)]
        b_key = stem + suffix_b
        assert b_key in b_keys, f"Missing B factor for {a_key}"
        assert stem.startswith(prefix), f"Unexpected PEFT key prefix: {a_key}"
        base_name = stem[len(prefix) :] + ".weight"
        assert base_name not in pairs, f"Duplicate adapter target: {base_name}"
        pairs[base_name] = (a_key, b_key)
    assert b_keys == {b for _, b in pairs.values()}, "Unpaired LoRA B factor found"
    return pairs


def make_shard_plan(names: list[str], specs: dict[str, tuple[list[int], str, int]]) -> list[list[str]]:
    shards: list[list[str]] = []
    current: list[str] = []
    current_bytes = 0
    for name in names:
        size = specs[name][2]
        if size > MAX_SHARD_BYTES:
            if current:
                shards.append(current)
                current, current_bytes = [], 0
            shards.append([name])
        else:
            if current and current_bytes + size > MAX_SHARD_BYTES:
                shards.append(current)
                current, current_bytes = [], 0
            current.append(name)
            current_bytes += size
    if current:
        shards.append(current)
    return shards


def main() -> None:
    base_index = read_json(BASE_DIR / INDEX_NAME)
    weight_map: dict[str, str] = base_index["weight_map"]
    base_names = list(weight_map)
    assert len(base_names) == len(set(base_names)), "Duplicate tensor name in base index"
    specs = tensor_specs(weight_map)
    assert set(specs) == set(base_names), "Base index and shard contents disagree"

    config = read_json(LORA_DIR / "adapter_config.json")
    rank = config["r"]
    alpha = config["lora_alpha"]
    assert isinstance(rank, int) and rank > 0, f"Invalid LoRA rank: {rank}"
    assert config.get("fan_in_fan_out") is False, "This merger expects nn.Linear (fan_in_fan_out=false) layout"
    scale = float(alpha) / rank

    adapter_path = LORA_DIR / "adapter_model.safetensors"
    with safe_open(adapter_path, framework="pt", device="cpu") as adapter:
        pairs = adapter_pairs(set(adapter.keys()))

    # All task-mandated checks occur before the first output checkpoint write.
    assert len(pairs) == 32, f"Expected exactly 32 adapter pairs, found {len(pairs)}"
    missing_targets = sorted(set(pairs) - set(base_names))
    assert not missing_targets, f"Adapter targets absent from base: {missing_targets}"
    assert not any("lora_" in name for name in base_names), "LoRA tensor name would enter output"
    probe = "model.layers.0.self_attn.q_proj.weight"
    assert specs[probe][0] == [2048, 2048], f"Unexpected {probe} shape: {specs[probe][0]}"
    assert len(base_names) == 114, f"Expected exactly 114 output tensors, found {len(base_names)}"

    plan = make_shard_plan(base_names, specs)
    shard_names = [f"model-{i:05d}-of-{len(plan):05d}.safetensors" for i in range(1, len(plan) + 1)]
    output_map = {
        tensor_name: shard_name
        for shard_name, tensor_names in zip(shard_names, plan)
        for tensor_name in tensor_names
    }
    assert set(output_map) == set(base_names)
    for tensor_names in plan:
        size = sum(specs[name][2] for name in tensor_names)
        assert size <= MAX_SHARD_BYTES or len(tensor_names) == 1

    # Refuse to mix a prior checkpoint with this export.
    conflicts = [OUT_DIR / INDEX_NAME, *(OUT_DIR / name for name in shard_names)]
    assert not any(path.exists() for path in conflicts), "Output checkpoint already exists; remove it before rerunning"

    with ExitStack() as stack:
        base_readers = {
            filename: stack.enter_context(safe_open(BASE_DIR / filename, framework="pt", device="cpu"))
            for filename in sorted(set(weight_map.values()))
        }
        adapter = stack.enter_context(safe_open(adapter_path, framework="pt", device="cpu"))

        for shard_name, tensor_names in zip(shard_names, plan):
            tensors: dict[str, torch.Tensor] = {}
            for name in tensor_names:
                base = base_readers[weight_map[name]].get_tensor(name)
                if name in pairs:
                    a_key, b_key = pairs[name]
                    a = adapter.get_tensor(a_key)
                    b = adapter.get_tensor(b_key)
                    assert base.dtype == a.dtype == b.dtype == torch.float32, f"Non-float32 merge inputs for {name}"
                    assert tuple(a.shape) == (rank, base.shape[1]), f"Bad A shape for {name}: {tuple(a.shape)}"
                    assert tuple(b.shape) == (base.shape[0], rank), f"Bad B shape for {name}: {tuple(b.shape)}"
                    merged = torch.addmm(base, b, a, beta=1.0, alpha=scale)
                    assert merged.shape == base.shape and merged.dtype == torch.float32
                    tensors[name] = merged
                else:
                    tensors[name] = base
            save_file(tensors, OUT_DIR / shard_name)

    output_index = {
        "metadata": {"total_size": sum(spec[2] for spec in specs.values())},
        "weight_map": output_map,
    }
    with (OUT_DIR / INDEX_NAME).open("w", encoding="utf-8") as handle:
        json.dump(output_index, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(f"Merged {len(pairs)} LoRA pairs into {len(base_names)} tensors across {len(plan)} shards")


if __name__ == "__main__":
    main()
