#!/usr/bin/env python3
"""Merge the supplied PEFT LoRA into the sharded base checkpoint."""

import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "inputs" / "base"
LORA = ROOT / "inputs" / "lora"
OUT = ROOT / "out" / "T5"
CAP = 512 * 1024 * 1024
FORCE_ALONE = {"model.embed_tokens.weight", "lm_head.weight"}


def tensor_info(path: Path, key: str):
    with safe_open(path, framework="pt", device="cpu") as f:
        view = f.get_slice(key)
        shape = tuple(view.get_shape())
        dtype = f.get_tensor(key).dtype
    return shape, dtype


def main() -> None:
    config = json.loads((LORA / "adapter_config.json").read_text())
    assert config["r"] == 16, f"unexpected LoRA rank: {config['r']}"
    assert config["lora_alpha"] == 32, "unexpected LoRA alpha"
    assert config["fan_in_fan_out"] is False, "transposed LoRA is unsupported here"
    scale = float(config["lora_alpha"]) / int(config["r"])

    index = json.loads((BASE / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    assert len(weight_map) == 114, f"expected 114 base tensors, found {len(weight_map)}"
    assert not any("lora_" in key for key in weight_map), "LoRA name in output key set"

    adapter_path = LORA / "adapter_model.safetensors"
    with safe_open(adapter_path, framework="pt", device="cpu") as adapter:
        adapter_keys = set(adapter.keys())

    pairs = {}
    prefix = "base_model.model."
    a_suffix = ".lora_A.weight"
    for a_key in sorted(k for k in adapter_keys if k.endswith(a_suffix)):
        stem = a_key[: -len(a_suffix)]
        b_key = stem + ".lora_B.weight"
        assert b_key in adapter_keys, f"missing pair for {a_key}"
        assert stem.startswith(prefix), f"unexpected adapter key: {a_key}"
        base_key = stem[len(prefix) :] + ".weight"
        assert base_key in weight_map, f"adapter target absent from base: {base_key}"
        pairs[base_key] = (a_key, b_key)
    paired_adapter_keys = {k for pair in pairs.values() for k in pair}
    assert paired_adapter_keys == adapter_keys, "unpaired or unexpected adapter tensors found"
    assert len(pairs) == 32, f"expected exactly 32 adapter pairs, found {len(pairs)}"

    # Validate every required invariant and all merge shapes before any output is written.
    sizes = {}
    dtypes = {}
    shapes = {}
    for key, shard in weight_map.items():
        shape, dtype = tensor_info(BASE / shard, key)
        shapes[key], dtypes[key] = shape, dtype
        sizes[key] = dtype.itemsize * int(torch.tensor(shape).prod().item())
    check_key = "model.layers.0.self_attn.q_proj.weight"
    assert shapes[check_key] == (2048, 2048), f"wrong check tensor shape: {shapes[check_key]}"
    for base_key, (a_key, b_key) in pairs.items():
        a_shape, a_dtype = tensor_info(adapter_path, a_key)
        b_shape, b_dtype = tensor_info(adapter_path, b_key)
        assert shapes[base_key] == (2048, 2048)
        assert (a_shape, b_shape) == ((16, 2048), (2048, 16))
        assert dtypes[base_key] == a_dtype == b_dtype == torch.float32

    # Greedy sharding, with the two large embedding matrices explicitly isolated.
    plans, current, current_size = [], [], 0
    for key in weight_map:
        size = sizes[key]
        assert size <= CAP, f"single tensor exceeds shard cap: {key} ({size} bytes)"
        if key in FORCE_ALONE:
            if current:
                plans.append(current)
                current, current_size = [], 0
            plans.append([key])
        else:
            if current and current_size + size > CAP:
                plans.append(current)
                current, current_size = [], 0
            current.append(key)
            current_size += size
    if current:
        plans.append(current)

    count = len(plans)
    output_map = {}
    for number, keys in enumerate(plans, 1):
        filename = f"model-{number:05d}-of-{count:05d}.safetensors"
        tensors = {}
        with safe_open(adapter_path, framework="pt", device="cpu") as adapter:
            for key in keys:
                source = BASE / weight_map[key]
                with safe_open(source, framework="pt", device="cpu") as base:
                    value = base.get_tensor(key)
                if key in pairs:
                    a_key, b_key = pairs[key]
                    a = adapter.get_tensor(a_key)
                    b = adapter.get_tensor(b_key)
                    value = value + scale * torch.matmul(b, a)
                tensors[key] = value.contiguous()
                output_map[key] = filename
        assert sum(t.numel() * t.element_size() for t in tensors.values()) <= CAP
        save_file(tensors, OUT / filename)

    assert set(output_map) == set(weight_map) and len(output_map) == 114
    output_index = {
        "metadata": {"total_size": sum(sizes.values())},
        "weight_map": output_map,
    }
    (OUT / "model.safetensors.index.json").write_text(
        json.dumps(output_index, indent=2, sort_keys=True) + "\n"
    )
    print(f"Merged {len(pairs)} LoRA pairs into 114 tensors across {count} shards")


if __name__ == "__main__":
    main()
