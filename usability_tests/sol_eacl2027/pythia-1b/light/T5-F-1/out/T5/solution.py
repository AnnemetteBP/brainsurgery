#!/usr/bin/env python3
"""Merge the supplied PEFT LoRA into Pythia and export safetensors shards."""

import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "inputs/base/model.safetensors"
ADAPTER = ROOT / "inputs/lora/adapter_model.safetensors"
CONFIG = ROOT / "inputs/lora/adapter_config.json"
OUT = ROOT / "out/T5"
MAX_SHARD_BYTES = 512 * 1024 * 1024
FORCE_SOLO = {"gpt_neox.embed_in.weight", "embed_out.weight"}


def tensor_bytes(tensor):
    return tensor.numel() * tensor.element_size()


def main():
    config = json.loads(CONFIG.read_text())
    rank = config["r"]
    scale = config["lora_alpha"] / rank
    fan_in_fan_out = config["fan_in_fan_out"]

    with safe_open(BASE, framework="pt", device="cpu") as base_file:
        base_keys = list(base_file.keys())
        shapes = {key: tuple(base_file.get_slice(key).get_shape()) for key in base_keys}
        sizes = {
            key: tensor_bytes(base_file.get_tensor(key))
            for key in base_keys
        }

    with safe_open(ADAPTER, framework="pt", device="cpu") as adapter_file:
        adapter_keys = set(adapter_file.keys())

    suffix_a = ".lora_A.weight"
    prefix = "base_model.model."
    pairs = {}
    for a_key in sorted(key for key in adapter_keys if key.endswith(suffix_a)):
        stem = a_key[: -len(suffix_a)]
        b_key = stem + ".lora_B.weight"
        if b_key not in adapter_keys:
            raise RuntimeError(f"missing LoRA B tensor for {a_key}")
        if not stem.startswith(prefix):
            raise RuntimeError(f"unexpected PEFT tensor name: {a_key}")
        base_key = stem[len(prefix):] + ".weight"
        if base_key not in shapes:
            raise RuntimeError(f"mapped base tensor does not exist: {base_key}")
        pairs[base_key] = (a_key, b_key)

    paired_adapter_keys = {key for pair in pairs.values() for key in pair}
    if paired_adapter_keys != adapter_keys:
        raise RuntimeError("adapter contains unpaired or unexpected tensors")

    # Required pre-write checks.
    if len(pairs) != 16:
        raise RuntimeError(f"expected exactly 16 adapter pairs, found {len(pairs)}")
    if any("lora_" in key for key in base_keys):
        raise RuntimeError("output key set would contain a LoRA tensor")
    probe = "gpt_neox.layers.0.attention.query_key_value.weight"
    if shapes.get(probe) != (6144, 2048):
        raise RuntimeError(f"wrong probe shape: {shapes.get(probe)}")
    if len(base_keys) != 244:
        raise RuntimeError(f"expected exactly 244 output tensors, found {len(base_keys)}")

    # Pack deterministically. The two embeddings are explicitly required to be
    # alone; all other shards use greedy packing under the byte cap.
    shards = []
    current, current_bytes = [], 0
    for key in sorted(base_keys):
        size = sizes[key]
        if size > MAX_SHARD_BYTES:
            raise RuntimeError(f"tensor {key} exceeds the shard cap")
        if key in FORCE_SOLO:
            if current:
                shards.append(current)
                current, current_bytes = [], 0
            shards.append([key])
        else:
            if current and current_bytes + size > MAX_SHARD_BYTES:
                shards.append(current)
                current, current_bytes = [], 0
            current.append(key)
            current_bytes += size
    if current:
        shards.append(current)

    weight_map = {}
    total = len(shards)
    with safe_open(BASE, framework="pt", device="cpu") as base_file, safe_open(
        ADAPTER, framework="pt", device="cpu"
    ) as adapter_file:
        for number, keys in enumerate(shards, 1):
            filename = f"model-{number:05d}-of-{total:05d}.safetensors"
            tensors = {}
            for key in keys:
                base = base_file.get_tensor(key)
                if key in pairs:
                    a_key, b_key = pairs[key]
                    a = adapter_file.get_tensor(a_key).float()
                    b = adapter_file.get_tensor(b_key).float()
                    delta = b @ a
                    if fan_in_fan_out:
                        delta = delta.T
                    if tuple(delta.shape) != tuple(base.shape):
                        raise RuntimeError(f"LoRA delta shape mismatch for {key}")
                    base = (base.float() + scale * delta).to(base.dtype)
                tensors[key] = base.contiguous()
                weight_map[key] = filename
            actual_bytes = sum(tensor_bytes(t) for t in tensors.values())
            if actual_bytes > MAX_SHARD_BYTES:
                raise RuntimeError(f"shard {filename} exceeds the byte cap")
            save_file(tensors, OUT / filename, metadata={"format": "pt"})

    index = {
        "metadata": {"total_size": sum(sizes.values())},
        "weight_map": weight_map,
    }
    (OUT / "model.safetensors.index.json").write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n"
    )
    print(f"Merged {len(pairs)} LoRA pairs; wrote {len(shards)} shards and {len(weight_map)} tensors")


if __name__ == "__main__":
    main()
