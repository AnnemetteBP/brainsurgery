#!/usr/bin/env python3
"""Merge the supplied PEFT LoRA weights into GPT-2 and export shards."""

import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


ROOT = Path(__file__).resolve().parents[2]
BASE_PATH = ROOT / "inputs/base/model.safetensors"
ADAPTER_PATH = ROOT / "inputs/lora/adapter_model.safetensors"
CONFIG_PATH = ROOT / "inputs/lora/adapter_config.json"
OUTPUT_DIR = ROOT / "out/T5"
MAX_SHARD_BYTES = 100 * 1024 * 1024
ADAPTER_PREFIX = "base_model.model."


def tensor_bytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def main() -> None:
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    rank = int(config["r"])
    scale = float(config["lora_alpha"]) / rank
    fan_in_fan_out = bool(config["fan_in_fan_out"])

    base = load_file(str(BASE_PATH), device="cpu")
    adapter = load_file(str(ADAPTER_PATH), device="cpu")

    a_suffix = ".lora_A.weight"
    a_names = sorted(name for name in adapter if name.endswith(a_suffix))
    pairs = []
    for a_name in a_names:
        stem = a_name[: -len(a_suffix)]
        b_name = stem + ".lora_B.weight"
        if b_name not in adapter:
            raise AssertionError(f"missing LoRA B tensor for {a_name}")
        if not stem.startswith(ADAPTER_PREFIX):
            raise AssertionError(f"unexpected adapter prefix: {a_name}")
        base_name = stem[len(ADAPTER_PREFIX) :] + ".weight"
        if base_name not in base:
            raise AssertionError(f"mapped base tensor does not exist: {base_name}")
        pairs.append((a_name, b_name, base_name))

    paired_adapter_names = {name for a, b, _ in pairs for name in (a, b)}
    if paired_adapter_names != set(adapter):
        extras = sorted(set(adapter) - paired_adapter_names)
        raise AssertionError(f"unpaired or unexpected adapter tensors: {extras}")
    if len(pairs) != 12:
        raise AssertionError(f"expected exactly 12 adapter pairs, found {len(pairs)}")

    for a_name, b_name, base_name in pairs:
        a = adapter[a_name].to(torch.float32)
        b = adapter[b_name].to(torch.float32)
        if a.shape[0] != rank or b.shape[1] != rank:
            raise AssertionError(
                f"adapter rank mismatch for {base_name}: A={tuple(a.shape)}, B={tuple(b.shape)}"
            )
        update = b @ a
        if fan_in_fan_out:
            update = update.T
        weight = base[base_name]
        if weight.dtype != torch.float32 or update.shape != weight.shape:
            raise AssertionError(
                f"shape/dtype mismatch for {base_name}: weight={tuple(weight.shape)}/{weight.dtype}, "
                f"update={tuple(update.shape)}/{update.dtype}"
            )
        base[base_name] = weight + update.mul(scale)

    # Required pre-write checks.
    if len(pairs) != 12:
        raise AssertionError("exactly 12 adapter pairs must be merged")
    if any("lora_" in name for name in base):
        raise AssertionError("LoRA tensor found in dense output")
    if tuple(base["h.0.attn.c_attn.weight"].shape) != (768, 2304):
        raise AssertionError("h.0.attn.c_attn.weight has the wrong shape")
    if len(base) != 160:
        raise AssertionError(f"expected 160 output tensors, found {len(base)}")

    shards = []
    current = {}
    current_bytes = 0
    for name, tensor in base.items():
        size = tensor_bytes(tensor)
        if current and current_bytes + size > MAX_SHARD_BYTES:
            shards.append(current)
            current = {}
            current_bytes = 0
        current[name] = tensor
        current_bytes += size
        if size > MAX_SHARD_BYTES:
            shards.append(current)
            current = {}
            current_bytes = 0
    if current:
        shards.append(current)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for old_shard in OUTPUT_DIR.glob("model-*-of-*.safetensors"):
        old_shard.unlink()

    weight_map = {}
    shard_count = len(shards)
    for number, shard in enumerate(shards, start=1):
        shard_size = sum(tensor_bytes(tensor) for tensor in shard.values())
        oversized = len(shard) == 1 and shard_size > MAX_SHARD_BYTES
        if shard_size > MAX_SHARD_BYTES and not oversized:
            raise AssertionError(f"shard {number} exceeds the size limit")
        filename = f"model-{number:05d}-of-{shard_count:05d}.safetensors"
        save_file(shard, str(OUTPUT_DIR / filename))
        weight_map.update({name: filename for name in shard})

    index = {
        "metadata": {"total_size": sum(tensor_bytes(tensor) for tensor in base.values())},
        "weight_map": weight_map,
    }
    with (OUTPUT_DIR / "model.safetensors.index.json").open("w", encoding="utf-8") as handle:
        json.dump(index, handle, indent=2, sort_keys=True)
        handle.write("\n")


if __name__ == "__main__":
    main()
