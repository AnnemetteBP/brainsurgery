#!/usr/bin/env python3
"""Merge a PEFT LoRA adapter into GPT-2 and export sharded safetensors."""

import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


ROOT = Path(__file__).resolve().parents[2]
BASE_PATH = ROOT / "inputs/base/model.safetensors"
ADAPTER_PATH = ROOT / "inputs/lora/adapter_model.safetensors"
CONFIG_PATH = ROOT / "inputs/lora/adapter_config.json"
OUTPUT_DIR = ROOT / "out/T5"
INDEX_PATH = OUTPUT_DIR / "model.safetensors.index.json"
MAX_SHARD_BYTES = 100 * 1024 * 1024


def tensor_nbytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def make_shards(state: dict[str, torch.Tensor]) -> list[dict[str, torch.Tensor]]:
    """Greedily pack sorted tensors, except that oversized tensors stand alone."""
    shards: list[dict[str, torch.Tensor]] = []
    current: dict[str, torch.Tensor] = {}
    current_bytes = 0

    for name in sorted(state):
        tensor = state[name]
        size = tensor_nbytes(tensor)

        if size > MAX_SHARD_BYTES:
            if current:
                shards.append(current)
                current = {}
                current_bytes = 0
            shards.append({name: tensor})
        else:
            if current and current_bytes + size > MAX_SHARD_BYTES:
                shards.append(current)
                current = {}
                current_bytes = 0
            current[name] = tensor
            current_bytes += size

    if current:
        shards.append(current)
    return shards


def main() -> None:
    with CONFIG_PATH.open(encoding="utf-8") as handle:
        config = json.load(handle)

    rank = config["r"]
    alpha = config["lora_alpha"]
    if not isinstance(rank, int) or rank <= 0:
        raise ValueError(f"invalid LoRA rank: {rank!r}")
    if config.get("fan_in_fan_out") is not True:
        raise ValueError("this checkpoint requires fan_in_fan_out=true")
    scale = float(alpha) / rank

    base = load_file(str(BASE_PATH), device="cpu")
    adapter = load_file(str(ADAPTER_PATH), device="cpu")

    a_suffix = ".lora_A.weight"
    b_suffix = ".lora_B.weight"
    prefix = "base_model.model."
    a_stems = {name[: -len(a_suffix)] for name in adapter if name.endswith(a_suffix)}
    b_stems = {name[: -len(b_suffix)] for name in adapter if name.endswith(b_suffix)}
    if a_stems != b_stems:
        raise ValueError(
            f"unpaired adapter tensors: A-only={sorted(a_stems - b_stems)}, "
            f"B-only={sorted(b_stems - a_stems)}"
        )
    if len(adapter) != 2 * len(a_stems):
        raise ValueError("adapter contains tensors other than paired LoRA A/B weights")

    merged_count = 0
    for stem in sorted(a_stems):
        if not stem.startswith(prefix):
            raise ValueError(f"unexpected adapter prefix: {stem}")
        base_name = stem[len(prefix) :] + ".weight"
        if base_name not in base:
            raise KeyError(f"adapter target missing from base: {base_name}")

        a = adapter[stem + a_suffix]
        b = adapter[stem + b_suffix]
        weight = base[base_name]
        if a.dtype != torch.float32 or b.dtype != torch.float32 or weight.dtype != torch.float32:
            raise TypeError(f"non-float32 merge inputs for {base_name}")
        if a.ndim != 2 or b.ndim != 2 or a.shape[0] != rank or b.shape[1] != rank:
            raise ValueError(f"invalid LoRA factor shapes for {base_name}: A={a.shape}, B={b.shape}")

        delta = (b @ a).T
        if delta.shape != weight.shape:
            raise ValueError(
                f"LoRA delta shape mismatch for {base_name}: {delta.shape} vs {weight.shape}"
            )
        base[base_name] = weight + delta * scale
        merged_count += 1

    # Required pre-write checks.
    if merged_count != 12:
        raise AssertionError(f"expected exactly 12 merged adapter pairs, got {merged_count}")
    if any("lora_" in name for name in base):
        raise AssertionError("output state contains a LoRA tensor name")
    expected_shape = (768, 2304)
    actual_shape = tuple(base["h.0.attn.c_attn.weight"].shape)
    if actual_shape != expected_shape:
        raise AssertionError(f"h.0.attn.c_attn.weight shape is {actual_shape}, expected {expected_shape}")
    if len(base) != 160:
        raise AssertionError(f"expected exactly 160 output tensors, got {len(base)}")

    shards = make_shards(base)
    for shard in shards:
        shard_bytes = sum(tensor_nbytes(tensor) for tensor in shard.values())
        if shard_bytes > MAX_SHARD_BYTES and len(shard) != 1:
            raise AssertionError("an oversized shard contains more than one tensor")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for old_path in OUTPUT_DIR.glob("model-*-of-*.safetensors"):
        old_path.unlink()
    if INDEX_PATH.exists():
        INDEX_PATH.unlink()

    weight_map: dict[str, str] = {}
    shard_count = len(shards)
    for number, shard in enumerate(shards, start=1):
        filename = f"model-{number:05d}-of-{shard_count:05d}.safetensors"
        save_file(shard, str(OUTPUT_DIR / filename))
        weight_map.update({name: filename for name in shard})

    index = {
        "metadata": {"total_size": sum(tensor_nbytes(tensor) for tensor in base.values())},
        "weight_map": weight_map,
    }
    with INDEX_PATH.open("w", encoding="utf-8") as handle:
        json.dump(index, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(f"Merged {merged_count} LoRA pairs and wrote {len(base)} tensors in {shard_count} shards")


if __name__ == "__main__":
    main()
