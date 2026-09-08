#!/usr/bin/env python3
"""Merge the supplied PEFT LoRA into GPT-2 and export safetensors shards."""

import json
import re
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


BASE_PATH = Path("inputs/base/model.safetensors")
ADAPTER_PATH = Path("inputs/lora/adapter_model.safetensors")
CONFIG_PATH = Path("inputs/lora/adapter_config.json")
OUTPUT_DIR = Path("out/T5")
INDEX_PATH = OUTPUT_DIR / "model.safetensors.index.json"
MAX_SHARD_BYTES = 100 * 1024 * 1024
ADAPTER_RE = re.compile(
    r"^base_model\.model\.(?P<base>.+)\.lora_(?P<factor>[AB])\.weight$"
)


def tensor_bytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def make_shards(tensors: dict[str, torch.Tensor]) -> list[dict[str, torch.Tensor]]:
    """Greedily shard sorted tensors, placing every oversized tensor alone."""
    shards: list[dict[str, torch.Tensor]] = []
    current: dict[str, torch.Tensor] = {}
    current_bytes = 0

    for name in sorted(tensors):
        tensor = tensors[name]
        size = tensor_bytes(tensor)
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
    config = json.loads(CONFIG_PATH.read_text())
    rank = config["r"]
    alpha = config["lora_alpha"]
    if not isinstance(rank, int) or rank <= 0:
        raise ValueError(f"invalid LoRA rank: {rank!r}")
    if config.get("fan_in_fan_out") is not True:
        raise ValueError("this merge expects fan_in_fan_out=true")
    scale = float(alpha) / rank

    base = load_file(BASE_PATH, device="cpu")
    adapter = load_file(ADAPTER_PATH, device="cpu")

    pairs: dict[str, dict[str, torch.Tensor]] = {}
    for adapter_name, tensor in adapter.items():
        match = ADAPTER_RE.fullmatch(adapter_name)
        if match is None:
            raise ValueError(f"unexpected adapter tensor: {adapter_name}")
        base_name = match.group("base") + ".weight"
        factor = match.group("factor")
        if factor in pairs.setdefault(base_name, {}):
            raise ValueError(f"duplicate factor {factor} for {base_name}")
        pairs[base_name][factor] = tensor

    incomplete = {name: sorted(parts) for name, parts in pairs.items()
                  if set(parts) != {"A", "B"}}
    if incomplete:
        raise ValueError(f"incomplete adapter pairs: {incomplete}")
    if len(pairs) != 12:
        raise AssertionError(f"expected exactly 12 adapter pairs, found {len(pairs)}")

    output = dict(base)
    for base_name, factors in pairs.items():
        if base_name not in output:
            raise KeyError(f"adapter target is absent from base: {base_name}")
        weight = output[base_name]
        a = factors["A"]
        b = factors["B"]
        if weight.dtype != torch.float32 or a.dtype != torch.float32 or b.dtype != torch.float32:
            raise TypeError(f"non-float32 merge inputs for {base_name}")
        if a.shape != (rank, weight.shape[0]) or b.shape != (weight.shape[1], rank):
            raise ValueError(
                f"factor shapes do not match {base_name}: "
                f"weight={tuple(weight.shape)}, A={tuple(a.shape)}, B={tuple(b.shape)}"
            )
        delta = torch.matmul(b, a).transpose(0, 1).mul_(scale)
        if delta.shape != weight.shape:
            raise AssertionError(f"delta shape mismatch for {base_name}")
        output[base_name] = weight + delta

    # Required pre-write checks.
    if len(pairs) != 12:
        raise AssertionError("exactly 12 adapter pairs must be merged")
    if any("lora_" in name for name in output):
        raise AssertionError("LoRA tensor name leaked into dense output")
    expected_shape = (768, 2304)
    actual_shape = tuple(output["h.0.attn.c_attn.weight"].shape)
    if actual_shape != expected_shape:
        raise AssertionError(f"h.0 c_attn shape is {actual_shape}, expected {expected_shape}")
    if len(output) != 160:
        raise AssertionError(f"expected exactly 160 output tensors, found {len(output)}")

    shards = make_shards(output)
    for shard in shards:
        shard_size = sum(tensor_bytes(tensor) for tensor in shard.values())
        if shard_size > MAX_SHARD_BYTES and len(shard) != 1:
            raise AssertionError("multi-tensor shard exceeds 100 MiB")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for stale in OUTPUT_DIR.glob("model-*-of-*.safetensors"):
        stale.unlink()
    if INDEX_PATH.exists():
        INDEX_PATH.unlink()

    weight_map: dict[str, str] = {}
    shard_count = len(shards)
    for number, shard in enumerate(shards, start=1):
        filename = f"model-{number:05d}-of-{shard_count:05d}.safetensors"
        save_file(shard, OUTPUT_DIR / filename)
        weight_map.update({name: filename for name in shard})

    index = {
        "metadata": {"total_size": sum(tensor_bytes(t) for t in output.values())},
        "weight_map": weight_map,
    }
    INDEX_PATH.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
    print(f"Merged {len(pairs)} adapter pairs into {len(output)} tensors; wrote {shard_count} shards")


if __name__ == "__main__":
    main()
