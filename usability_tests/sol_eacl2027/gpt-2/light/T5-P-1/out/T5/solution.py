#!/usr/bin/env python3
import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


ROOT = Path(__file__).resolve().parents[2]
BASE_PATH = ROOT / "inputs/base/model.safetensors"
ADAPTER_PATH = ROOT / "inputs/lora/adapter_model.safetensors"
CONFIG_PATH = ROOT / "inputs/lora/adapter_config.json"
OUT_DIR = ROOT / "out/T5"
MAX_SHARD_BYTES = 100 * 1024 * 1024


def tensor_bytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def main() -> None:
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    rank = config["r"]
    scale = config["lora_alpha"] / rank
    if not config.get("fan_in_fan_out", False):
        raise ValueError("This task requires fan_in_fan_out=true")

    output = load_file(str(BASE_PATH), device="cpu")
    adapter = load_file(str(ADAPTER_PATH), device="cpu")

    a_suffix = ".lora_A.weight"
    prefix = "base_model.model."
    a_names = sorted(name for name in adapter if name.endswith(a_suffix))
    merged = 0

    for a_name in a_names:
        stem = a_name[: -len(a_suffix)]
        b_name = stem + ".lora_B.weight"
        if b_name not in adapter:
            raise KeyError(f"Missing matching LoRA B tensor for {a_name}")
        if not stem.startswith(prefix):
            raise ValueError(f"Unexpected adapter tensor name: {a_name}")

        base_name = stem[len(prefix) :] + ".weight"
        if base_name not in output:
            raise KeyError(f"Missing base tensor {base_name}")

        a = adapter[a_name]
        b = adapter[b_name]
        if a.dtype != torch.float32 or b.dtype != torch.float32:
            raise TypeError(f"Adapter tensors for {base_name} are not float32")
        delta = (b @ a).T.mul(scale)
        if delta.shape != output[base_name].shape:
            raise ValueError(
                f"Shape mismatch for {base_name}: {tuple(output[base_name].shape)} "
                f"vs delta {tuple(delta.shape)}"
            )
        output[base_name] = output[base_name] + delta
        merged += 1

    expected_adapter_names = {
        name
        for a_name in a_names
        for name in (a_name, a_name[: -len(a_suffix)] + ".lora_B.weight")
    }
    if expected_adapter_names != set(adapter):
        unexpected = sorted(set(adapter) - expected_adapter_names)
        raise ValueError(f"Unexpected or unpaired adapter tensors: {unexpected}")

    # Required pre-write checks.
    if merged != 12:
        raise ValueError(f"Expected exactly 12 adapter pairs, found {merged}")
    if any("lora_" in name for name in output):
        raise ValueError("Output contains a LoRA tensor")
    if tuple(output["h.0.attn.c_attn.weight"].shape) != (768, 2304):
        raise ValueError("h.0.attn.c_attn.weight has the wrong shape")
    if len(output) != 160:
        raise ValueError(f"Expected 160 output tensors, found {len(output)}")

    shards: list[dict[str, torch.Tensor]] = []
    current: dict[str, torch.Tensor] = {}
    current_bytes = 0
    for name in sorted(output):
        size = tensor_bytes(output[name])
        if current and current_bytes + size > MAX_SHARD_BYTES:
            shards.append(current)
            current = {}
            current_bytes = 0
        current[name] = output[name].contiguous()
        current_bytes += size
        if size > MAX_SHARD_BYTES:
            shards.append(current)
            current = {}
            current_bytes = 0
    if current:
        shards.append(current)

    total = len(shards)
    weight_map: dict[str, str] = {}
    for number, shard in enumerate(shards, start=1):
        filename = f"model-{number:05d}-of-{total:05d}.safetensors"
        save_file(shard, str(OUT_DIR / filename), metadata={"format": "pt"})
        weight_map.update({name: filename for name in shard})

    index = {
        "metadata": {"total_size": sum(tensor_bytes(t) for t in output.values())},
        "weight_map": weight_map,
    }
    with (OUT_DIR / "model.safetensors.index.json").open("w", encoding="utf-8") as handle:
        json.dump(index, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(f"Merged {merged} LoRA pairs and wrote {len(output)} tensors to {total} shards")


if __name__ == "__main__":
    main()
