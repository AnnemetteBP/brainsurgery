#!/usr/bin/env python3
"""Merge the supplied PEFT LoRA into GPT-2 and export safetensors shards."""

import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
BASE_PATH = ROOT / "inputs/base/model.safetensors"
ADAPTER_PATH = ROOT / "inputs/lora/adapter_model.safetensors"
CONFIG_PATH = ROOT / "inputs/lora/adapter_config.json"
MAX_SHARD_BYTES = 100 * 1024 * 1024


def tensor_bytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def main() -> None:
    config = json.loads(CONFIG_PATH.read_text())
    base = load_file(BASE_PATH, device="cpu")
    adapter = load_file(ADAPTER_PATH, device="cpu")

    r = config["r"]
    alpha = config["lora_alpha"]
    assert r == 16 and alpha == 32, f"unexpected LoRA parameters: r={r}, alpha={alpha}"
    assert config.get("fan_in_fan_out") is True, "this merge requires fan_in_fan_out=true"
    scale = alpha / r

    a_suffix = ".lora_A.weight"
    a_names = sorted(name for name in adapter if name.endswith(a_suffix))
    merged_names = []
    for a_name in a_names:
        stem = a_name[: -len(a_suffix)]
        b_name = stem + ".lora_B.weight"
        assert b_name in adapter, f"missing B factor for {a_name}"
        prefix = "base_model.model."
        assert stem.startswith(prefix), f"unexpected adapter key: {a_name}"
        base_name = stem[len(prefix) :] + ".weight"
        assert base_name in base, f"missing base tensor for {a_name}: {base_name}"

        a = adapter[a_name]
        b = adapter[b_name]
        assert a.dtype == torch.float32 and b.dtype == torch.float32
        delta = (b @ a).T.mul(scale)
        assert delta.shape == base[base_name].shape, (
            f"shape mismatch for {base_name}: {tuple(delta.shape)} vs "
            f"{tuple(base[base_name].shape)}"
        )
        base[base_name] = base[base_name] + delta
        merged_names.append(base_name)

    # Required pre-write checks.
    adapter_pair_count = len(a_names)
    assert adapter_pair_count == 12, f"expected 12 adapter pairs, found {adapter_pair_count}"
    assert len(adapter) == 2 * adapter_pair_count, "unpaired or unexpected adapter tensors found"
    assert len(set(merged_names)) == 12, "adapter pairs did not map to 12 unique base tensors"
    assert not any("lora_" in name for name in base), "LoRA tensor leaked into output"
    assert tuple(base["h.0.attn.c_attn.weight"].shape) == (768, 2304)
    assert len(base) == 160, f"expected 160 output tensors, found {len(base)}"

    # Greedily pack tensors in base-file key order. Oversized tensors form a
    # legal one-tensor shard; ordinary shards never exceed MAX_SHARD_BYTES.
    shards: list[dict[str, torch.Tensor]] = []
    current: dict[str, torch.Tensor] = {}
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
            assert len(current) == 1
            shards.append(current)
            current = {}
            current_bytes = 0
    if current:
        shards.append(current)

    for old in OUT.glob("model-*-of-*.safetensors"):
        old.unlink()
    index_path = OUT / "model.safetensors.index.json"
    if index_path.exists():
        index_path.unlink()

    weight_map: dict[str, str] = {}
    shard_count = len(shards)
    for number, shard in enumerate(shards, start=1):
        filename = f"model-{number:05d}-of-{shard_count:05d}.safetensors"
        shard_size = sum(tensor_bytes(tensor) for tensor in shard.values())
        assert shard_size <= MAX_SHARD_BYTES or len(shard) == 1
        save_file(shard, OUT / filename, metadata={"format": "pt"})
        weight_map.update({name: filename for name in shard})

    index = {
        "metadata": {"total_size": sum(tensor_bytes(t) for t in base.values())},
        "weight_map": weight_map,
    }
    index_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
    print(f"Merged {adapter_pair_count} LoRA pairs; wrote {len(base)} tensors in {shard_count} shards")


if __name__ == "__main__":
    main()
