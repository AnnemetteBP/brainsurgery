#!/usr/bin/env python3
"""Merge the supplied GPT-2 LoRA adapter and export 100 MiB shards."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


MAX_SHARD_BYTES = 100 * 1024 * 1024
ROOT = Path(__file__).resolve().parents[2]
BASE_PATH = ROOT / "inputs/base/model.safetensors"
ADAPTER_PATH = ROOT / "inputs/lora/adapter_model.safetensors"
CONFIG_PATH = ROOT / "inputs/lora/adapter_config.json"
OUTPUT_DIR = Path(__file__).resolve().parent


def tensor_info(path: Path) -> tuple[list[str], dict[str, tuple[int, ...]], dict[str, str]]:
    """Read names, shapes, and safetensors dtype codes without loading all data."""
    with safe_open(path, framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        shapes = {key: tuple(handle.get_slice(key).get_shape()) for key in keys}
        dtypes = {key: handle.get_slice(key).get_dtype() for key in keys}
    return keys, shapes, dtypes


def main() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    rank = int(config["r"])
    alpha = float(config["lora_alpha"])
    assert rank > 0, "adapter rank must be positive"
    assert config.get("fan_in_fan_out") is True, "this merge expects fan_in_fan_out=true"
    scale = alpha / rank

    base_keys, base_shapes, base_dtypes = tensor_info(BASE_PATH)
    adapter_keys, adapter_shapes, adapter_dtypes = tensor_info(ADAPTER_PATH)

    prefix = "base_model.model."
    a_suffix = ".lora_A.weight"
    b_suffix = ".lora_B.weight"
    a_keys = sorted(key for key in adapter_keys if key.endswith(a_suffix))
    pairs: dict[str, tuple[str, str]] = {}
    consumed_adapter_keys: set[str] = set()

    for a_key in a_keys:
        assert a_key.startswith(prefix), f"unexpected adapter prefix: {a_key}"
        stem = a_key[: -len(a_suffix)]
        b_key = stem + b_suffix
        assert b_key in adapter_shapes, f"missing B factor for {a_key}"
        base_key = stem[len(prefix) :] + ".weight"
        assert base_key in base_shapes, f"missing base tensor for {a_key}: {base_key}"
        assert base_key not in pairs, f"duplicate adapter pair for {base_key}"
        pairs[base_key] = (a_key, b_key)
        consumed_adapter_keys.update((a_key, b_key))

    # Required pre-write checks, plus complete pairing/layout validation.
    assert len(pairs) == 12, f"expected exactly 12 adapter pairs, found {len(pairs)}"
    assert consumed_adapter_keys == set(adapter_keys), "unpaired or unexpected adapter tensors found"
    assert len(base_keys) == 160, f"expected exactly 160 output tensors, found {len(base_keys)}"
    assert not any("lora_" in key for key in base_keys), "LoRA tensor name would enter output"
    assert base_shapes.get("h.0.attn.c_attn.weight") == (768, 2304), (
        "h.0.attn.c_attn.weight does not have shape [768, 2304]"
    )
    assert all(dtype == "F32" for dtype in base_dtypes.values()), "base tensors must be float32"
    assert all(dtype == "F32" for dtype in adapter_dtypes.values()), "adapter tensors must be float32"

    for base_key, (a_key, b_key) in pairs.items():
        in_features, out_features = base_shapes[base_key]
        assert adapter_shapes[a_key] == (rank, in_features), f"bad A shape for {base_key}"
        assert adapter_shapes[b_key] == (out_features, rank), f"bad B shape for {base_key}"

    tensor_sizes = {
        key: torch.Size(base_shapes[key]).numel() * 4
        for key in base_keys
    }

    # Greedy deterministic packing. Oversized tensors become singleton shards.
    shard_key_groups: list[list[str]] = []
    current: list[str] = []
    current_size = 0
    for key in sorted(base_keys):
        size = tensor_sizes[key]
        if size > MAX_SHARD_BYTES:
            if current:
                shard_key_groups.append(current)
                current, current_size = [], 0
            shard_key_groups.append([key])
        else:
            if current and current_size + size > MAX_SHARD_BYTES:
                shard_key_groups.append(current)
                current, current_size = [], 0
            current.append(key)
            current_size += size
    if current:
        shard_key_groups.append(current)

    assert sum(map(len, shard_key_groups)) == 160, "shard plan lost output tensors"
    for group in shard_key_groups:
        size = sum(tensor_sizes[key] for key in group)
        assert size <= MAX_SHARD_BYTES or (
            len(group) == 1 and tensor_sizes[group[0]] > MAX_SHARD_BYTES
        ), f"invalid shard plan ({size} tensor bytes)"

    for old_shard in OUTPUT_DIR.glob("model-*-of-*.safetensors"):
        old_shard.unlink()
    index_path = OUTPUT_DIR / "model.safetensors.index.json"
    if index_path.exists():
        index_path.unlink()

    shard_count = len(shard_key_groups)
    weight_map: dict[str, str] = {}
    with safe_open(BASE_PATH, framework="pt", device="cpu") as base_file, safe_open(
        ADAPTER_PATH, framework="pt", device="cpu"
    ) as adapter_file:
        for shard_number, group in enumerate(shard_key_groups, start=1):
            shard_name = f"model-{shard_number:05d}-of-{shard_count:05d}.safetensors"
            tensors: dict[str, torch.Tensor] = {}
            for key in group:
                base_tensor = base_file.get_tensor(key)
                if key in pairs:
                    a_key, b_key = pairs[key]
                    a = adapter_file.get_tensor(a_key)
                    b = adapter_file.get_tensor(b_key)
                    delta = (b @ a).T.mul(scale)
                    merged = base_tensor + delta
                    assert merged.shape == base_tensor.shape, f"merged shape changed for {key}"
                    assert merged.dtype == torch.float32, f"merged dtype changed for {key}"
                    tensors[key] = merged.contiguous()
                else:
                    tensors[key] = base_tensor.contiguous()
                weight_map[key] = shard_name
            save_file(tensors, OUTPUT_DIR / shard_name, metadata={"format": "pt"})

    assert len(weight_map) == 160, "written weight map does not contain 160 tensors"
    assert not any("lora_" in key for key in weight_map), "LoRA tensor entered written weight map"
    index = {
        "metadata": {"total_size": sum(tensor_sizes.values())},
        "weight_map": weight_map,
    }
    index_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Merged {len(pairs)} LoRA pairs into 160 tensors across {shard_count} shards")


if __name__ == "__main__":
    main()
