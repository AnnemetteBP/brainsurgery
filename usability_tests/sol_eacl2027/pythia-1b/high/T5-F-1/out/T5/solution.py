#!/usr/bin/env python3
"""Merge the supplied PEFT LoRA into the base checkpoint and shard the result."""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


BASE_PATH = Path("inputs/base/model.safetensors")
ADAPTER_PATH = Path("inputs/lora/adapter_model.safetensors")
CONFIG_PATH = Path("inputs/lora/adapter_config.json")
OUTPUT_DIR = Path("out/T5")
INDEX_PATH = OUTPUT_DIR / "model.safetensors.index.json"
MAX_SHARD_BYTES = 512 * 1024 * 1024
EXPECTED_TENSORS = 244
EXPECTED_PAIRS = 16
SHAPE_CHECK_KEY = "gpt_neox.layers.0.attention.query_key_value.weight"
SINGLETON_SHARD_KEYS = {"gpt_neox.embed_in.weight", "embed_out.weight"}

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
    "I64": 8,
    "U64": 8,
    "F64": 8,
}


def tensor_nbytes(handle, name: str) -> int:
    view = handle.get_slice(name)
    dtype = str(view.get_dtype())
    assert dtype in DTYPE_BYTES, f"unsupported safetensors dtype {dtype} for {name}"
    return math.prod(view.get_shape()) * DTYPE_BYTES[dtype]


def discover_pairs(adapter_keys: set[str], base_keys: set[str]) -> dict[str, tuple[str, str]]:
    prefix = "base_model.model."
    a_suffix = ".lora_A.weight"
    pairs: dict[str, tuple[str, str]] = {}

    a_keys = {key for key in adapter_keys if key.endswith(a_suffix)}
    b_keys = {key for key in adapter_keys if key.endswith(".lora_B.weight")}
    assert len(a_keys) == EXPECTED_PAIRS, f"expected 16 lora_A tensors, found {len(a_keys)}"
    assert len(b_keys) == EXPECTED_PAIRS, f"expected 16 lora_B tensors, found {len(b_keys)}"

    for a_key in sorted(a_keys):
        assert a_key.startswith(prefix), f"unexpected adapter prefix: {a_key}"
        stem = a_key[: -len(a_suffix)]
        b_key = stem + ".lora_B.weight"
        base_key = stem.removeprefix(prefix) + ".weight"
        assert b_key in b_keys, f"missing B factor for {a_key}"
        assert base_key in base_keys, f"adapter target absent from base: {base_key}"
        assert base_key not in pairs, f"duplicate adapter target: {base_key}"
        pairs[base_key] = (a_key, b_key)

    consumed = {key for pair in pairs.values() for key in pair}
    assert consumed == adapter_keys, "adapter contains unpaired or non-LoRA tensors"
    assert len(pairs) == EXPECTED_PAIRS, f"expected 16 adapter pairs, found {len(pairs)}"
    return pairs


def plan_shards(base_handle, base_keys: list[str]) -> tuple[list[list[str]], dict[str, int]]:
    sizes = {key: tensor_nbytes(base_handle, key) for key in base_keys}
    shards: list[list[str]] = []
    current: list[str] = []
    current_size = 0

    for key in base_keys:
        size = sizes[key]
        if key in SINGLETON_SHARD_KEYS:
            if current:
                shards.append(current)
                current = []
                current_size = 0
            shards.append([key])
            continue
        assert size <= MAX_SHARD_BYTES, f"non-singleton tensor exceeds shard limit: {key}"
        if current and current_size + size > MAX_SHARD_BYTES:
            shards.append(current)
            current = []
            current_size = 0
        current.append(key)
        current_size += size

    if current:
        shards.append(current)

    assert all(
        len(shard) == 1 or sum(sizes[key] for key in shard) <= MAX_SHARD_BYTES
        for shard in shards
    ), "planned shard exceeds 512 MiB"
    assert all([key] in shards for key in SINGLETON_SHARD_KEYS), "embedding singleton plan failed"
    return shards, sizes


def main() -> None:
    config = json.loads(CONFIG_PATH.read_text())
    rank = int(config["r"])
    alpha = float(config["lora_alpha"])
    scale = alpha / rank
    fan_in_fan_out = bool(config.get("fan_in_fan_out", False))

    assert rank == 16, f"unexpected LoRA rank: {rank}"
    assert scale == 2.0, f"unexpected LoRA scale: {scale}"
    assert fan_in_fan_out is False, "this checkpoint expects nn.Linear [out, in] layout"
    assert not INDEX_PATH.exists(), f"refusing to overwrite existing output: {INDEX_PATH}"

    with safe_open(BASE_PATH, framework="pt", device="cpu") as base_handle, safe_open(
        ADAPTER_PATH, framework="pt", device="cpu"
    ) as adapter_handle:
        base_keys = sorted(base_handle.keys())
        adapter_keys = set(adapter_handle.keys())
        base_key_set = set(base_keys)
        pairs = discover_pairs(adapter_keys, base_key_set)

        # Required checks are deliberately completed before any shard is written.
        assert len(pairs) == EXPECTED_PAIRS, f"expected 16 adapter pairs, found {len(pairs)}"
        assert not any("lora_" in key for key in base_keys), "LoRA tensor leaked into output keys"
        assert SHAPE_CHECK_KEY in base_key_set, f"missing required tensor: {SHAPE_CHECK_KEY}"
        assert base_handle.get_slice(SHAPE_CHECK_KEY).get_shape() == [6144, 2048], (
            f"wrong shape for {SHAPE_CHECK_KEY}: "
            f"{base_handle.get_slice(SHAPE_CHECK_KEY).get_shape()}"
        )
        assert len(base_keys) == EXPECTED_TENSORS, (
            f"expected 244 output tensors, found {len(base_keys)}"
        )

        for base_key, (a_key, b_key) in pairs.items():
            base_shape = base_handle.get_slice(base_key).get_shape()
            a_shape = adapter_handle.get_slice(a_key).get_shape()
            b_shape = adapter_handle.get_slice(b_key).get_shape()
            expected_shape = [b_shape[0], a_shape[1]]
            assert a_shape[0] == rank and b_shape[1] == rank, (
                f"rank mismatch for {base_key}: A={a_shape}, B={b_shape}"
            )
            assert base_shape == expected_shape, (
                f"merge shape mismatch for {base_key}: base={base_shape}, B@A={expected_shape}"
            )

        shards, sizes = plan_shards(base_handle, base_keys)
        total_shards = len(shards)
        weight_map: dict[str, str] = {}

        for shard_number, shard_keys in enumerate(shards, start=1):
            filename = f"model-{shard_number:05d}-of-{total_shards:05d}.safetensors"
            tensors: dict[str, torch.Tensor] = {}
            for key in shard_keys:
                base_tensor = base_handle.get_tensor(key)
                if key in pairs:
                    a_key, b_key = pairs[key]
                    a = adapter_handle.get_tensor(a_key).to(torch.float32)
                    b = adapter_handle.get_tensor(b_key).to(torch.float32)
                    update = b @ a
                    if fan_in_fan_out:
                        update = update.T
                    merged = base_tensor.to(torch.float32).add_(update, alpha=scale)
                    base_tensor = merged.to(dtype=base_tensor.dtype)
                    assert list(base_tensor.shape) == base_handle.get_slice(key).get_shape()
                tensors[key] = base_tensor.contiguous()
                weight_map[key] = filename

            actual_bytes = sum(tensor.numel() * tensor.element_size() for tensor in tensors.values())
            planned_bytes = sum(sizes[key] for key in shard_keys)
            assert actual_bytes == planned_bytes, f"size accounting mismatch for {filename}"
            assert len(shard_keys) == 1 or actual_bytes <= MAX_SHARD_BYTES, (
                f"shard exceeds 512 MiB: {filename} has {actual_bytes} bytes"
            )
            save_file(tensors, OUTPUT_DIR / filename, metadata={"format": "pt"})

    assert len(weight_map) == EXPECTED_TENSORS
    assert set(weight_map) == set(base_keys)
    index = {
        "metadata": {"total_size": sum(sizes.values())},
        "weight_map": weight_map,
    }
    INDEX_PATH.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
    print(
        f"Merged {len(pairs)} LoRA pairs into {len(base_keys)} tensors; "
        f"wrote {len(shards)} shards and {INDEX_PATH}"
    )


if __name__ == "__main__":
    main()
