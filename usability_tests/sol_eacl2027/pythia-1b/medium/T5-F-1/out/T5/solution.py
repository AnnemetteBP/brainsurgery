#!/usr/bin/env python3
"""Merge the supplied PEFT LoRA into Pythia-1B and export safetensor shards."""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
BASE_PATH = ROOT / "inputs/base/model.safetensors"
ADAPTER_PATH = ROOT / "inputs/lora/adapter_model.safetensors"
CONFIG_PATH = ROOT / "inputs/lora/adapter_config.json"
INDEX_PATH = OUT / "model.safetensors.index.json"
MAX_SHARD_BYTES = 512 * 1024 * 1024
FORCE_ALONE = {"gpt_neox.embed_in.weight", "embed_out.weight"}
EXPECTED_TARGET = "gpt_neox.layers.0.attention.query_key_value.weight"
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


def tensor_bytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def inspect_base() -> tuple[list[str], dict[str, tuple[tuple[int, ...], str, int]]]:
    with safe_open(BASE_PATH, framework="pt", device="cpu") as handle:
        keys = sorted(handle.keys())
        info = {}
        for key in keys:
            tensor = handle.get_slice(key)
            shape = tuple(tensor.get_shape())
            dtype = tensor.get_dtype()
            assert dtype in DTYPE_BYTES, f"unsupported safetensors dtype for {key}: {dtype}"
            itemsize = DTYPE_BYTES[dtype]
            info[key] = (shape, dtype, itemsize * int(torch.tensor(shape).prod().item()))
    return keys, info


def adapter_base_name(adapter_a_name: str) -> str:
    prefix = "base_model.model."
    suffix = ".lora_A.weight"
    assert adapter_a_name.startswith(prefix), f"unexpected adapter prefix: {adapter_a_name}"
    assert adapter_a_name.endswith(suffix), f"unexpected LoRA A name: {adapter_a_name}"
    return adapter_a_name[len(prefix) : -len(suffix)] + ".weight"


@torch.no_grad()
def prepare_merged(
    base_info: dict[str, tuple[tuple[int, ...], str, int]],
) -> dict[str, torch.Tensor]:
    config = json.loads(CONFIG_PATH.read_text())
    rank = int(config["r"])
    alpha = float(config["lora_alpha"])
    scale = alpha / rank
    fan_in_fan_out = bool(config.get("fan_in_fan_out", False))

    with safe_open(ADAPTER_PATH, framework="pt", device="cpu") as adapter:
        adapter_keys = set(adapter.keys())
        a_names = sorted(key for key in adapter_keys if key.endswith(".lora_A.weight"))
        b_names = sorted(key for key in adapter_keys if key.endswith(".lora_B.weight"))
        assert len(a_names) == 16, f"required exactly 16 LoRA A tensors, found {len(a_names)}"
        assert len(b_names) == 16, f"required exactly 16 LoRA B tensors, found {len(b_names)}"
        assert len(adapter_keys) == 32, f"unexpected adapter tensor count: {len(adapter_keys)}"

        merged: dict[str, torch.Tensor] = {}
        with safe_open(BASE_PATH, framework="pt", device="cpu") as base:
            for a_name in a_names:
                b_name = a_name.removesuffix(".lora_A.weight") + ".lora_B.weight"
                assert b_name in adapter_keys, f"missing LoRA B partner for {a_name}"
                base_name = adapter_base_name(a_name)
                assert base_name in base_info, f"adapter target absent from base: {base_name}"
                assert base_name not in merged, f"duplicate adapter target: {base_name}"

                a = adapter.get_tensor(a_name).float()
                b = adapter.get_tensor(b_name).float()
                original = base.get_tensor(base_name)
                assert a.shape[0] == rank and b.shape[1] == rank, (
                    f"adapter rank mismatch for {base_name}: A={tuple(a.shape)}, B={tuple(b.shape)}, r={rank}"
                )
                delta = b @ a
                if fan_in_fan_out:
                    delta = delta.T
                assert tuple(delta.shape) == tuple(original.shape), (
                    f"LoRA/base shape mismatch for {base_name}: delta={tuple(delta.shape)}, "
                    f"base={tuple(original.shape)}"
                )
                merged[base_name] = (original.float() + scale * delta).to(original.dtype).contiguous()

    # All mandated checks occur before the first output checkpoint file is written.
    assert len(merged) == 16, f"required exactly 16 adapter pairs merged, got {len(merged)}"
    output_names = set(base_info)
    assert not any("lora_" in name for name in output_names), "LoRA tensor leaked into output key set"
    assert EXPECTED_TARGET in base_info, f"required target missing: {EXPECTED_TARGET}"
    assert base_info[EXPECTED_TARGET][0] == (6144, 2048), (
        f"required target has shape {base_info[EXPECTED_TARGET][0]}, expected (6144, 2048)"
    )
    assert len(output_names) == 244, f"required exactly 244 output tensors, got {len(output_names)}"
    return merged


def plan_shards(keys: list[str], info: dict[str, tuple[tuple[int, ...], str, int]]) -> list[list[str]]:
    shards: list[list[str]] = []
    current: list[str] = []
    current_bytes = 0

    def flush() -> None:
        nonlocal current, current_bytes
        if current:
            shards.append(current)
            current = []
            current_bytes = 0

    for key in keys:
        size = info[key][2]
        assert size <= MAX_SHARD_BYTES, f"single tensor exceeds 512 MiB: {key} ({size} bytes)"
        if key in FORCE_ALONE:
            flush()
            shards.append([key])
        else:
            if current and current_bytes + size > MAX_SHARD_BYTES:
                flush()
            current.append(key)
            current_bytes += size
    flush()
    return shards


def write_checkpoint(
    shards: list[list[str]], merged: dict[str, torch.Tensor], info: dict[str, tuple[tuple[int, ...], str, int]]
) -> None:
    # Remove only files produced by an earlier invocation of this exporter.
    for old_shard in OUT.glob("model-*-of-*.safetensors"):
        old_shard.unlink()
    INDEX_PATH.unlink(missing_ok=True)

    weight_map: dict[str, str] = {}
    total = len(shards)
    with safe_open(BASE_PATH, framework="pt", device="cpu") as base:
        for number, shard_keys in enumerate(shards, start=1):
            filename = f"model-{number:05d}-of-{total:05d}.safetensors"
            tensors: OrderedDict[str, torch.Tensor] = OrderedDict()
            for key in shard_keys:
                tensors[key] = merged[key] if key in merged else base.get_tensor(key)
                weight_map[key] = filename
            shard_bytes = sum(tensor_bytes(tensor) for tensor in tensors.values())
            assert shard_bytes <= MAX_SHARD_BYTES, f"oversized shard {filename}: {shard_bytes} bytes"
            save_file(tensors, OUT / filename, metadata={"format": "pt"})

    index = {
        "metadata": {"total_size": sum(entry[2] for entry in info.values())},
        "weight_map": weight_map,
    }
    INDEX_PATH.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")


def verify_export(keys: list[str], info: dict[str, tuple[tuple[int, ...], str, int]]) -> None:
    index = json.loads(INDEX_PATH.read_text())
    weight_map = index["weight_map"]
    assert set(weight_map) == set(keys), "index key set differs from base key set"
    assert len(weight_map) == 244, f"index contains {len(weight_map)} tensors, expected 244"
    assert not any("lora_" in key for key in weight_map), "index contains a LoRA tensor"

    seen: set[str] = set()
    for filename in sorted(set(weight_map.values())):
        path = OUT / filename
        assert path.is_file(), f"index references missing shard: {filename}"
        expected = {key for key, mapped_file in weight_map.items() if mapped_file == filename}
        with safe_open(path, framework="pt", device="cpu") as shard:
            actual = set(shard.keys())
            assert actual == expected, f"shard/index disagreement for {filename}"
            shard_bytes = 0
            for key in actual:
                tensor = shard.get_slice(key)
                shape = tuple(tensor.get_shape())
                dtype = tensor.get_dtype()
                assert shape == info[key][0], f"shape changed for {key}: {shape}"
                assert dtype == info[key][1], f"dtype changed for {key}: {dtype}"
                shard_bytes += info[key][2]
            assert shard_bytes <= MAX_SHARD_BYTES, f"oversized output shard {filename}: {shard_bytes} bytes"
            seen.update(actual)
    assert seen == set(keys), "not all output tensors were found exactly once"
    for key in FORCE_ALONE:
        filename = weight_map[key]
        assert sum(value == filename for value in weight_map.values()) == 1, f"{key} is not alone in its shard"


def main() -> None:
    torch.set_grad_enabled(False)
    keys, info = inspect_base()
    merged = prepare_merged(info)
    shards = plan_shards(keys, info)
    write_checkpoint(shards, merged, info)
    verify_export(keys, info)
    print(f"Merged {len(merged)} LoRA pairs and wrote {len(keys)} tensors across {len(shards)} shards.")


if __name__ == "__main__":
    main()
