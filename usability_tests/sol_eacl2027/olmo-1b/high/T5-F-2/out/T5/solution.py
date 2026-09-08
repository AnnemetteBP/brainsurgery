#!/usr/bin/env python3
"""Merge the supplied PEFT LoRA into the base checkpoint and shard the result."""

from __future__ import annotations

import json
from contextlib import ExitStack
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


ROOT = Path(__file__).resolve().parents[2]
BASE_DIR = ROOT / "inputs" / "base"
LORA_DIR = ROOT / "inputs" / "lora"
OUT_DIR = ROOT / "out" / "T5"
BASE_INDEX = BASE_DIR / "model.safetensors.index.json"
ADAPTER_FILE = LORA_DIR / "adapter_model.safetensors"
ADAPTER_CONFIG = LORA_DIR / "adapter_config.json"
INDEX_NAME = "model.safetensors.index.json"
MAX_SHARD_BYTES = 536_870_912
FORCE_SINGLETON = {"model.embed_tokens.weight", "lm_head.weight"}


def tensor_nbytes(shape: list[int] | tuple[int, ...], dtype: torch.dtype) -> int:
    return int(torch.tensor([], dtype=dtype).element_size()) * int(
        torch.tensor(shape).prod().item()
    )


def base_name_from_adapter(a_name: str) -> tuple[str, str]:
    prefix = "base_model.model."
    if not a_name.startswith(prefix):
        raise AssertionError(f"unexpected adapter prefix: {a_name}")
    short = a_name.removeprefix(prefix)
    if short.endswith(".lora_A.weight"):
        return short.removesuffix(".lora_A.weight") + ".weight", "A"
    if short.endswith(".lora_B.weight"):
        return short.removesuffix(".lora_B.weight") + ".weight", "B"
    raise AssertionError(f"unexpected adapter tensor name: {a_name}")


def preflight() -> tuple[
    dict[str, str], dict[str, dict[str, str]], float, dict[str, tuple[list[int], torch.dtype, int]]
]:
    """Validate all required conditions before creating any checkpoint file."""
    base_index = json.loads(BASE_INDEX.read_text())
    weight_map = base_index["weight_map"]
    assert len(weight_map) == 114, f"expected 114 base/output tensors, found {len(weight_map)}"
    assert not any("lora_" in name for name in weight_map), "LoRA name present in output key set"

    config = json.loads(ADAPTER_CONFIG.read_text())
    rank = config["r"]
    alpha = config["lora_alpha"]
    assert rank == 16, f"unexpected LoRA rank: {rank}"
    assert config.get("fan_in_fan_out") is False, "only nn.Linear LoRA layout is supported"
    scale = float(alpha) / float(rank)

    pairs: dict[str, dict[str, str]] = {}
    with safe_open(ADAPTER_FILE, framework="pt", device="cpu") as adapter:
        adapter_keys = list(adapter.keys())
        for adapter_name in adapter_keys:
            base_name, factor = base_name_from_adapter(adapter_name)
            factors = pairs.setdefault(base_name, {})
            assert factor not in factors, f"duplicate {factor} factor for {base_name}"
            factors[factor] = adapter_name

        assert len(adapter_keys) == 64, f"expected 64 adapter tensors, found {len(adapter_keys)}"
        assert len(pairs) == 32, f"expected exactly 32 adapter pairs, found {len(pairs)}"
        for base_name, factors in pairs.items():
            assert set(factors) == {"A", "B"}, f"incomplete adapter pair for {base_name}"
            assert base_name in weight_map, f"adapter has no matching base tensor: {base_name}"
            a_slice = adapter.get_slice(factors["A"])
            b_slice = adapter.get_slice(factors["B"])
            assert a_slice.get_shape() == [16, 2048], (
                f"bad A shape for {base_name}: {a_slice.get_shape()}"
            )
            assert b_slice.get_shape() == [2048, 16], (
                f"bad B shape for {base_name}: {b_slice.get_shape()}"
            )
            assert a_slice.get_dtype() == "F32" and b_slice.get_dtype() == "F32", (
                f"adapter pair is not float32: {base_name}"
            )

    metadata: dict[str, tuple[list[int], torch.dtype, int]] = {}
    with ExitStack() as stack:
        readers = {
            shard: stack.enter_context(safe_open(BASE_DIR / shard, framework="pt", device="cpu"))
            for shard in set(weight_map.values())
        }
        for name, shard in weight_map.items():
            tensor_slice = readers[shard].get_slice(name)
            shape = tensor_slice.get_shape()
            dtype_name = tensor_slice.get_dtype()
            assert dtype_name == "F32", f"base tensor is not float32: {name} ({dtype_name})"
            dtype = torch.float32
            metadata[name] = (shape, dtype, tensor_nbytes(shape, dtype))

    q_shape = metadata["model.layers.0.self_attn.q_proj.weight"][0]
    assert q_shape == [2048, 2048], f"unexpected q_proj shape: {q_shape}"
    for name in pairs:
        assert metadata[name][0] == [2048, 2048], f"bad adapted base shape: {name}"

    # These are the four required checks, all completed before the first save_file call.
    assert len(pairs) == 32
    assert not any("lora_" in name for name in weight_map)
    assert q_shape == [2048, 2048]
    assert len(weight_map) == 114
    return weight_map, pairs, scale, metadata


def make_shard_plan(
    names: list[str], metadata: dict[str, tuple[list[int], torch.dtype, int]]
) -> list[list[str]]:
    shards: list[list[str]] = []
    current: list[str] = []
    current_bytes = 0

    def flush() -> None:
        nonlocal current, current_bytes
        if current:
            shards.append(current)
            current = []
            current_bytes = 0

    for name in names:
        size = metadata[name][2]
        if name in FORCE_SINGLETON or size > MAX_SHARD_BYTES:
            flush()
            shards.append([name])
            continue
        if current and current_bytes + size > MAX_SHARD_BYTES:
            flush()
        current.append(name)
        current_bytes += size
    flush()

    for shard in shards:
        shard_bytes = sum(metadata[name][2] for name in shard)
        if shard_bytes > MAX_SHARD_BYTES:
            assert len(shard) == 1, f"multi-tensor shard exceeds limit: {shard_bytes}"
        if any(name in FORCE_SINGLETON for name in shard):
            assert len(shard) == 1, f"required singleton tensor was packed with another: {shard}"
    return shards


def main() -> None:
    weight_map, pairs, scale, metadata = preflight()
    names = sorted(weight_map)
    shard_plan = make_shard_plan(names, metadata)
    shard_count = len(shard_plan)
    output_weight_map: dict[str, str] = {}

    existing = list(OUT_DIR.glob("model-*-of-*.safetensors"))
    assert not existing and not (OUT_DIR / INDEX_NAME).exists(), (
        "checkpoint output already exists; remove it before rerunning: "
        + ", ".join(str(path) for path in existing + [OUT_DIR / INDEX_NAME] if path.exists())
    )

    with ExitStack() as stack:
        base_readers = {
            shard: stack.enter_context(safe_open(BASE_DIR / shard, framework="pt", device="cpu"))
            for shard in set(weight_map.values())
        }
        adapter = stack.enter_context(safe_open(ADAPTER_FILE, framework="pt", device="cpu"))

        merged_count = 0
        for shard_number, shard_names in enumerate(shard_plan, start=1):
            tensors: dict[str, torch.Tensor] = {}
            for name in shard_names:
                base = base_readers[weight_map[name]].get_tensor(name)
                if name in pairs:
                    a = adapter.get_tensor(pairs[name]["A"])
                    b = adapter.get_tensor(pairs[name]["B"])
                    assert base.dtype == a.dtype == b.dtype == torch.float32
                    merged = base + scale * (b @ a)
                    assert merged.shape == base.shape and merged.dtype == torch.float32
                    tensors[name] = merged.contiguous()
                    merged_count += 1
                else:
                    tensors[name] = base

            filename = f"model-{shard_number:05d}-of-{shard_count:05d}.safetensors"
            save_file(tensors, OUT_DIR / filename, metadata={"format": "pt"})
            output_weight_map.update({name: filename for name in shard_names})
            print(f"wrote {filename}: {len(shard_names)} tensors")

    assert merged_count == 32, f"merged {merged_count} pairs instead of 32"
    assert len(output_weight_map) == 114
    assert set(output_weight_map) == set(weight_map)
    assert not any("lora_" in name for name in output_weight_map)

    total_size = sum(item[2] for item in metadata.values())
    output_index = {
        "metadata": {"total_size": total_size},
        "weight_map": output_weight_map,
    }
    (OUT_DIR / INDEX_NAME).write_text(json.dumps(output_index, indent=2, sort_keys=True) + "\n")

    # Validate the files and index after export as a second line of defense.
    seen: set[str] = set()
    for shard_names in shard_plan:
        filename = output_weight_map[shard_names[0]]
        with safe_open(OUT_DIR / filename, framework="pt", device="cpu") as reader:
            actual = set(reader.keys())
        expected = set(shard_names)
        assert actual == expected, f"shard contents mismatch in {filename}"
        seen.update(actual)
        shard_bytes = sum(metadata[name][2] for name in shard_names)
        assert shard_bytes <= MAX_SHARD_BYTES or len(shard_names) == 1
    assert seen == set(weight_map) and len(seen) == 114
    print(f"merged 32 LoRA pairs at scale {scale:g}; exported 114 tensors in {shard_count} shards")


if __name__ == "__main__":
    main()
