#!/usr/bin/env python3
"""Merge the supplied PEFT LoRA into the base checkpoint and shard the result."""

from __future__ import annotations

import json
from contextlib import ExitStack
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file


MAX_SHARD_BYTES = 512 * 1024 * 1024
EXPECTED_TENSORS = 114
EXPECTED_PAIRS = 32
LARGE_TENSORS = {"model.embed_tokens.weight", "lm_head.weight"}
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
    "F64": 8,
    "I64": 8,
    "U64": 8,
}

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE_DIR = ROOT / "inputs" / "base"
LORA_DIR = ROOT / "inputs" / "lora"


def tensor_metadata(weight_map: dict[str, str]) -> dict[str, tuple[tuple[int, ...], str, int]]:
    """Read tensor shapes/dtypes/byte sizes from safetensors headers only."""
    metadata: dict[str, tuple[tuple[int, ...], str, int]] = {}
    by_file: dict[str, list[str]] = {}
    for name, filename in weight_map.items():
        by_file.setdefault(filename, []).append(name)

    for filename, expected_names in by_file.items():
        with safe_open(BASE_DIR / filename, framework="pt", device="cpu") as handle:
            actual_names = set(handle.keys())
            assert set(expected_names) <= actual_names, f"index references missing tensors in {filename}"
            for name in expected_names:
                view = handle.get_slice(name)
                shape = tuple(view.get_shape())
                dtype = view.get_dtype()
                assert dtype in DTYPE_BYTES, f"unsupported dtype {dtype} for {name}"
                numel = 1
                for dimension in shape:
                    numel *= dimension
                metadata[name] = (shape, dtype, numel * DTYPE_BYTES[dtype])
    return metadata


def plan_shards(names: list[str], sizes: dict[str, int]) -> list[list[str]]:
    """Put designated deployment-sized matrices alone; greedily pack the rest."""
    shards = [[name] for name in sorted(LARGE_TENSORS)]
    current: list[str] = []
    current_bytes = 0
    for name in sorted(set(names) - LARGE_TENSORS):
        size = sizes[name]
        assert size <= MAX_SHARD_BYTES, f"non-isolated tensor {name} exceeds shard limit"
        if current and current_bytes + size > MAX_SHARD_BYTES:
            shards.append(current)
            current = []
            current_bytes = 0
        current.append(name)
        current_bytes += size
    if current:
        shards.append(current)
    return shards


def main() -> None:
    config = json.loads((LORA_DIR / "adapter_config.json").read_text())
    rank = config["r"]
    alpha = config["lora_alpha"]
    assert isinstance(rank, int) and rank > 0, f"invalid LoRA rank: {rank!r}"
    assert isinstance(alpha, (int, float)), f"invalid LoRA alpha: {alpha!r}"
    assert config.get("fan_in_fan_out") is False, "this exporter expects nn.Linear [out,in] layout"
    scale = alpha / rank

    base_index = json.loads((BASE_DIR / "model.safetensors.index.json").read_text())
    weight_map: dict[str, str] = base_index["weight_map"]
    output_names = set(weight_map)
    assert len(output_names) == EXPECTED_TENSORS, (
        f"output would have {len(output_names)} tensors, expected {EXPECTED_TENSORS}"
    )
    assert not any("lora_" in name for name in output_names), "LoRA tensor name leaked into output"

    metadata = tensor_metadata(weight_map)
    q0_shape = metadata["model.layers.0.self_attn.q_proj.weight"][0]
    assert q0_shape == (2048, 2048), f"unexpected q_proj shape: {q0_shape}"

    adapter = load_file(LORA_DIR / "adapter_model.safetensors", device="cpu")
    adapter_names = set(adapter)
    a_suffix = ".lora_A.weight"
    b_suffix = ".lora_B.weight"
    a_names = {name for name in adapter_names if name.endswith(a_suffix)}
    b_names = {name for name in adapter_names if name.endswith(b_suffix)}
    assert adapter_names == a_names | b_names, "adapter contains tensors other than LoRA A/B weights"

    merge_for_base: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for a_name in sorted(a_names):
        prefix = a_name[: -len(a_suffix)]
        b_name = prefix + b_suffix
        assert b_name in b_names, f"missing B factor for {a_name}"
        assert prefix.startswith("base_model.model."), f"unexpected PEFT prefix: {prefix}"
        base_name = prefix.removeprefix("base_model.model.") + ".weight"
        assert base_name in output_names, f"mapped base tensor is absent: {base_name}"
        assert base_name not in merge_for_base, f"duplicate adapter mapping: {base_name}"

        a = adapter[a_name]
        b = adapter[b_name]
        assert a.dtype == torch.float32 and b.dtype == torch.float32, f"non-float32 adapter pair: {prefix}"
        assert tuple(a.shape) == (rank, metadata[base_name][0][1]), f"bad A shape for {base_name}"
        assert tuple(b.shape) == (metadata[base_name][0][0], rank), f"bad B shape for {base_name}"
        assert metadata[base_name][1] == "F32", f"base tensor is not float32: {base_name}"
        merge_for_base[base_name] = (a, b)

    assert len(merge_for_base) == EXPECTED_PAIRS, (
        f"found {len(merge_for_base)} adapter pairs, expected {EXPECTED_PAIRS}"
    )
    assert len(b_names) == EXPECTED_PAIRS, f"found {len(b_names)} B factors, expected {EXPECTED_PAIRS}"

    # All required checks above are complete before any output checkpoint is written.
    sizes = {name: fields[2] for name, fields in metadata.items()}
    shard_plan = plan_shards(list(output_names), sizes)
    assert all(sum(sizes[name] for name in shard) <= MAX_SHARD_BYTES for shard in shard_plan)
    assert all([name] in shard_plan for name in LARGE_TENSORS), "large tensors must be isolated"

    for stale in HERE.glob("model-*-of-*.safetensors"):
        stale.unlink()
    index_path = HERE / "model.safetensors.index.json"
    if index_path.exists():
        index_path.unlink()

    output_weight_map: dict[str, str] = {}
    shard_count = len(shard_plan)
    with ExitStack() as stack:
        source_handles = {
            filename: stack.enter_context(safe_open(BASE_DIR / filename, framework="pt", device="cpu"))
            for filename in sorted(set(weight_map.values()))
        }
        for shard_number, shard_names in enumerate(shard_plan, start=1):
            filename = f"model-{shard_number:05d}-of-{shard_count:05d}.safetensors"
            tensors: dict[str, torch.Tensor] = {}
            for name in shard_names:
                base = source_handles[weight_map[name]].get_tensor(name)
                if name in merge_for_base:
                    a, b = merge_for_base[name]
                    base = base + scale * torch.matmul(b, a)
                tensors[name] = base.contiguous()
                output_weight_map[name] = filename
            save_file(tensors, HERE / filename, metadata={"format": "pt"})

    assert set(output_weight_map) == output_names
    output_index = {
        "metadata": {"total_size": sum(sizes.values())},
        "weight_map": dict(sorted(output_weight_map.items())),
    }
    index_path.write_text(json.dumps(output_index, indent=2) + "\n")

    # Validate the completed export, including actual shard membership and byte totals.
    seen: set[str] = set()
    for filename in sorted(set(output_weight_map.values())):
        expected = {name for name, mapped in output_weight_map.items() if mapped == filename}
        with safe_open(HERE / filename, framework="pt", device="cpu") as handle:
            actual = set(handle.keys())
            assert actual == expected, f"index/file mismatch for {filename}"
            shard_bytes = sum(sizes[name] for name in actual)
            assert shard_bytes <= MAX_SHARD_BYTES, f"oversized shard {filename}: {shard_bytes} bytes"
            seen.update(actual)
    assert seen == output_names and len(seen) == EXPECTED_TENSORS
    assert not any("lora_" in name for name in seen)
    print(
        f"Merged {len(merge_for_base)} LoRA pairs into {len(seen)} tensors; "
        f"wrote {shard_count} shards (scale={scale:g})."
    )


if __name__ == "__main__":
    main()
