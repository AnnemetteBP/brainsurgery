#!/usr/bin/env python3
"""Export the OLMo checkpoint with projection-only BF16 and <=256 MiB shards."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


INPUT_DIR = Path("inputs/base")
OUTPUT_DIR = Path("out/T3")
INDEX_NAME = "model.safetensors.index.json"
MAX_SHARD_BYTES = 256 * 1024 * 1024
EXPECTED_TENSORS = 114
EXPECTED_BF16 = 112

PROJECTION_RE = re.compile(
    r"^model\.layers\.(?:[0-9]|1[0-5])\."
    r"(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|"
    r"mlp\.(?:gate_proj|up_proj|down_proj))\.weight$"
)

SAFETENSORS_TO_TORCH_DTYPE = {
    "F32": torch.float32,
    "BF16": torch.bfloat16,
}


def slice_dtype(view: object) -> torch.dtype:
    """Convert safetensors' metadata dtype label to a torch dtype."""
    label = view.get_dtype()
    assert label in SAFETENSORS_TO_TORCH_DTYPE, f"unsupported safetensors dtype: {label}"
    return SAFETENSORS_TO_TORCH_DTYPE[label]


def tensor_nbytes(shape: tuple[int, ...], dtype: torch.dtype) -> int:
    return torch.tensor([], dtype=dtype).element_size() * int(torch.Size(shape).numel())


def build_manifest() -> tuple[dict[str, str], dict[str, tuple[tuple[int, ...], torch.dtype]], dict[str, torch.dtype]]:
    """Read only tensor metadata and determine every output dtype."""
    index = json.loads((INPUT_DIR / INDEX_NAME).read_text())
    source_map: dict[str, str] = index["weight_map"]
    specs: dict[str, tuple[tuple[int, ...], torch.dtype]] = {}

    by_shard: dict[str, list[str]] = defaultdict(list)
    for name, shard in source_map.items():
        by_shard[shard].append(name)

    for shard, expected_names in by_shard.items():
        with safe_open(INPUT_DIR / shard, framework="pt", device="cpu") as handle:
            actual_names = set(handle.keys())
            assert actual_names == set(expected_names), (
                f"source index mismatch for {shard}: "
                f"indexed={len(expected_names)}, stored={len(actual_names)}"
            )
            for name in expected_names:
                view = handle.get_slice(name)
                specs[name] = (tuple(view.get_shape()), slice_dtype(view))

    # Confirm that the exact intended set exists: 16 layers * 7 projections.
    projection_names = {name for name in specs if PROJECTION_RE.fullmatch(name)}
    expected_projection_names = {
        f"model.layers.{layer}.{block}.{projection}.weight"
        for layer in range(16)
        for block, projections in (
            ("self_attn", ("q_proj", "k_proj", "v_proj", "o_proj")),
            ("mlp", ("gate_proj", "up_proj", "down_proj")),
        )
        for projection in projections
    }
    assert projection_names == expected_projection_names, (
        "projection key mismatch: "
        f"missing={sorted(expected_projection_names - projection_names)}, "
        f"unexpected={sorted(projection_names - expected_projection_names)}"
    )
    assert all(dtype == torch.float32 for _, dtype in specs.values()), (
        "expected every source tensor to be float32"
    )

    output_dtypes = {
        name: torch.bfloat16 if name in projection_names else torch.float32
        for name in specs
    }

    # Mandatory pre-write checks. No output checkpoint files exist at this point.
    assert sum(dtype == torch.bfloat16 for dtype in output_dtypes.values()) == EXPECTED_BF16
    assert output_dtypes["model.layers.0.self_attn.q_proj.weight"] == torch.bfloat16
    assert output_dtypes["model.embed_tokens.weight"] == torch.float32
    assert len(output_dtypes) == EXPECTED_TENSORS
    return source_map, specs, output_dtypes


def plan_shards(
    specs: dict[str, tuple[tuple[int, ...], torch.dtype]],
    output_dtypes: dict[str, torch.dtype],
) -> list[list[str]]:
    """Pack in index order; an oversized tensor is always a singleton shard."""
    shards: list[list[str]] = []
    current: list[str] = []
    current_bytes = 0

    for name, (shape, _) in specs.items():
        size = tensor_nbytes(shape, output_dtypes[name])
        if size > MAX_SHARD_BYTES:
            if current:
                shards.append(current)
                current, current_bytes = [], 0
            shards.append([name])
        elif current and current_bytes + size > MAX_SHARD_BYTES:
            shards.append(current)
            current, current_bytes = [name], size
        else:
            current.append(name)
            current_bytes += size
    if current:
        shards.append(current)
    return shards


def write_checkpoint(
    source_map: dict[str, str],
    specs: dict[str, tuple[tuple[int, ...], torch.dtype]],
    output_dtypes: dict[str, torch.dtype],
    shards: list[list[str]],
) -> None:
    output_weight_map: dict[str, str] = {}
    total_shards = len(shards)

    for shard_number, names in enumerate(shards, start=1):
        filename = f"model-{shard_number:05d}-of-{total_shards:05d}.safetensors"
        tensors: dict[str, torch.Tensor] = {}
        handles: dict[str, object] = {}
        try:
            for name in names:
                source_shard = source_map[name]
                if source_shard not in handles:
                    handles[source_shard] = safe_open(
                        INPUT_DIR / source_shard, framework="pt", device="cpu"
                    )
                tensor = handles[source_shard].get_tensor(name)
                target_dtype = output_dtypes[name]
                if tensor.dtype != target_dtype:
                    tensor = tensor.to(target_dtype)
                tensors[name] = tensor.contiguous()
                output_weight_map[name] = filename

            data_bytes = sum(t.numel() * t.element_size() for t in tensors.values())
            assert data_bytes <= MAX_SHARD_BYTES or len(tensors) == 1, (
                f"invalid shard plan for {filename}: {data_bytes} bytes, "
                f"{len(tensors)} tensors"
            )
            save_file(tensors, OUTPUT_DIR / filename)
        finally:
            # safe_open objects close when released; explicit __exit__ is not public.
            handles.clear()
            tensors.clear()

    total_size = sum(
        tensor_nbytes(shape, output_dtypes[name])
        for name, (shape, _) in specs.items()
    )
    output_index = {
        "metadata": {"total_size": total_size},
        "weight_map": output_weight_map,
    }
    (OUTPUT_DIR / INDEX_NAME).write_text(json.dumps(output_index, indent=2) + "\n")


def validate_output(
    specs: dict[str, tuple[tuple[int, ...], torch.dtype]],
    output_dtypes: dict[str, torch.dtype],
) -> None:
    index = json.loads((OUTPUT_DIR / INDEX_NAME).read_text())
    weight_map = index["weight_map"]
    assert set(weight_map) == set(specs)
    assert len(weight_map) == EXPECTED_TENSORS

    by_shard: dict[str, list[str]] = defaultdict(list)
    for name, shard in weight_map.items():
        by_shard[shard].append(name)

    observed: set[str] = set()
    for shard, mapped_names in by_shard.items():
        shard_path = OUTPUT_DIR / shard
        assert shard_path.is_file(), f"missing output shard: {shard}"
        with safe_open(shard_path, framework="pt", device="cpu") as handle:
            stored_names = set(handle.keys())
            assert stored_names == set(mapped_names), f"weight_map mismatch for {shard}"
            data_bytes = 0
            for name in stored_names:
                view = handle.get_slice(name)
                shape = tuple(view.get_shape())
                dtype = slice_dtype(view)
                assert shape == specs[name][0], f"shape mismatch: {name}"
                assert dtype == output_dtypes[name], f"dtype mismatch: {name}"
                data_bytes += tensor_nbytes(shape, dtype)
            assert data_bytes <= MAX_SHARD_BYTES or len(stored_names) == 1, (
                f"oversized non-singleton shard: {shard}"
            )
            observed.update(stored_names)

    assert observed == set(specs)
    assert sum(dtype == torch.bfloat16 for dtype in output_dtypes.values()) == EXPECTED_BF16
    print(
        f"Wrote and validated {len(observed)} tensors in {len(by_shard)} shards "
        f"({EXPECTED_BF16} bfloat16, {EXPECTED_TENSORS - EXPECTED_BF16} float32)."
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    existing = list(OUTPUT_DIR.glob("*.safetensors")) + list(OUTPUT_DIR.glob("*.index.json"))
    assert not existing, f"refusing to overwrite existing checkpoint files: {existing}"
    source_map, specs, output_dtypes = build_manifest()
    shards = plan_shards(specs, output_dtypes)
    write_checkpoint(source_map, specs, output_dtypes, shards)
    validate_output(specs, output_dtypes)


if __name__ == "__main__":
    main()
