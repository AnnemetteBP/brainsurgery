#!/usr/bin/env python3
"""Drop Pythia blocks 2, 6, 10, and 14 and compact layer indices."""

from __future__ import annotations

import gc
import os
import re
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file


INPUT = Path("inputs/base/model.safetensors")
OUTPUT = Path("out/T1/model.safetensors")
TEMP_OUTPUT = OUTPUT.with_suffix(".safetensors.tmp")
LAYER_KEY = re.compile(r"^gpt_neox\.layers\.(\d+)\.(.+)$")
REMOVED = {2, 6, 10, 14}
SURVIVORS = [index for index in range(16) if index not in REMOVED]
OLD_TO_NEW = {old: new for new, old in enumerate(SURVIVORS)}
QKV_SUFFIX = "attention.query_key_value.weight"


def split_layer_key(key: str) -> tuple[int, str] | None:
    match = LAYER_KEY.fullmatch(key)
    if match is None:
        return None
    return int(match.group(1)), match.group(2)


def validate_source(keys: list[str]) -> None:
    assert len(keys) == 244, f"expected 244 source tensors, found {len(keys)}"

    layer_counts = {index: 0 for index in range(16)}
    non_block_count = 0
    qkv_layers: set[int] = set()
    for key in keys:
        parsed = split_layer_key(key)
        if parsed is None:
            non_block_count += 1
            continue
        index, suffix = parsed
        assert index in layer_counts, f"unexpected source layer index {index}: {key}"
        layer_counts[index] += 1
        if suffix == QKV_SUFFIX:
            qkv_layers.add(index)

    assert non_block_count == 4, f"expected 4 non-block tensors, found {non_block_count}"
    assert all(count == 15 for count in layer_counts.values()), (
        f"expected 15 tensors per source block, found {layer_counts}"
    )
    assert qkv_layers == set(range(16)), (
        f"source QKV layer set is incomplete or unexpected: {sorted(qkv_layers)}"
    )


def destination_key(source_key: str) -> str | None:
    parsed = split_layer_key(source_key)
    if parsed is None:
        return source_key
    old_index, suffix = parsed
    if old_index in REMOVED:
        return None
    assert old_index in OLD_TO_NEW, f"no layer mapping for source key: {source_key}"
    return f"gpt_neox.layers.{OLD_TO_NEW[old_index]}.{suffix}"


def validate_destination_keys(keys: list[str]) -> None:
    assert len(keys) == len(set(keys)), "destination key collision detected"
    assert len(keys) == 184, f"expected 184 output tensors, found {len(keys)}"

    layer_indices: set[int] = set()
    qkv_layers: set[int] = set()
    for key in keys:
        parsed = split_layer_key(key)
        if parsed is None:
            continue
        index, suffix = parsed
        layer_indices.add(index)
        if suffix == QKV_SUFFIX:
            qkv_layers.add(index)

    forbidden = layer_indices & {12, 13, 14, 15}
    assert not forbidden, f"forbidden output layer indices remain: {sorted(forbidden)}"
    assert layer_indices == set(range(12)), (
        f"output layer indices are not contiguous 0..11: {sorted(layer_indices)}"
    )
    assert qkv_layers == set(range(12)), (
        "expected exactly one QKV weight for each of 12 layers; "
        f"found layers {sorted(qkv_layers)}"
    )


def main() -> None:
    assert INPUT.is_file(), f"input checkpoint not found: {INPUT}"
    assert not OUTPUT.exists(), f"refusing to overwrite existing output: {OUTPUT}"
    assert not TEMP_OUTPUT.exists(), f"stale temporary output exists: {TEMP_OUTPUT}"

    with safe_open(INPUT, framework="pt", device="cpu") as source:
        source_keys = list(source.keys())
        source_metadata = source.metadata()
        validate_source(source_keys)

        mapped_pairs = [
            (key, mapped)
            for key in source_keys
            if (mapped := destination_key(key)) is not None
        ]
        destination_keys = [mapped for _, mapped in mapped_pairs]
        validate_destination_keys(destination_keys)

        # The destination dictionary is constructed from immutable source names,
        # so no in-place rename can overwrite another surviving layer.
        tensors = {mapped: source.get_tensor(key) for key, mapped in mapped_pairs}

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    try:
        save_file(tensors, TEMP_OUTPUT, metadata=source_metadata)
        del tensors
        gc.collect()

        with safe_open(TEMP_OUTPUT, framework="pt", device="cpu") as written:
            written_keys = list(written.keys())
        validate_destination_keys(written_keys)
        os.replace(TEMP_OUTPUT, OUTPUT)
    except BaseException:
        TEMP_OUTPUT.unlink(missing_ok=True)
        raise

    print(f"wrote {OUTPUT} with {len(destination_keys)} tensors")


if __name__ == "__main__":
    main()
