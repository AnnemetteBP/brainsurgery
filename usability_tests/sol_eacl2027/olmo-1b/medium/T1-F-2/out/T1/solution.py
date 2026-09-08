#!/usr/bin/env python3
"""Prune four OLMo blocks and write one validated safetensors checkpoint."""

import json
import os
import re
import sys
import tempfile
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file


BASE = Path("inputs/base")
INDEX = BASE / "model.safetensors.index.json"
DEST = Path("out/T1/model.safetensors")
LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.(.+)$")
Q_PROJ_RE = re.compile(r"^model\.layers\.(\d+)\.self_attn\.q_proj\.weight$")
DROP = {2, 6, 10, 14}
SURVIVORS = [i for i in range(16) if i not in DROP]
RENUMBER = {old: new for new, old in enumerate(SURVIVORS)}
NON_BLOCK_KEYS = {"model.embed_tokens.weight", "lm_head.weight"}


def check_output_keys(keys: set[str]) -> None:
    """Enforce all task-required structural checks."""
    assert len(keys) == 86, f"expected 86 tensors, found {len(keys)}"

    layer_indices = {
        int(match.group(1))
        for key in keys
        if (match := LAYER_RE.fullmatch(key)) is not None
    }
    forbidden = layer_indices & {12, 13, 14, 15}
    assert not forbidden, f"forbidden output block indices remain: {sorted(forbidden)}"
    assert layer_indices == set(range(12)), (
        f"expected layer indices 0..11, found {sorted(layer_indices)}"
    )

    q_proj_indices = {
        int(match.group(1))
        for key in keys
        if (match := Q_PROJ_RE.fullmatch(key)) is not None
    }
    assert q_proj_indices == set(range(12)), (
        "expected exactly one q_proj marker for each of 12 blocks; "
        f"found indices {sorted(q_proj_indices)}"
    )


def main() -> None:
    # A failed run must not leave the requested artifact behind.
    DEST.unlink(missing_ok=True)

    with INDEX.open("r", encoding="utf-8") as handle:
        weight_map = json.load(handle)["weight_map"]
    assert len(weight_map) == 114, f"expected 114 input tensors, found {len(weight_map)}"

    shard_names = sorted(set(weight_map.values()))
    tensors = {}
    source_keys = set()
    source_layer_counts = {i: 0 for i in range(16)}

    for shard_name in shard_names:
        with safe_open(BASE / shard_name, framework="pt", device="cpu") as shard:
            for old_key in shard.keys():
                assert old_key in weight_map, f"unindexed input tensor: {old_key}"
                assert weight_map[old_key] == shard_name, (
                    f"index assigns {old_key} to a different shard"
                )
                assert old_key not in source_keys, f"duplicate input tensor: {old_key}"
                source_keys.add(old_key)

                match = LAYER_RE.fullmatch(old_key)
                if match is None:
                    assert old_key in NON_BLOCK_KEYS, f"unexpected non-block tensor: {old_key}"
                    new_key = old_key
                else:
                    old_layer = int(match.group(1))
                    assert old_layer in source_layer_counts, f"unexpected input block: {old_layer}"
                    source_layer_counts[old_layer] += 1
                    if old_layer in DROP:
                        continue
                    new_key = f"model.layers.{RENUMBER[old_layer]}.{match.group(2)}"

                assert new_key not in tensors, f"rename collision at {new_key}"
                tensors[new_key] = shard.get_tensor(old_key)

    assert source_keys == set(weight_map), "index and shard tensor keys differ"
    assert source_layer_counts == {i: 7 for i in range(16)}, (
        f"expected seven tensors per source block, found {source_layer_counts}"
    )
    assert NON_BLOCK_KEYS <= source_keys, "one or more required non-block tensors are missing"
    check_output_keys(set(tensors))

    DEST.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=DEST.parent, prefix=".model.", suffix=".safetensors", delete=False
        ) as handle:
            temp_path = Path(handle.name)
        save_file(tensors, temp_path)

        # Validate the serialized artifact, not merely the in-memory mapping.
        with safe_open(temp_path, framework="pt", device="cpu") as output:
            persisted_keys = set(output.keys())
        check_output_keys(persisted_keys)
        assert persisted_keys == set(tensors), "serialized key set differs from planned output"

        os.replace(temp_path, DEST)
        temp_path = None
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)

    print(f"wrote {DEST} with {len(tensors)} tensors and 12 contiguous blocks")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        DEST.unlink(missing_ok=True)
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
