#!/usr/bin/env python3
"""Prune OLMo transformer blocks and write one validated safetensors file."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file


BASE = Path("inputs/base")
OUTPUT = Path("out/T1/model.safetensors")
TEMP = OUTPUT.with_suffix(".safetensors.tmp")
LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.(.+)$")
Q_PROJ_RE = re.compile(r"^model\.layers\.(\d+)\.self_attn\.q_proj\.weight$")
DROP = {2, 6, 10, 14}
SURVIVORS = [i for i in range(16) if i not in DROP]
RENUMBER = {old: new for new, old in enumerate(SURVIVORS)}


def fail(message: str) -> None:
    raise RuntimeError(message)


def main() -> None:
    index_path = BASE / "model.safetensors.index.json"
    with index_path.open(encoding="utf-8") as stream:
        weight_map = json.load(stream)["weight_map"]

    if len(weight_map) != 114:
        fail(f"expected 114 input tensors, found {len(weight_map)}")

    output_sources: dict[str, tuple[Path, str]] = {}
    source_layer_counts = {i: 0 for i in range(16)}

    for old_key, shard_name in weight_map.items():
        match = LAYER_RE.fullmatch(old_key)
        if match:
            old_layer = int(match.group(1))
            if old_layer not in source_layer_counts:
                fail(f"unexpected source layer index in {old_key}")
            source_layer_counts[old_layer] += 1
            if old_layer in DROP:
                continue
            new_key = f"model.layers.{RENUMBER[old_layer]}.{match.group(2)}"
        else:
            new_key = old_key

        if new_key in output_sources:
            fail(f"destination collision at {new_key}")
        output_sources[new_key] = (BASE / shard_name, old_key)

    if any(count != 7 for count in source_layer_counts.values()):
        fail(f"expected 7 tensors in every input block, found {source_layer_counts}")

    keys = set(output_sources)
    output_layers = {
        int(match.group(1))
        for key in keys
        if (match := LAYER_RE.fullmatch(key))
    }
    forbidden = {
        key
        for key in keys
        if (match := LAYER_RE.fullmatch(key)) and int(match.group(1)) in {12, 13, 14, 15}
    }
    q_proj_layers = {
        int(match.group(1))
        for key in keys
        if (match := Q_PROJ_RE.fullmatch(key))
    }

    if forbidden:
        fail(f"forbidden output block tensors remain: {sorted(forbidden)}")
    if output_layers != set(range(12)):
        fail(f"expected output block indices 0..11, found {sorted(output_layers)}")
    if q_proj_layers != set(range(12)) or len(q_proj_layers) != 12:
        fail(f"expected exactly one q_proj tensor for each block 0..11, found {sorted(q_proj_layers)}")
    if len(keys) != 86:
        fail(f"expected 86 output tensors, found {len(keys)}")

    tensors = {}
    by_shard: dict[Path, list[tuple[str, str]]] = {}
    for new_key, (shard_path, old_key) in output_sources.items():
        by_shard.setdefault(shard_path, []).append((new_key, old_key))
    for shard_path, entries in by_shard.items():
        with safe_open(shard_path, framework="pt", device="cpu") as shard:
            for new_key, old_key in entries:
                tensors[new_key] = shard.get_tensor(old_key)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    TEMP.unlink(missing_ok=True)
    try:
        save_file(tensors, TEMP)
        with safe_open(TEMP, framework="pt", device="cpu") as saved:
            saved_keys = set(saved.keys())
        if saved_keys != keys or len(saved_keys) != 86:
            fail("saved-file key validation failed")
        saved_q = {int(m.group(1)) for key in saved_keys if (m := Q_PROJ_RE.fullmatch(key))}
        if saved_q != set(range(12)):
            fail(f"saved file does not contain exactly blocks 0..11: {sorted(saved_q)}")
        if any(
            (m := LAYER_RE.fullmatch(key)) and int(m.group(1)) in {12, 13, 14, 15}
            for key in saved_keys
        ):
            fail("saved file contains a forbidden block index")
        os.replace(TEMP, OUTPUT)
    except BaseException:
        TEMP.unlink(missing_ok=True)
        raise

    print(f"wrote {OUTPUT} with {len(keys)} tensors and layers 0..11")


if __name__ == "__main__":
    main()
