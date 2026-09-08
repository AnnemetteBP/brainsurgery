#!/usr/bin/env python3
"""Drop OLMo blocks 2/6/10/14 and compact the remaining layer names."""

import json
import os
import re
from collections import defaultdict
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file


BASE = Path("inputs/base")
OUTPUT = Path("out/T1/model.safetensors")
TEMP = OUTPUT.with_suffix(".safetensors.tmp")
LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.(.+)$")
DROPPED = {2, 6, 10, 14}
SURVIVORS = [i for i in range(16) if i not in DROPPED]
RENUMBER = {old: new for new, old in enumerate(SURVIVORS)}


def validate_keys(keys: set[str]) -> None:
    layer_indices = {
        int(match.group(1))
        for key in keys
        if (match := LAYER_RE.match(key))
    }
    assert layer_indices == set(range(12)), f"wrong layer indices: {sorted(layer_indices)}"
    assert not any(i in layer_indices for i in (12, 13, 14, 15)), "unrenumbered high layer remains"
    q_proj = re.compile(r"^model\.layers\.(\d+)\.self_attn\.q_proj\.weight$")
    assert sum(q_proj.match(key) is not None for key in keys) == 12, "expected 12 q_proj tensors"
    assert len(keys) == 86, f"expected 86 tensors, got {len(keys)}"


def main() -> None:
    index = json.loads((BASE / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    assert len(weight_map) == 114, f"expected 114 input tensors, got {len(weight_map)}"

    by_shard: dict[str, list[str]] = defaultdict(list)
    for key, shard in weight_map.items():
        by_shard[shard].append(key)

    output = {}
    for shard, keys in by_shard.items():
        with safe_open(BASE / shard, framework="pt", device="cpu") as source:
            for old_key in keys:
                match = LAYER_RE.match(old_key)
                if match:
                    old_layer = int(match.group(1))
                    assert old_layer in range(16), f"unexpected input layer {old_layer}"
                    if old_layer in DROPPED:
                        continue
                    new_key = f"model.layers.{RENUMBER[old_layer]}.{match.group(2)}"
                else:
                    new_key = old_key
                assert new_key not in output, f"key collision: {new_key}"
                output[new_key] = source.get_tensor(old_key)

    validate_keys(set(output))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    TEMP.unlink(missing_ok=True)
    try:
        save_file(output, TEMP)
        with safe_open(TEMP, framework="pt", device="cpu") as saved:
            validate_keys(set(saved.keys()))
        os.replace(TEMP, OUTPUT)
    except BaseException:
        TEMP.unlink(missing_ok=True)
        OUTPUT.unlink(missing_ok=True)
        raise


if __name__ == "__main__":
    main()
