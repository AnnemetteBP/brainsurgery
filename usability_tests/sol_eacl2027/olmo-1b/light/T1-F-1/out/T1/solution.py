#!/usr/bin/env python3
"""Prune and contiguously renumber OLMo transformer blocks."""

import json
import re
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file


BASE = Path("inputs/base")
OUTPUT = Path("out/T1/model.safetensors")
INDEX = BASE / "model.safetensors.index.json"
LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.(.+)$")
DROPPED = {2, 6, 10, 14}
SURVIVORS = [i for i in range(16) if i not in DROPPED]
RENUMBER = {old: new for new, old in enumerate(SURVIVORS)}


def main() -> None:
    with INDEX.open(encoding="utf-8") as handle:
        weight_map = json.load(handle)["weight_map"]

    if len(weight_map) != 114:
        raise RuntimeError(f"expected 114 source tensors, found {len(weight_map)}")

    output = {}
    shard_names = sorted(set(weight_map.values()))
    for shard_name in shard_names:
        shard_path = BASE / shard_name
        with safe_open(shard_path, framework="pt", device="cpu") as shard:
            shard_keys = set(shard.keys())
            expected_keys = {k for k, v in weight_map.items() if v == shard_name}
            if shard_keys != expected_keys:
                raise RuntimeError(f"index/key mismatch in {shard_name}")
            for key in shard.keys():
                match = LAYER_RE.match(key)
                if match:
                    old_index = int(match.group(1))
                    if old_index in DROPPED:
                        continue
                    if old_index not in RENUMBER:
                        raise RuntimeError(f"unexpected source layer index {old_index}")
                    new_key = f"model.layers.{RENUMBER[old_index]}.{match.group(2)}"
                else:
                    new_key = key
                if new_key in output:
                    raise RuntimeError(f"rename collision at {new_key}")
                output[new_key] = shard.get_tensor(key)

    layer_indices = {
        int(match.group(1))
        for key in output
        if (match := LAYER_RE.match(key))
    }
    forbidden = [key for key in output if (m := LAYER_RE.match(key)) and int(m.group(1)) >= 12]
    q_proj = re.compile(r"^model\.layers\.(\d+)\.self_attn\.q_proj\.weight$")
    q_proj_count = sum(q_proj.match(key) is not None for key in output)

    if forbidden:
        raise RuntimeError(f"forbidden layer indices remain: {forbidden[:3]}")
    if layer_indices != set(range(12)):
        raise RuntimeError(f"expected layers 0..11, found {sorted(layer_indices)}")
    if q_proj_count != 12:
        raise RuntimeError(f"expected 12 q_proj tensors, found {q_proj_count}")
    if len(output) != 86:
        raise RuntimeError(f"expected 86 output tensors, found {len(output)}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix(".safetensors.tmp")
    save_file(output, temporary)
    temporary.replace(OUTPUT)


if __name__ == "__main__":
    main()
