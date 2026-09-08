#!/usr/bin/env python3
import json
import os
import re
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


BASE = Path("inputs/base")
OUTPUT = Path("out/T1/model.safetensors")
TEMP_OUTPUT = OUTPUT.with_suffix(".safetensors.tmp")
INDEX = BASE / "model.safetensors.index.json"
LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.(.+)$")
REMOVED = {2, 6, 10, 14}
SURVIVORS = [i for i in range(16) if i not in REMOVED]
RENUMBER = {old: new for new, old in enumerate(SURVIVORS)}


def fail(message: str) -> None:
    raise RuntimeError(message)


def main() -> None:
    # torch is intentionally part of this standalone PyTorch/safetensors script.
    if not torch.__version__:
        fail("PyTorch is unavailable")

    with INDEX.open("r", encoding="utf-8") as handle:
        weight_map = json.load(handle)["weight_map"]

    if len(weight_map) != 114:
        fail(f"expected 114 input tensors, found {len(weight_map)}")

    output_tensors = {}
    open_shards = {}
    try:
        for old_name, shard_name in weight_map.items():
            match = LAYER_RE.match(old_name)
            if match:
                old_layer = int(match.group(1))
                if old_layer in REMOVED:
                    continue
                if old_layer not in RENUMBER:
                    fail(f"unexpected input layer {old_layer}: {old_name}")
                new_name = f"model.layers.{RENUMBER[old_layer]}.{match.group(2)}"
            else:
                new_name = old_name

            if new_name in output_tensors:
                fail(f"rename collision at {new_name}")

            if shard_name not in open_shards:
                open_shards[shard_name] = safe_open(
                    BASE / shard_name, framework="pt", device="cpu"
                )
            output_tensors[new_name] = open_shards[shard_name].get_tensor(old_name)

        output_layers = {
            int(match.group(1))
            for name in output_tensors
            if (match := LAYER_RE.match(name))
        }
        forbidden = output_layers & {12, 13, 14, 15}
        if forbidden:
            fail(f"forbidden output layer indices remain: {sorted(forbidden)}")

        q_proj_layers = {
            int(match.group(1))
            for name in output_tensors
            if (
                match := re.fullmatch(
                    r"model\.layers\.(\d+)\.self_attn\.q_proj\.weight", name
                )
            )
        }
        if q_proj_layers != set(range(12)):
            fail(f"expected q_proj tensors for layers 0..11, found {sorted(q_proj_layers)}")
        if output_layers != set(range(12)):
            fail(f"expected exactly layers 0..11, found {sorted(output_layers)}")
        if len(output_tensors) != 86:
            fail(f"expected 86 output tensors, found {len(output_tensors)}")

        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        if TEMP_OUTPUT.exists():
            TEMP_OUTPUT.unlink()
        save_file(output_tensors, TEMP_OUTPUT)
        os.replace(TEMP_OUTPUT, OUTPUT)
        print(f"Wrote {OUTPUT} with {len(output_tensors)} tensors")
    finally:
        open_shards.clear()
        if TEMP_OUTPUT.exists():
            TEMP_OUTPUT.unlink()


if __name__ == "__main__":
    main()
