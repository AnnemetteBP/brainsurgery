#!/usr/bin/env python3
"""Prune four OLMo transformer blocks and densely renumber survivors."""

import json
import os
import re
from contextlib import ExitStack
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


INPUT_DIR = Path("inputs/base")
OUTPUT_PATH = Path("out/T1/model.safetensors")
TEMP_PATH = Path("out/T1/model.safetensors.tmp")
INDEX_PATH = INPUT_DIR / "model.safetensors.index.json"
REMOVED_LAYERS = {2, 6, 10, 14}
SURVIVING_LAYERS = [i for i in range(16) if i not in REMOVED_LAYERS]
LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.(.+)$")
Q_PROJ_RE = re.compile(
    r"^model\.layers\.(\d+)\.self_attn\.q_proj\.weight$"
)


def fail(message: str) -> None:
    raise RuntimeError(message)


def main() -> None:
    # PyTorch is deliberately part of this standalone state-dict workflow.
    if not hasattr(torch, "Tensor"):
        fail("PyTorch is unavailable")

    with INDEX_PATH.open("r", encoding="utf-8") as handle:
        weight_map = json.load(handle)["weight_map"]

    if len(weight_map) != 114:
        fail(f"expected 114 input tensors, found {len(weight_map)}")

    old_to_new = {old: new for new, old in enumerate(SURVIVING_LAYERS)}
    rename_map: dict[str, str] = {}
    output_names: set[str] = set()

    for old_name in weight_map:
        match = LAYER_RE.fullmatch(old_name)
        if match:
            old_layer = int(match.group(1))
            if old_layer in REMOVED_LAYERS:
                continue
            if old_layer not in old_to_new:
                fail(f"unexpected input layer in {old_name}")
            new_name = f"model.layers.{old_to_new[old_layer]}.{match.group(2)}"
        else:
            new_name = old_name

        if new_name in output_names:
            fail(f"rename collision at {new_name}")
        output_names.add(new_name)
        rename_map[old_name] = new_name

    # Required checks are performed before any output file is written.
    remaining_layer_indices = {
        int(match.group(1))
        for name in output_names
        if (match := LAYER_RE.fullmatch(name))
    }
    forbidden = remaining_layer_indices & {12, 13, 14, 15}
    if forbidden:
        fail(f"forbidden output layer indices remain: {sorted(forbidden)}")

    q_proj_indices = {
        int(match.group(1))
        for name in output_names
        if (match := Q_PROJ_RE.fullmatch(name))
    }
    if q_proj_indices != set(range(12)):
        fail(
            "expected one q_proj tensor for each layer 0..11; "
            f"found indices {sorted(q_proj_indices)}"
        )
    if len(output_names) != 86:
        fail(f"expected 86 output tensors, found {len(output_names)}")

    tensors: dict[str, torch.Tensor] = {}
    readers: dict[str, object] = {}
    try:
        with ExitStack() as stack:
            for old_name, new_name in rename_map.items():
                shard_name = weight_map[old_name]
                if shard_name not in readers:
                    readers[shard_name] = stack.enter_context(
                        safe_open(
                            INPUT_DIR / shard_name, framework="pt", device="cpu"
                        )
                    )
                tensors[new_name] = readers[shard_name].get_tensor(old_name)

            if set(tensors) != output_names:
                fail("loaded tensor keys differ from the validated rename map")

            OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
            TEMP_PATH.unlink(missing_ok=True)
            save_file(tensors, TEMP_PATH)

            with safe_open(TEMP_PATH, framework="pt", device="cpu") as saved:
                saved_names = set(saved.keys())
            if saved_names != output_names or len(saved_names) != 86:
                fail("serialized checkpoint failed key-set validation")

            os.replace(TEMP_PATH, OUTPUT_PATH)
    finally:
        TEMP_PATH.unlink(missing_ok=True)

    print(f"Wrote {OUTPUT_PATH} with {len(tensors)} tensors")


if __name__ == "__main__":
    main()
