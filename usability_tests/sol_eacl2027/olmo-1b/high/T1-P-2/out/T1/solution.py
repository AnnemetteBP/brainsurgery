#!/usr/bin/env python3
"""Prune four OLMo transformer blocks and renumber the survivors."""

import json
import re
from pathlib import Path

from safetensors.torch import load_file, save_file


INPUT_DIR = Path("inputs/base")
INDEX_PATH = INPUT_DIR / "model.safetensors.index.json"
OUTPUT_PATH = Path("out/T1/model.safetensors")

LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.(.+)$")
Q_PROJ_RE = re.compile(
    r"^model\.layers\.(\d+)\.self_attn\.q_proj\.weight$"
)
REMOVED_LAYERS = {2, 6, 10, 14}
SURVIVING_LAYERS = [i for i in range(16) if i not in REMOVED_LAYERS]
LAYER_MAP = {old: new for new, old in enumerate(SURVIVING_LAYERS)}


def main() -> None:
    with INDEX_PATH.open("r", encoding="utf-8") as handle:
        index = json.load(handle)

    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError("Safetensors index has no valid weight_map")
    if len(weight_map) != 114:
        raise ValueError(f"Expected 114 input tensors, found {len(weight_map)}")

    output = {}
    loaded_input_keys = set()
    shard_names = sorted(set(weight_map.values()))

    for shard_name in shard_names:
        shard = load_file(INPUT_DIR / shard_name, device="cpu")
        expected_shard_keys = {
            key for key, filename in weight_map.items() if filename == shard_name
        }
        if set(shard) != expected_shard_keys:
            missing = sorted(expected_shard_keys - set(shard))
            extra = sorted(set(shard) - expected_shard_keys)
            raise ValueError(
                f"Index/shard mismatch in {shard_name}: missing={missing}, extra={extra}"
            )

        for old_name, tensor in shard.items():
            loaded_input_keys.add(old_name)
            match = LAYER_RE.fullmatch(old_name)
            if match is None:
                new_name = old_name
            else:
                old_layer = int(match.group(1))
                if old_layer in REMOVED_LAYERS:
                    continue
                if old_layer not in LAYER_MAP:
                    raise ValueError(f"Unexpected input layer index in {old_name}")
                new_name = f"model.layers.{LAYER_MAP[old_layer]}.{match.group(2)}"

            if new_name in output:
                raise ValueError(f"Rename collision at {new_name}")
            output[new_name] = tensor

    if loaded_input_keys != set(weight_map):
        raise ValueError("Did not load exactly the tensors listed in the index")

    # Perform every required structural check before writing the destination.
    forbidden = [
        name
        for name in output
        if (match := LAYER_RE.fullmatch(name))
        and int(match.group(1)) in {12, 13, 14, 15}
    ]
    if forbidden:
        raise ValueError(f"Forbidden output layers remain: {sorted(forbidden)}")

    q_proj_layers = {
        int(match.group(1))
        for name in output
        if (match := Q_PROJ_RE.fullmatch(name))
    }
    if q_proj_layers != set(range(12)):
        raise ValueError(
            "Expected exactly one q_proj tensor for each layer 0..11; "
            f"found layers {sorted(q_proj_layers)}"
        )
    q_proj_count = sum(Q_PROJ_RE.fullmatch(name) is not None for name in output)
    if q_proj_count != 12:
        raise ValueError(f"Expected 12 q_proj tensors, found {q_proj_count}")

    if len(output) != 86:
        raise ValueError(f"Expected 86 output tensors, found {len(output)}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_file(output, OUTPUT_PATH)
    print(f"Wrote {len(output)} tensors across 12 layers to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
