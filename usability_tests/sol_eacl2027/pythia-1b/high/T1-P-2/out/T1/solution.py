#!/usr/bin/env python3
"""Remove four Pythia transformer blocks and compact the layer indices."""

import re
from pathlib import Path

from safetensors.torch import load_file, save_file


INPUT_PATH = Path("inputs/base/model.safetensors")
OUTPUT_PATH = Path("out/T1/model.safetensors")
LAYER_RE = re.compile(r"^gpt_neox\.layers\.(\d+)\.(.+)$")
DROPPED_LAYERS = {2, 6, 10, 14}
SURVIVING_LAYERS = [i for i in range(16) if i not in DROPPED_LAYERS]
LAYER_MAP = {old: new for new, old in enumerate(SURVIVING_LAYERS)}


def main() -> None:
    source = load_file(INPUT_PATH, device="cpu")
    result = {}

    for old_name, tensor in source.items():
        match = LAYER_RE.match(old_name)
        if match is None:
            new_name = old_name
        else:
            old_layer = int(match.group(1))
            if old_layer in DROPPED_LAYERS:
                continue
            if old_layer not in LAYER_MAP:
                raise AssertionError(f"unexpected source layer index: {old_layer}")
            new_name = f"gpt_neox.layers.{LAYER_MAP[old_layer]}.{match.group(2)}"

        if new_name in result:
            raise AssertionError(f"renaming collision at {new_name}")
        result[new_name] = tensor

    # Validate the complete in-memory checkpoint before creating the output file.
    forbidden = [
        name
        for name in result
        if (match := LAYER_RE.match(name)) and int(match.group(1)) in {12, 13, 14, 15}
    ]
    if forbidden:
        raise AssertionError(f"forbidden layer indices remain: {forbidden[:5]}")

    qkv_weight_re = re.compile(
        r"^gpt_neox\.layers\.(\d+)\.attention\.query_key_value\.weight$"
    )
    qkv_layers = sorted(
        int(match.group(1))
        for name in result
        if (match := qkv_weight_re.match(name))
    )
    if qkv_layers != list(range(12)):
        raise AssertionError(
            f"expected exactly one QKV weight for each layer 0..11, got {qkv_layers}"
        )

    if len(result) != 184:
        raise AssertionError(f"expected 184 output tensors, got {len(result)}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_file(result, OUTPUT_PATH)


if __name__ == "__main__":
    main()
