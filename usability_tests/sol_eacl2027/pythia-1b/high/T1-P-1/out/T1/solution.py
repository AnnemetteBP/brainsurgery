#!/usr/bin/env python3
"""Remove four Pythia transformer blocks and compact the layer numbering."""

import os
import re
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


INPUT_PATH = Path("inputs/base/model.safetensors")
OUTPUT_PATH = Path("out/T1/model.safetensors")
TEMP_PATH = Path("out/T1/model.safetensors.tmp")
REMOVED_LAYERS = {2, 6, 10, 14}
LAYER_RE = re.compile(r"gpt_neox\.layers\.(\d+)\.(.+)")
QKV_WEIGHT_RE = re.compile(
    r"gpt_neox\.layers\.(\d+)\.attention\.query_key_value\.weight"
)


def main() -> None:
    source = load_file(str(INPUT_PATH), device="cpu")

    source_layer_counts: dict[int, int] = {}
    non_block_count = 0
    for name, tensor in source.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} is not a torch.Tensor")
        match = LAYER_RE.fullmatch(name)
        if match is None:
            non_block_count += 1
            continue
        old_index = int(match.group(1))
        source_layer_counts[old_index] = source_layer_counts.get(old_index, 0) + 1

    if set(source_layer_counts) != set(range(16)):
        raise ValueError(
            f"expected source layers 0..15, found {sorted(source_layer_counts)}"
        )
    if any(count != 15 for count in source_layer_counts.values()):
        raise ValueError(f"expected 15 tensors per source layer: {source_layer_counts}")
    if non_block_count != 4 or len(source) != 244:
        raise ValueError(
            f"unexpected source inventory: {len(source)} tensors, "
            f"{non_block_count} non-block tensors"
        )

    kept_layers = [i for i in range(16) if i not in REMOVED_LAYERS]
    old_to_new = {old: new for new, old in enumerate(kept_layers)}
    output: dict[str, torch.Tensor] = {}

    for old_name, tensor in source.items():
        match = LAYER_RE.fullmatch(old_name)
        if match is None:
            new_name = old_name
        else:
            old_index = int(match.group(1))
            if old_index in REMOVED_LAYERS:
                continue
            new_name = f"gpt_neox.layers.{old_to_new[old_index]}.{match.group(2)}"

        if new_name in output:
            raise ValueError(f"renaming collision at {new_name}")
        output[new_name] = tensor

    # Required post-transform checks, all completed before anything is saved.
    forbidden = [
        name
        for name in output
        if (match := LAYER_RE.fullmatch(name))
        and int(match.group(1)) in {12, 13, 14, 15}
    ]
    if forbidden:
        raise ValueError(f"forbidden output layers 12..15 remain: {forbidden[:5]}")

    qkv_indices = sorted(
        int(match.group(1))
        for name in output
        if (match := QKV_WEIGHT_RE.fullmatch(name))
    )
    if qkv_indices != list(range(12)):
        raise ValueError(
            "expected exactly one QKV weight for each layer 0..11, "
            f"found indices {qkv_indices}"
        )
    if len(output) != 184:
        raise ValueError(f"expected 184 output tensors, found {len(output)}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        save_file(output, str(TEMP_PATH))
        os.replace(TEMP_PATH, OUTPUT_PATH)
    finally:
        TEMP_PATH.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
