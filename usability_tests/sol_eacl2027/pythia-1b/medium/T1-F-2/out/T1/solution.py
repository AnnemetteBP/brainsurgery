#!/usr/bin/env python3
"""Remove four Pythia blocks and densely renumber the survivors."""

import os
import re
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import load_file, save_file


INPUT = Path("inputs/base/model.safetensors")
OUTPUT = Path("out/T1/model.safetensors")
TEMP_OUTPUT = OUTPUT.with_suffix(".safetensors.tmp")
LAYER_RE = re.compile(r"^gpt_neox\.layers\.(\d+)\.(.+)$")
QKV_RE = re.compile(
    r"^gpt_neox\.layers\.(\d+)\.attention\.query_key_value\.weight$"
)
DROP = {2, 6, 10, 14}
SURVIVORS = [i for i in range(16) if i not in DROP]
RENUMBER = {old: new for new, old in enumerate(SURVIVORS)}


def validate_keys(tensors: dict[str, object]) -> None:
    """Enforce all required output checks plus the expected layer key shape."""
    if len(tensors) != 184:
        raise ValueError(f"expected 184 tensors, found {len(tensors)}")

    layer_indices = {
        int(match.group(1))
        for key in tensors
        if (match := LAYER_RE.match(key)) is not None
    }
    if layer_indices != set(range(12)):
        raise ValueError(f"expected layer indices 0..11, found {sorted(layer_indices)}")
    if layer_indices & {12, 13, 14, 15}:
        raise ValueError("one or more tensors from output blocks 12..15 remain")

    qkv_indices = {
        int(match.group(1))
        for key in tensors
        if (match := QKV_RE.match(key)) is not None
    }
    if qkv_indices != set(range(12)) or len(qkv_indices) != 12:
        raise ValueError(
            "expected exactly one query_key_value.weight tensor in each block 0..11"
        )


def main() -> None:
    source = load_file(INPUT, device="cpu")
    if len(source) != 244:
        raise ValueError(f"expected 244 input tensors, found {len(source)}")

    result = {}
    dropped_count = 0
    for old_key, tensor in source.items():
        match = LAYER_RE.match(old_key)
        if match is None:
            new_key = old_key
        else:
            old_layer = int(match.group(1))
            if old_layer in DROP:
                dropped_count += 1
                continue
            if old_layer not in RENUMBER:
                raise ValueError(f"unexpected input layer index {old_layer}: {old_key}")
            new_key = f"gpt_neox.layers.{RENUMBER[old_layer]}.{match.group(2)}"

        if new_key in result:
            raise ValueError(f"key collision while creating {new_key}")
        result[new_key] = tensor

    if dropped_count != 60:
        raise ValueError(f"expected to remove 60 block tensors, removed {dropped_count}")
    validate_keys(result)

    metadata = None
    with safe_open(INPUT, framework="pt", device="cpu") as handle:
        metadata = handle.metadata()

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    TEMP_OUTPUT.unlink(missing_ok=True)
    try:
        save_file(result, TEMP_OUTPUT, metadata=metadata)
        written = load_file(TEMP_OUTPUT, device="cpu")
        validate_keys(written)
        for key, tensor in result.items():
            candidate = written[key]
            if candidate.dtype != tensor.dtype or candidate.shape != tensor.shape:
                raise ValueError(f"shape or dtype changed for {key}")
            if not candidate.equal(tensor):
                raise ValueError(f"tensor values changed for {key}")
        os.replace(TEMP_OUTPUT, OUTPUT)
    except BaseException:
        TEMP_OUTPUT.unlink(missing_ok=True)
        OUTPUT.unlink(missing_ok=True)
        raise

    print(f"wrote {OUTPUT} with {len(result)} tensors and layers 0..11")


if __name__ == "__main__":
    main()
