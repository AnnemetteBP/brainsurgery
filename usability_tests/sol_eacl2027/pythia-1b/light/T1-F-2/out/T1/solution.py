#!/usr/bin/env python3
"""Remove four Pythia blocks and contiguously renumber the survivors."""

import os
import re
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file


SOURCE = Path("inputs/base/model.safetensors")
OUTPUT = Path("out/T1/model.safetensors")
TEMP = OUTPUT.with_suffix(".safetensors.tmp")
LAYER_RE = re.compile(r"^gpt_neox\.layers\.(\d+)\.(.+)$")
DROP = {2, 6, 10, 14}
SURVIVORS = [i for i in range(16) if i not in DROP]
RENUMBER = {old: new for new, old in enumerate(SURVIVORS)}


def validate_keys(keys: set[str]) -> None:
    """Enforce all task-required structural checks before publishing output."""
    layer_indices = {
        int(match.group(1))
        for key in keys
        if (match := LAYER_RE.fullmatch(key))
    }
    forbidden = layer_indices & {12, 13, 14, 15}
    if forbidden:
        raise RuntimeError(f"forbidden output block indices remain: {sorted(forbidden)}")

    qkv_weights = [
        key
        for key in keys
        if re.fullmatch(
            r"gpt_neox\.layers\.\d+\.attention\.query_key_value\.weight", key
        )
    ]
    if len(qkv_weights) != 12:
        raise RuntimeError(f"expected 12 block QKV weights, found {len(qkv_weights)}")
    if layer_indices != set(range(12)):
        raise RuntimeError(f"expected contiguous blocks 0..11, found {sorted(layer_indices)}")
    if len(keys) != 184:
        raise RuntimeError(f"expected 184 tensors, found {len(keys)}")


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    TEMP.unlink(missing_ok=True)

    tensors = {}
    with safe_open(SOURCE, framework="pt", device="cpu") as source:
        source_keys = list(source.keys())
        if len(source_keys) != 244:
            raise RuntimeError(f"expected 244 input tensors, found {len(source_keys)}")

        for old_key in source_keys:
            match = LAYER_RE.fullmatch(old_key)
            if match:
                old_index = int(match.group(1))
                if old_index in DROP:
                    continue
                if old_index not in RENUMBER:
                    raise RuntimeError(f"unexpected source block index {old_index}")
                new_key = f"gpt_neox.layers.{RENUMBER[old_index]}.{match.group(2)}"
            else:
                new_key = old_key

            if new_key in tensors:
                raise RuntimeError(f"rename collision at {new_key}")
            tensors[new_key] = source.get_tensor(old_key)

    validate_keys(set(tensors))
    save_file(tensors, TEMP)

    with safe_open(TEMP, framework="pt", device="cpu") as written:
        validate_keys(set(written.keys()))
    os.replace(TEMP, OUTPUT)
    print(f"wrote {OUTPUT} with {len(tensors)} tensors")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        TEMP.unlink(missing_ok=True)
        OUTPUT.unlink(missing_ok=True)
        raise
