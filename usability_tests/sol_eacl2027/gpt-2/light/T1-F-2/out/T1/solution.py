#!/usr/bin/env python3
"""Remove GPT-2 blocks 2, 5, and 8 and compact the surviving indices."""

from __future__ import annotations

import os
import re
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import load_file, save_file


SOURCE = Path("inputs/base/model.safetensors")
OUTPUT = Path("out/T1/model.safetensors")
TEMP = OUTPUT.with_suffix(".safetensors.tmp")
BLOCK_KEY = re.compile(r"^h\.(\d+)\.(.+)$")
REMOVED = {2, 5, 8}
SURVIVORS = [0, 1, 3, 4, 6, 7, 9, 10, 11]
RENUMBER = {old: new for new, old in enumerate(SURVIVORS)}


def validate_keys(keys: set[str]) -> None:
    """Enforce all task-level structural checks before publishing output."""
    block_indices = {
        int(match.group(1))
        for key in keys
        if (match := BLOCK_KEY.fullmatch(key)) is not None
    }
    assert not ({9, 10, 11} & block_indices), (
        f"unrenumbered source blocks remain: {sorted({9, 10, 11} & block_indices)}"
    )
    assert block_indices == set(range(9)), (
        f"expected block indices 0..8, got {sorted(block_indices)}"
    )
    attention_weights = [
        key for key in keys if re.fullmatch(r"h\.[0-8]\.attn\.c_attn\.weight", key)
    ]
    assert len(attention_weights) == 9, (
        f"expected 9 c_attn weights, got {len(attention_weights)}"
    )
    assert len(keys) == 121, f"expected 121 tensors, got {len(keys)}"


def main() -> None:
    source = load_file(SOURCE)
    assert len(source) == 160, f"expected 160 source tensors, got {len(source)}"

    result = {}
    for old_key, tensor in source.items():
        match = BLOCK_KEY.fullmatch(old_key)
        if match is None:
            new_key = old_key
        else:
            old_index = int(match.group(1))
            assert old_index in set(range(12)), f"unexpected block index in {old_key}"
            if old_index in REMOVED:
                continue
            new_key = f"h.{RENUMBER[old_index]}.{match.group(2)}"

        assert new_key not in result, f"destination-key collision: {new_key}"
        result[new_key] = tensor

    validate_keys(set(result))

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    try:
        save_file(result, TEMP)
        with safe_open(TEMP, framework="pt", device="cpu") as checkpoint:
            written_keys = set(checkpoint.keys())
        validate_keys(written_keys)
        assert written_keys == set(result), "saved key set differs from candidate"
        os.replace(TEMP, OUTPUT)
    finally:
        TEMP.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
