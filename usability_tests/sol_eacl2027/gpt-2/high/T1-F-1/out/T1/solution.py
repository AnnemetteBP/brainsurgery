#!/usr/bin/env python3
"""Remove GPT-2 blocks 2, 5, and 8 and compact the remaining block names."""

from __future__ import annotations

import os
import re
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import load_file, save_file


INPUT = Path("inputs/base/model.safetensors")
OUTPUT_DIR = Path("out/T1")
OUTPUT = OUTPUT_DIR / "model.safetensors"
TEMP_OUTPUT = OUTPUT_DIR / ".model.safetensors.tmp"
BLOCK_KEY = re.compile(r"^h\.(\d+)\.(.+)$")
REMOVED = {2, 5, 8}
SURVIVORS = [0, 1, 3, 4, 6, 7, 9, 10, 11]
RENUMBER = {old: new for new, old in enumerate(SURVIVORS)}


def validate_keys(keys: set[str]) -> None:
    """Enforce every required postcondition before the final file is published."""
    parsed = [(key, BLOCK_KEY.match(key)) for key in keys]
    block_indices = {int(match.group(1)) for _, match in parsed if match}

    assert not (block_indices & {9, 10, 11}), (
        "output still contains tensors from blocks 9, 10, or 11"
    )
    assert block_indices == set(range(9)), (
        f"expected contiguous block indices 0..8, got {sorted(block_indices)}"
    )

    attention_weights = [
        key
        for key in keys
        if re.fullmatch(r"h\.\d+\.attn\.c_attn\.weight", key)
    ]
    assert len(attention_weights) == 9, (
        f"expected 9 block attention weights, got {len(attention_weights)}"
    )
    assert len(keys) == 121, f"expected 121 tensors, got {len(keys)}"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_OUTPUT.unlink(missing_ok=True)

    source = load_file(INPUT, device="cpu")
    source_blocks: dict[int, list[str]] = {}
    non_block_keys: set[str] = set()
    for key in source:
        match = BLOCK_KEY.match(key)
        if match:
            source_blocks.setdefault(int(match.group(1)), []).append(key)
        else:
            non_block_keys.add(key)

    assert set(source_blocks) == set(range(12)), (
        f"expected source blocks 0..11, got {sorted(source_blocks)}"
    )
    assert all(len(keys) == 13 for keys in source_blocks.values()), (
        "every source block must own exactly 13 tensors"
    )
    assert len(non_block_keys) == 4, (
        f"expected 4 non-block tensors, got {len(non_block_keys)}"
    )
    assert len(source) == 160, f"expected 160 source tensors, got {len(source)}"

    result = {}
    for old_index in SURVIVORS:
        new_index = RENUMBER[old_index]
        for old_key in source_blocks[old_index]:
            match = BLOCK_KEY.match(old_key)
            assert match is not None
            new_key = f"h.{new_index}.{match.group(2)}"
            assert new_key not in result, f"destination collision: {new_key}"
            result[new_key] = source[old_key]

    for key in non_block_keys:
        assert key not in result, f"destination collision: {key}"
        result[key] = source[key]

    # Validate before writing anything and then validate the serialized file too.
    validate_keys(set(result))
    try:
        save_file(result, TEMP_OUTPUT)
        with safe_open(TEMP_OUTPUT, framework="pt", device="cpu") as checkpoint:
            written_keys = set(checkpoint.keys())
        validate_keys(written_keys)
        assert written_keys == set(result), "serialized key set differs from result"
        os.replace(TEMP_OUTPUT, OUTPUT)
    except BaseException:
        TEMP_OUTPUT.unlink(missing_ok=True)
        raise

    print(f"wrote {OUTPUT} with {len(result)} tensors and blocks 0..8")


if __name__ == "__main__":
    main()
