#!/usr/bin/env python3
"""Drop GPT-2 blocks 2, 5, and 8 and contiguously renumber survivors."""

from __future__ import annotations

import os
import re
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import load_file, save_file


INPUT = Path("inputs/base/model.safetensors")
OUTPUT = Path("out/T1/model.safetensors")
TEMP_OUTPUT = OUTPUT.with_suffix(".safetensors.tmp")
BLOCK_KEY = re.compile(r"^h\.(\d+)\.(.+)$")
REMOVED = {2, 5, 8}
SURVIVORS = [0, 1, 3, 4, 6, 7, 9, 10, 11]
RENUMBER = {old: new for new, old in enumerate(SURVIVORS)}


def check_result(tensors: dict[str, object]) -> None:
    """Enforce the task's required checks plus collision/contiguity checks."""
    keys = set(tensors)
    assert len(keys) == len(tensors), "duplicate output keys"
    assert len(keys) == 121, f"expected 121 tensors, found {len(keys)}"

    block_indices = {
        int(match.group(1))
        for key in keys
        if (match := BLOCK_KEY.fullmatch(key)) is not None
    }
    assert not (block_indices & {9, 10, 11}), (
        "output still contains a tensor from block 9, 10, or 11"
    )
    assert block_indices == set(range(9)), (
        f"expected contiguous blocks 0..8, found {sorted(block_indices)}"
    )

    attention_weights = [
        key
        for key in keys
        if re.fullmatch(r"h\.\d+\.attn\.c_attn\.weight", key)
    ]
    assert len(attention_weights) == 9, (
        "expected exactly 9 h.<i>.attn.c_attn.weight tensors, "
        f"found {len(attention_weights)}"
    )


def main() -> None:
    source = load_file(INPUT, device="cpu")
    result = {}

    for old_key, tensor in source.items():
        match = BLOCK_KEY.fullmatch(old_key)
        if match is None:
            new_key = old_key
        else:
            old_index = int(match.group(1))
            if old_index in REMOVED:
                continue
            assert old_index in RENUMBER, f"unexpected source block {old_index}"
            new_key = f"h.{RENUMBER[old_index]}.{match.group(2)}"

        assert new_key not in result, f"rename collision at {new_key}"
        result[new_key] = tensor

    # All required checks happen before any output file is written.
    check_result(result)

    metadata = None
    with safe_open(INPUT, framework="pt", device="cpu") as source_file:
        metadata = source_file.metadata()

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    try:
        save_file(result, TEMP_OUTPUT, metadata=metadata)
        persisted = load_file(TEMP_OUTPUT, device="cpu")
        check_result(persisted)
        assert set(persisted) == set(result), "saved key set differs from result"
        os.replace(TEMP_OUTPUT, OUTPUT)
    finally:
        TEMP_OUTPUT.unlink(missing_ok=True)

    print(f"wrote {OUTPUT} with {len(result)} tensors")


if __name__ == "__main__":
    main()
