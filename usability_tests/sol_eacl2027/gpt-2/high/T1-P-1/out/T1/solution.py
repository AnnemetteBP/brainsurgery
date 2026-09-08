from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path

from safetensors.torch import load_file, save_file


INPUT = Path("inputs/base/model.safetensors")
OUTPUT = Path("out/T1/model.safetensors")
BLOCK_KEY = re.compile(r"^h\.(\d+)\.(.+)$")
OLD_TO_NEW = {
    0: 0,
    1: 1,
    3: 2,
    4: 3,
    6: 4,
    7: 5,
    9: 6,
    10: 7,
    11: 8,
}
REMOVED = {2, 5, 8}


def fail(message: str) -> None:
    raise RuntimeError(message)


def main() -> None:
    source = load_file(INPUT, device="cpu")
    result = {}
    source_block_counts = {i: 0 for i in range(12)}

    for old_key, tensor in source.items():
        match = BLOCK_KEY.fullmatch(old_key)
        if match is None:
            new_key = old_key
        else:
            old_index = int(match.group(1))
            if old_index not in source_block_counts:
                fail(f"unexpected source block index in {old_key!r}")
            source_block_counts[old_index] += 1
            if old_index in REMOVED:
                continue
            if old_index not in OLD_TO_NEW:
                fail(f"source block {old_index} is neither retained nor removed")
            new_key = f"h.{OLD_TO_NEW[old_index]}.{match.group(2)}"

        if new_key in result:
            fail(f"renaming collision at {new_key!r}")
        result[new_key] = tensor

    if set(OLD_TO_NEW) | REMOVED != set(range(12)):
        fail("the retain/remove mapping does not cover exactly blocks 0..11")
    if any(count != 13 for count in source_block_counts.values()):
        fail(f"expected 13 tensors in every source block, got {source_block_counts}")

    output_block_indices = {
        int(match.group(1))
        for key in result
        if (match := BLOCK_KEY.fullmatch(key)) is not None
    }
    if output_block_indices & {9, 10, 11}:
        fail("output still contains a tensor from block 9, 10, or 11")
    if output_block_indices != set(range(9)):
        fail(f"expected output block indices 0..8, got {sorted(output_block_indices)}")

    c_attn_weights = [
        key
        for key in result
        if re.fullmatch(r"h\.[0-8]\.attn\.c_attn\.weight", key)
    ]
    if len(c_attn_weights) != 9:
        fail(f"expected 9 block c_attn weights, got {len(c_attn_weights)}")
    if len(result) != 121:
        fail(f"expected exactly 121 output tensors, got {len(result)}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=OUTPUT.parent, prefix=".model.", suffix=".safetensors", delete=False
        ) as temp_file:
            temp_path = Path(temp_file.name)
        save_file(result, temp_path)
        os.replace(temp_path, OUTPUT)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


if __name__ == "__main__":
    main()
