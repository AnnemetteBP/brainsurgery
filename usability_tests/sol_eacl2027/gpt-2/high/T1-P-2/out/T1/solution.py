from __future__ import annotations

import os
import re
from pathlib import Path

from safetensors.torch import load_file, save_file


INPUT_PATH = Path("inputs/base/model.safetensors")
OUTPUT_PATH = Path("out/T1/model.safetensors")
TEMP_PATH = OUTPUT_PATH.with_suffix(".safetensors.tmp")
BLOCK_KEY = re.compile(r"^h\.(\d+)\.(.+)$")
REMOVED_BLOCKS = {2, 5, 8}
SURVIVING_BLOCKS = [0, 1, 3, 4, 6, 7, 9, 10, 11]
OLD_TO_NEW = {old: new for new, old in enumerate(SURVIVING_BLOCKS)}


def main() -> None:
    source = load_file(INPUT_PATH, device="cpu")
    result = {}

    for old_name, tensor in source.items():
        match = BLOCK_KEY.fullmatch(old_name)
        if match is None:
            new_name = old_name
        else:
            old_index = int(match.group(1))
            if old_index in REMOVED_BLOCKS:
                continue
            if old_index not in OLD_TO_NEW:
                raise AssertionError(f"unexpected source block index: {old_index}")
            new_name = f"h.{OLD_TO_NEW[old_index]}.{match.group(2)}"

        if new_name in result:
            raise AssertionError(f"renaming collision at {new_name}")
        result[new_name] = tensor

    output_block_indices = {
        int(match.group(1))
        for name in result
        if (match := BLOCK_KEY.fullmatch(name)) is not None
    }
    stale_keys = [
        name
        for name in result
        if (match := BLOCK_KEY.fullmatch(name)) is not None
        and int(match.group(1)) in {9, 10, 11}
    ]
    anchor_keys = [
        name
        for name in result
        if re.fullmatch(r"h\.\d+\.attn\.c_attn\.weight", name)
    ]

    assert not stale_keys, f"old high-index block tensors remain: {stale_keys[:3]}"
    assert output_block_indices == set(range(9)), (
        f"expected contiguous block indices 0..8, got {sorted(output_block_indices)}"
    )
    assert len(anchor_keys) == 9, (
        f"expected 9 attention-weight block anchors, got {len(anchor_keys)}"
    )
    assert len(result) == 121, f"expected 121 tensors, got {len(result)}"

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    TEMP_PATH.unlink(missing_ok=True)
    try:
        save_file(result, TEMP_PATH)
        os.replace(TEMP_PATH, OUTPUT_PATH)
    finally:
        TEMP_PATH.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
