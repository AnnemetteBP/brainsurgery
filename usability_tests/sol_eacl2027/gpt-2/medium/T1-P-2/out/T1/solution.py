#!/usr/bin/env python3
"""Remove GPT-2 blocks 2, 5, and 8 and compact the remaining indices."""

import re
from pathlib import Path

from safetensors.torch import load_file, save_file


INPUT_PATH = Path("inputs/base/model.safetensors")
OUTPUT_PATH = Path("out/T1/model.safetensors")
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


def main() -> None:
    source = load_file(INPUT_PATH)
    result = {}

    for old_name, tensor in source.items():
        match = BLOCK_KEY.match(old_name)
        if match is None:
            new_name = old_name
        else:
            old_index = int(match.group(1))
            if old_index in {2, 5, 8}:
                continue
            if old_index not in OLD_TO_NEW:
                raise RuntimeError(f"unexpected transformer block in input: {old_index}")
            new_name = f"h.{OLD_TO_NEW[old_index]}.{match.group(2)}"

        if new_name in result:
            raise RuntimeError(f"renaming collision at output key: {new_name}")
        result[new_name] = tensor

    output_block_indices = {
        int(match.group(1))
        for name in result
        if (match := BLOCK_KEY.match(name)) is not None
    }
    if output_block_indices & {9, 10, 11}:
        raise RuntimeError("output still contains a tensor from block 9, 10, or 11")

    attention_weight_indices = {
        int(match.group(1))
        for name in result
        if (match := re.fullmatch(r"h\.(\d+)\.attn\.c_attn\.weight", name))
        is not None
    }
    if attention_weight_indices != set(range(9)):
        raise RuntimeError(
            "expected exactly one c_attn weight for each block 0..8; "
            f"found indices {sorted(attention_weight_indices)}"
        )

    if len(result) != 121:
        raise RuntimeError(f"expected 121 output tensors, found {len(result)}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_file(result, OUTPUT_PATH)


if __name__ == "__main__":
    main()
