#!/usr/bin/env python3
"""Remove GPT-2 blocks 2, 5, and 8 and compact the remaining block names."""

import re
from pathlib import Path

from safetensors.torch import load_file, save_file


SOURCE = Path("inputs/base/model.safetensors")
DESTINATION = Path("out/T1/model.safetensors")
BLOCK_KEY = re.compile(r"^h\.(\d+)\.(.+)$")
SURVIVORS = (0, 1, 3, 4, 6, 7, 9, 10, 11)
RENUMBER = {old: new for new, old in enumerate(SURVIVORS)}


def main() -> None:
    source = load_file(SOURCE, device="cpu")
    result = {}

    for key, tensor in source.items():
        match = BLOCK_KEY.fullmatch(key)
        if match is None:
            new_key = key
        else:
            old_index = int(match.group(1))
            if old_index not in RENUMBER:
                continue
            new_key = f"h.{RENUMBER[old_index]}.{match.group(2)}"

        if new_key in result:
            raise RuntimeError(f"rename collision at {new_key!r}")
        result[new_key] = tensor

    output_block_indices = {
        int(match.group(1))
        for key in result
        if (match := BLOCK_KEY.fullmatch(key)) is not None
    }
    c_attn_weights = [
        key for key in result if re.fullmatch(r"h\.\d+\.attn\.c_attn\.weight", key)
    ]

    # Required checks happen before save_file, so a bad transformation writes nothing.
    if output_block_indices & {9, 10, 11}:
        raise RuntimeError("output still contains block 9, 10, or 11")
    if output_block_indices != set(range(9)) or len(c_attn_weights) != 9:
        raise RuntimeError(
            f"expected exactly blocks 0..8 and 9 c_attn weights; got "
            f"{sorted(output_block_indices)} and {len(c_attn_weights)} weights"
        )
    if len(result) != 121:
        raise RuntimeError(f"expected 121 output tensors, got {len(result)}")

    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    temporary = DESTINATION.with_suffix(".safetensors.tmp")
    save_file(result, temporary)
    temporary.replace(DESTINATION)


if __name__ == "__main__":
    main()
