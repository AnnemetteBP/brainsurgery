#!/usr/bin/env python3
"""Remove GPT-2 blocks 2, 5, and 8 and compact the remaining indices."""

import os
import re
from pathlib import Path

from safetensors.torch import load_file, save_file


INPUT = Path("inputs/base/model.safetensors")
OUTPUT = Path("out/T1/model.safetensors")
TEMP_OUTPUT = OUTPUT.with_suffix(".safetensors.tmp")
BLOCK_KEY = re.compile(r"^h\.(\d+)\.(.+)$")
DROPPED = {2, 5, 8}
SURVIVORS = [0, 1, 3, 4, 6, 7, 9, 10, 11]
RENUMBER = {old: new for new, old in enumerate(SURVIVORS)}


def main() -> None:
    if OUTPUT.exists() or TEMP_OUTPUT.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing output: {OUTPUT} or {TEMP_OUTPUT}"
        )

    source = load_file(INPUT, device="cpu")
    result = {}
    source_for_output = {}

    for old_name, tensor in source.items():
        match = BLOCK_KEY.fullmatch(old_name)
        if match is None:
            new_name = old_name
        else:
            old_index = int(match.group(1))
            if old_index in DROPPED:
                continue
            if old_index not in RENUMBER:
                raise AssertionError(f"Unexpected source block index {old_index}: {old_name}")
            new_name = f"h.{RENUMBER[old_index]}.{match.group(2)}"

        if new_name in result:
            raise AssertionError(
                f"Rename collision: {old_name} and {source_for_output[new_name]} -> {new_name}"
            )
        result[new_name] = tensor
        source_for_output[new_name] = old_name

    # Validate the complete transformation before creating any output file.
    output_block_indices = {
        int(match.group(1))
        for name in result
        if (match := BLOCK_KEY.fullmatch(name)) is not None
    }
    assert output_block_indices == set(range(9)), output_block_indices
    assert not any(
        name.startswith(("h.9.", "h.10.", "h.11.")) for name in result
    ), "Old high block indices remain in output"

    anchors = [
        name
        for name in result
        if re.fullmatch(r"h\.[0-8]\.attn\.c_attn\.weight", name)
    ]
    assert len(anchors) == 9, f"Expected 9 block anchors, found {len(anchors)}"
    assert len(result) == 121, f"Expected 121 tensors, found {len(result)}"

    block_counts = {
        index: sum(name.startswith(f"h.{index}.") for name in result)
        for index in range(9)
    }
    assert set(block_counts.values()) == {13}, block_counts

    non_block_names = {name for name in source if BLOCK_KEY.fullmatch(name) is None}
    assert non_block_names == {"wte.weight", "wpe.weight", "ln_f.weight", "ln_f.bias"}
    assert all(result[name] is source[name] for name in non_block_names)

    try:
        save_file(result, TEMP_OUTPUT, metadata={"format": "pt"})
        os.replace(TEMP_OUTPUT, OUTPUT)
    except BaseException:
        TEMP_OUTPUT.unlink(missing_ok=True)
        raise

    print(f"Wrote {OUTPUT} with {len(result)} tensors and blocks 0..8")


if __name__ == "__main__":
    main()
