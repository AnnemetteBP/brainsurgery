#!/usr/bin/env python3
"""Remove selected Pythia blocks and contiguously renumber the survivors."""

import os
import re
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import load_file, save_file


INPUT = Path("inputs/base/model.safetensors")
OUTPUT = Path("out/T1/model.safetensors")
TEMP = OUTPUT.with_suffix(".safetensors.tmp")
LAYER_RE = re.compile(r"^gpt_neox\.layers\.(\d+)\.(.+)$")
DROP = {2, 6, 10, 14}
SURVIVORS = [i for i in range(16) if i not in DROP]
RENUMBER = {old: new for new, old in enumerate(SURVIVORS)}


def layer_index(key: str) -> int | None:
    match = LAYER_RE.fullmatch(key)
    return int(match.group(1)) if match else None


def validate_keys(keys: set[str]) -> None:
    indices = {index for key in keys if (index := layer_index(key)) is not None}
    assert not (indices & {12, 13, 14, 15}), (
        f"forbidden output block indices remain: {sorted(indices & {12, 13, 14, 15})}"
    )
    assert indices == set(range(12)), f"output block indices are {sorted(indices)}, expected 0..11"

    qkv_pattern = re.compile(
        r"^gpt_neox\.layers\.(\d+)\.attention\.query_key_value\.weight$"
    )
    qkv_keys = [key for key in keys if qkv_pattern.fullmatch(key)]
    assert len(qkv_keys) == 12, f"found {len(qkv_keys)} block QKV weights, expected 12"
    assert len(keys) == 184, f"found {len(keys)} tensors, expected 184"


def main() -> None:
    source = load_file(INPUT, device="cpu")
    assert len(source) == 244, f"input has {len(source)} tensors, expected 244"

    output = {}
    seen_input_layers = set()
    for old_key, tensor in source.items():
        match = LAYER_RE.fullmatch(old_key)
        if match is None:
            new_key = old_key
        else:
            old_index = int(match.group(1))
            seen_input_layers.add(old_index)
            if old_index in DROP:
                continue
            assert old_index in RENUMBER, f"unexpected input block index {old_index}"
            new_key = f"gpt_neox.layers.{RENUMBER[old_index]}.{match.group(2)}"

        assert new_key not in output, f"destination-key collision: {new_key}"
        output[new_key] = tensor

    assert seen_input_layers == set(range(16)), (
        f"input block indices are {sorted(seen_input_layers)}, expected 0..15"
    )
    validate_keys(set(output))

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    try:
        if TEMP.exists():
            TEMP.unlink()
        save_file(output, TEMP)
        with safe_open(TEMP, framework="pt", device="cpu") as saved:
            saved_keys = set(saved.keys())
        validate_keys(saved_keys)
        assert saved_keys == set(output), "saved key set differs from the planned key set"
        os.replace(TEMP, OUTPUT)
    except BaseException:
        TEMP.unlink(missing_ok=True)
        raise

    print(f"wrote {OUTPUT} with {len(output)} tensors and blocks 0..11")


if __name__ == "__main__":
    main()
