#!/usr/bin/env python3
"""Remove four Pythia transformer blocks and densely renumber survivors."""

import os
import re
from pathlib import Path

from safetensors.torch import load_file, save_file


INPUT = Path("inputs/base/model.safetensors")
OUTPUT = Path("out/T1/model.safetensors")
DROPPED = {2, 6, 10, 14}
SURVIVORS = [i for i in range(16) if i not in DROPPED]
RENUMBER = {old: new for new, old in enumerate(SURVIVORS)}
LAYER_KEY = re.compile(r"^gpt_neox\.layers\.(\d+)\.(.+)$")
QKV_WEIGHT = re.compile(
    r"^gpt_neox\.layers\.(\d+)\.attention\.query_key_value\.weight$"
)


def main() -> None:
    source = load_file(str(INPUT), device="cpu")
    transformed = {}

    for old_name, tensor in source.items():
        match = LAYER_KEY.fullmatch(old_name)
        if match is None:
            new_name = old_name
        else:
            old_index = int(match.group(1))
            if old_index in DROPPED:
                continue
            if old_index not in RENUMBER:
                raise ValueError(f"unexpected input block index in {old_name!r}")
            new_name = f"gpt_neox.layers.{RENUMBER[old_index]}.{match.group(2)}"

        if new_name in transformed:
            raise ValueError(f"renaming collision at {new_name!r}")
        transformed[new_name] = tensor

    # Complete every required validation before creating the output file.
    forbidden = [
        name
        for name in transformed
        if (match := LAYER_KEY.fullmatch(name))
        and int(match.group(1)) in {12, 13, 14, 15}
    ]
    if forbidden:
        raise ValueError(f"forbidden output block indices remain: {forbidden[:3]}")

    qkv_indices = sorted(
        int(match.group(1))
        for name in transformed
        if (match := QKV_WEIGHT.fullmatch(name))
    )
    if qkv_indices != list(range(12)):
        raise ValueError(
            "expected one query_key_value.weight tensor in each block 0..11; "
            f"found indices {qkv_indices}"
        )

    if len(transformed) != 184:
        raise ValueError(f"expected 184 output tensors, found {len(transformed)}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix(OUTPUT.suffix + ".tmp")
    try:
        save_file(transformed, str(temporary))
        os.replace(temporary, OUTPUT)
    finally:
        if temporary.exists():
            temporary.unlink()

    print(f"Wrote {len(transformed)} tensors to {OUTPUT}")


if __name__ == "__main__":
    main()
