#!/usr/bin/env python3
"""Drop Pythia blocks 2, 6, 10, and 14 and compact the layer indices."""

import os
import re
import sys
import tempfile
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file


INPUT = Path("inputs/base/model.safetensors")
OUTPUT = Path("out/T1/model.safetensors")
LAYER_RE = re.compile(r"^gpt_neox\.layers\.(\d+)\.(.+)$")
DROP = {2, 6, 10, 14}
SURVIVORS = [i for i in range(16) if i not in DROP]
RENUMBER = {old: new for new, old in enumerate(SURVIVORS)}


def fail(message: str) -> None:
    raise RuntimeError(message)


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    # Never allow an earlier artifact to masquerade as this run's result.
    OUTPUT.unlink(missing_ok=True)

    rewritten = {}
    source_counts = {i: 0 for i in range(16)}
    non_block_count = 0

    with safe_open(INPUT, framework="pt", device="cpu") as source:
        input_keys = list(source.keys())
        if len(input_keys) != 244:
            fail(f"expected 244 input tensors, found {len(input_keys)}")

        for old_key in input_keys:
            match = LAYER_RE.match(old_key)
            if match is None:
                new_key = old_key
                non_block_count += 1
            else:
                old_layer = int(match.group(1))
                if old_layer not in source_counts:
                    fail(f"unexpected input layer index {old_layer}: {old_key}")
                source_counts[old_layer] += 1
                if old_layer in DROP:
                    continue
                new_key = f"gpt_neox.layers.{RENUMBER[old_layer]}.{match.group(2)}"

            if new_key in rewritten:
                fail(f"rewrite collision at {new_key}")
            rewritten[new_key] = source.get_tensor(old_key)

    if non_block_count != 4:
        fail(f"expected 4 non-block tensors, found {non_block_count}")
    if any(count != 15 for count in source_counts.values()):
        fail(f"expected 15 tensors in every input block, found {source_counts}")

    output_layers = []
    for key in rewritten:
        match = LAYER_RE.match(key)
        if match:
            output_layers.append(int(match.group(1)))

    # Required checks, performed before any output is written.
    if any(layer in {12, 13, 14, 15} for layer in output_layers):
        fail("output still contains a tensor from block 12, 13, 14, or 15")
    qkv_weight_re = re.compile(
        r"^gpt_neox\.layers\.(\d+)\.attention\.query_key_value\.weight$"
    )
    qkv_layers = sorted(
        int(match.group(1))
        for key in rewritten
        if (match := qkv_weight_re.match(key))
    )
    if qkv_layers != list(range(12)):
        fail(f"expected one QKV weight in each block 0..11, found {qkv_layers}")
    if set(output_layers) != set(range(12)):
        fail(f"expected exactly blocks 0..11, found {sorted(set(output_layers))}")
    if len(rewritten) != 184:
        fail(f"expected 184 output tensors, found {len(rewritten)}")

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix="model.", suffix=".safetensors.tmp", dir=OUTPUT.parent, delete=False
        ) as temporary:
            temp_path = Path(temporary.name)
        save_file(rewritten, temp_path)
        os.replace(temp_path, OUTPUT)
        temp_path = None
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)

    print(f"wrote {OUTPUT} with {len(rewritten)} tensors and layers 0..11")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
