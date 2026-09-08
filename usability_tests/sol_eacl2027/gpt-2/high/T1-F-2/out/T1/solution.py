#!/usr/bin/env python3
"""Remove GPT-2 blocks 2, 5, and 8 and contiguously renumber survivors."""

from __future__ import annotations

import os
import re
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file


BLOCK_KEY = re.compile(r"^h\.(\d+)\.(.+)$")
REMOVED = {2, 5, 8}
SURVIVORS = [0, 1, 3, 4, 6, 7, 9, 10, 11]
RENUMBER = {old: new for new, old in enumerate(SURVIVORS)}
NON_BLOCK_KEYS = {"wte.weight", "wpe.weight", "ln_f.weight", "ln_f.bias"}
EXPECTED_TENSORS_PER_BLOCK = 13
EXPECTED_OUTPUT_TENSORS = 121


def split_key(key: str) -> tuple[int, str] | None:
    match = BLOCK_KEY.fullmatch(key)
    if match is None:
        return None
    return int(match.group(1)), match.group(2)


def validate_source(source: dict[str, torch.Tensor]) -> None:
    block_counts = {index: 0 for index in range(12)}
    non_block_keys: set[str] = set()

    for key in source:
        parsed = split_key(key)
        if parsed is None:
            non_block_keys.add(key)
            continue
        index, _ = parsed
        if index not in block_counts:
            raise ValueError(f"unexpected source block index in {key!r}")
        block_counts[index] += 1

    if non_block_keys != NON_BLOCK_KEYS:
        raise ValueError(
            f"unexpected non-block keys: got {sorted(non_block_keys)}, "
            f"expected {sorted(NON_BLOCK_KEYS)}"
        )
    if any(count != EXPECTED_TENSORS_PER_BLOCK for count in block_counts.values()):
        raise ValueError(f"source block tensor counts are not all 13: {block_counts}")
    if len(source) != 160:
        raise ValueError(f"source has {len(source)} tensors, expected 160")


def build_output(source: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    output: dict[str, torch.Tensor] = {}
    provenance: dict[str, str] = {}

    for old_key, tensor in source.items():
        parsed = split_key(old_key)
        if parsed is None:
            new_key = old_key
        else:
            old_index, suffix = parsed
            if old_index in REMOVED:
                continue
            new_key = f"h.{RENUMBER[old_index]}.{suffix}"

        if new_key in output:
            raise ValueError(
                f"key collision: {old_key!r} and {provenance[new_key]!r} both map to {new_key!r}"
            )
        output[new_key] = tensor
        provenance[new_key] = old_key

    return output, provenance


def validate_output(
    output: dict[str, torch.Tensor],
    source: dict[str, torch.Tensor],
    provenance: dict[str, str],
) -> None:
    parsed_keys = [(key, split_key(key)) for key in output]
    block_indices = {parsed[0] for _, parsed in parsed_keys if parsed is not None}

    if block_indices & {9, 10, 11}:
        raise ValueError("output still contains a tensor from block 9, 10, or 11")
    if block_indices != set(range(9)):
        raise ValueError(f"output block indices are {sorted(block_indices)}, expected 0..8")

    markers = [
        key
        for key, parsed in parsed_keys
        if parsed is not None and parsed[1] == "attn.c_attn.weight"
    ]
    if len(markers) != 9:
        raise ValueError(f"output has {len(markers)} block marker tensors, expected 9")
    if len(output) != EXPECTED_OUTPUT_TENSORS:
        raise ValueError(
            f"output has {len(output)} tensors, expected {EXPECTED_OUTPUT_TENSORS}"
        )

    for new_key, tensor in output.items():
        old_key = provenance[new_key]
        original = source[old_key]
        if tensor.shape != original.shape or tensor.dtype != original.dtype:
            raise ValueError(f"shape or dtype changed while mapping {old_key!r} to {new_key!r}")
        if not torch.equal(tensor, original):
            raise ValueError(f"tensor value changed while mapping {old_key!r} to {new_key!r}")


def main() -> None:
    task_dir = Path(__file__).resolve().parent
    sandbox = task_dir.parent.parent
    source_path = sandbox / "inputs" / "base" / "model.safetensors"
    output_path = task_dir / "model.safetensors"
    temporary_path = task_dir / ".model.safetensors.tmp"

    source = load_file(source_path, device="cpu")
    validate_source(source)
    output, provenance = build_output(source)
    validate_output(output, source, provenance)

    with safe_open(source_path, framework="pt", device="cpu") as handle:
        metadata = handle.metadata()

    try:
        save_file(output, temporary_path, metadata=metadata)
        written = load_file(temporary_path, device="cpu")
        validate_output(written, source, provenance)
        os.replace(temporary_path, output_path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise

    print(f"Wrote {output_path} with {len(output)} tensors and blocks 0..8")


if __name__ == "__main__":
    main()
