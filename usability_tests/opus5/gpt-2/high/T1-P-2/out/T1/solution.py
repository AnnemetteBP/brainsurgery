"""T1: depth-prune GPT-2 (124M) from 12 to 9 transformer blocks.

Drops blocks 2, 5 and 8 and renumbers the survivors so that block indices are
contiguous (0..8) again, preserving every name suffix, value, shape and dtype.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

HERE = Path(__file__).resolve().parent
SANDBOX = HERE.parent.parent
SRC = SANDBOX / "inputs" / "base" / "model.safetensors"
DST = HERE / "model.safetensors"

DROP = {2, 5, 8}
N_BLOCKS_IN = 12
TENSORS_PER_BLOCK = 13
NON_BLOCK = {"wte.weight", "wpe.weight", "ln_f.weight", "ln_f.bias"}

# "h.<idx>.<rest>"; the index group is anchored so that nothing else matches.
BLOCK_RE = re.compile(r"^h\.(\d+)\.(.+)$")


def fail(msg: str) -> None:
    print(f"CHECK FAILED: {msg}", file=sys.stderr)
    raise SystemExit(1)


def main() -> None:
    if not SRC.exists():
        fail(f"input checkpoint not found: {SRC}")

    # --- read source, recording dtypes/shapes straight from the file header ---
    with safe_open(str(SRC), framework="pt") as f:
        src_keys = list(f.keys())
    src = load_file(str(SRC))
    if set(src) != set(src_keys):
        fail("safe_open key set differs from loaded key set")

    src_meta = {k: (tuple(v.shape), v.dtype) for k, v in src.items()}

    # --- classify source keys ---
    blocks: dict[int, dict[str, str]] = {}  # old index -> {suffix: full key}
    others: set[str] = set()
    for key in src:
        m = BLOCK_RE.match(key)
        if m is None:
            others.add(key)
            continue
        idx, rest = int(m.group(1)), m.group(2)
        if rest in blocks.setdefault(idx, {}):
            fail(f"duplicate suffix {rest!r} in block {idx}")
        blocks[idx][rest] = key

    # --- sanity on the input itself ---
    if others != NON_BLOCK:
        fail(f"unexpected non-block tensors: {sorted(others ^ NON_BLOCK)}")
    if sorted(blocks) != list(range(N_BLOCKS_IN)):
        fail(f"expected blocks 0..{N_BLOCKS_IN - 1}, got {sorted(blocks)}")
    for idx, members in blocks.items():
        if len(members) != TENSORS_PER_BLOCK:
            fail(f"block {idx} has {len(members)} tensors, expected {TENSORS_PER_BLOCK}")
    expected_in = N_BLOCKS_IN * TENSORS_PER_BLOCK + len(NON_BLOCK)
    if len(src) != expected_in:
        fail(f"input has {len(src)} tensors, expected {expected_in}")
    if not DROP <= set(blocks):
        fail(f"blocks to drop {sorted(DROP)} are not all present")

    # --- old -> new index map: survivors keep their relative order ---
    survivors = [i for i in sorted(blocks) if i not in DROP]
    remap = {old: new for new, old in enumerate(survivors)}
    print("block remap (old -> new):", ", ".join(f"{o}->{n}" for o, n in remap.items()))

    # --- build the output dict; a fresh dict makes collisions impossible, but
    #     assert it anyway so a bad map cannot silently overwrite a block ---
    out: dict[str, torch.Tensor] = {}
    for key in NON_BLOCK:
        out[key] = src[key]
    for old, new in remap.items():
        for rest, key in blocks[old].items():
            new_key = f"h.{new}.{rest}"
            if new_key in out:
                fail(f"renumbering collision: {key} -> {new_key} already written")
            out[new_key] = src[key]

    # --- required checks, before anything is written ---
    n_expected = len(survivors) * TENSORS_PER_BLOCK + len(NON_BLOCK)

    stale = sorted(k for k in out if (m := BLOCK_RE.match(k)) and int(m.group(1)) >= len(survivors))
    if stale:
        fail(f"tensors of removed block indices {len(survivors)}..{N_BLOCKS_IN - 1} remain: {stale}")

    n_attn = sum(1 for k in out if re.fullmatch(r"h\.\d+\.attn\.c_attn\.weight", k))
    if n_attn != len(survivors):
        fail(f"{n_attn} blocks remain (by attn.c_attn.weight), expected {len(survivors)}")

    out_idx = sorted({int(m.group(1)) for k in out if (m := BLOCK_RE.match(k))})
    if out_idx != list(range(len(survivors))):
        fail(f"block indices are not contiguous 0..{len(survivors) - 1}: {out_idx}")

    if len(out) != n_expected:
        fail(f"output has {len(out)} tensors, expected {n_expected}")

    # values, shapes and dtypes must be carried over untouched
    for old, new in remap.items():
        for rest, key in blocks[old].items():
            new_key = f"h.{new}.{rest}"
            if src_meta[key] != (tuple(out[new_key].shape), out[new_key].dtype):
                fail(f"shape/dtype changed for {key} -> {new_key}")
            if out[new_key].data_ptr() != src[key].data_ptr():
                fail(f"{new_key} is not the original tensor of {key}")
    for key in NON_BLOCK:
        if out[key].data_ptr() != src[key].data_ptr():
            fail(f"non-block tensor {key} was replaced")

    # --- write atomically so a failure never leaves a partial output ---
    tmp = DST.with_suffix(".safetensors.tmp")
    save_file(out, str(tmp), metadata={"format": "pt"})
    os.replace(tmp, DST)

    # --- verify what actually landed on disk ---
    back = load_file(str(DST))
    if set(back) != set(out):
        fail("written key set differs from the intended one")
    if len(back) != n_expected:
        fail(f"written file has {len(back)} tensors, expected {n_expected}")
    for old, new in remap.items():
        for rest, key in blocks[old].items():
            new_key = f"h.{new}.{rest}"
            got = back[new_key]
            if (tuple(got.shape), got.dtype) != src_meta[key]:
                fail(f"on disk: shape/dtype mismatch for {new_key}")
            if not torch.equal(got, src[key]):
                fail(f"on disk: values differ for {new_key} (from {key})")
    for key in NON_BLOCK:
        if (tuple(back[key].shape), back[key].dtype) != src_meta[key]:
            fail(f"on disk: shape/dtype mismatch for {key}")
        if not torch.equal(back[key], src[key]):
            fail(f"on disk: values differ for {key}")

    print(f"OK: wrote {DST} with {len(back)} tensors, {len(survivors)} blocks")


if __name__ == "__main__":
    main()
