#!/usr/bin/env python
"""T1: depth-prune GPT-2 (124M) from 12 blocks to 9 and renumber contiguously.

Route: safetensors for I/O, torch-state-bridge for the block-index rewrite
(rule-based renaming with collision detection).  All required checks run on the
in-memory result *before* anything is written, and are re-verified against the
file after writing; any failure raises and leaves no output behind.
"""

from __future__ import annotations

import os
import re
import sys

from safetensors.torch import load_file, save_file
from torch_state_bridge import state_bridge

IN_PATH = "inputs/base/model.safetensors"
OUT_DIR = "out/T1"
OUT_PATH = os.path.join(OUT_DIR, "model.safetensors")

DROP = (2, 5, 8)
N_BLOCKS_IN = 12
N_BLOCKS_OUT = 9
TENSORS_PER_BLOCK = 13
NON_BLOCK = ("wte.weight", "wpe.weight", "ln_f.weight", "ln_f.bias")
EXPECTED_OUT_TENSORS = N_BLOCKS_OUT * TENSORS_PER_BLOCK + len(NON_BLOCK)  # 121

BLOCK_RE = re.compile(r"^h\.(\d+)\.(.+)$")


class CheckFailed(Exception):
    """A required check did not hold."""


def check(cond: bool, msg: str) -> None:
    if not cond:
        raise CheckFailed(msg)


def main() -> None:
    src = load_file(IN_PATH)

    # --- input sanity -----------------------------------------------------
    check(len(src) == N_BLOCKS_IN * TENSORS_PER_BLOCK + len(NON_BLOCK),
          f"input has {len(src)} tensors, expected 160")
    in_blocks = sorted({int(m.group(1)) for k in src if (m := BLOCK_RE.match(k))})
    check(in_blocks == list(range(N_BLOCKS_IN)),
          f"input blocks are {in_blocks}, expected 0..11")

    # --- 1. drop whole blocks 2, 5, 8 -------------------------------------
    kept = {k: v for k, v in src.items()
            if not ((m := BLOCK_RE.match(k)) and int(m.group(1)) in DROP)}
    check(len(kept) == len(src) - len(DROP) * TENSORS_PER_BLOCK,
          f"dropping {DROP} removed {len(src) - len(kept)} tensors, "
          f"expected {len(DROP) * TENSORS_PER_BLOCK}")

    # --- 2. renumber survivors, in original order, without collisions -----
    # Each old index is rewritten to a temporary "H." namespace first, so no
    # rule can cascade onto a key another rule has already renumbered.
    survivors = [i for i in range(N_BLOCKS_IN) if i not in DROP]
    check(survivors == sorted(survivors) and len(set(survivors)) == N_BLOCKS_OUT,
          f"survivors {survivors} are not the 9 remaining blocks in original order")
    mapping = {old: new for new, old in enumerate(survivors)}
    rules = "\n".join(f"h.{old}., H.{new}." for old, new in mapping.items())
    rules += "\nH.{n}., h.{n}."
    out = state_bridge(kept, rules, detect_collision=True)

    # --- required checks, before writing anything -------------------------
    expected_keys = {f"h.{mapping[int(m.group(1))]}.{m.group(2)}"
                     for k in kept if (m := BLOCK_RE.match(k))}
    expected_keys |= set(NON_BLOCK)
    check(set(out) == expected_keys,
          "renamed key set differs from the expected mapping: "
          f"missing={sorted(expected_keys - set(out))[:5]} "
          f"unexpected={sorted(set(out) - expected_keys)[:5]}")

    stale = sorted(k for k in out
                   if (m := BLOCK_RE.match(k)) and int(m.group(1)) >= N_BLOCKS_OUT)
    check(not stale, f"tensors of blocks >= {N_BLOCKS_OUT} remain: {stale[:5]}")

    n_attn = sum(1 for k in out if re.fullmatch(r"h\.\d+\.attn\.c_attn\.weight", k))
    check(n_attn == N_BLOCKS_OUT,
          f"{n_attn} blocks remain (by h.<i>.attn.c_attn.weight), expected {N_BLOCKS_OUT}")
    out_blocks = sorted({int(m.group(1)) for k in out if (m := BLOCK_RE.match(k))})
    check(out_blocks == list(range(N_BLOCKS_OUT)),
          f"block indices are {out_blocks}, expected 0..{N_BLOCKS_OUT - 1}")

    check(len(out) == EXPECTED_OUT_TENSORS,
          f"output has {len(out)} tensors, expected {EXPECTED_OUT_TENSORS}")

    # values, shapes and dtypes must be untouched
    for new_key, t in out.items():
        if (m := BLOCK_RE.match(new_key)):
            new_i, rest = int(m.group(1)), m.group(2)
            old_key = f"h.{survivors[new_i]}.{rest}"
        else:
            old_key = new_key
        o = src[old_key]
        check(t.shape == o.shape and t.dtype == o.dtype,
              f"{new_key}: {tuple(t.shape)}/{t.dtype} != {old_key}'s "
              f"{tuple(o.shape)}/{o.dtype}")
        check(t.data_ptr() == o.data_ptr() or bool((t == o).all()),
              f"{new_key}: values differ from {old_key}")
    for k in NON_BLOCK:
        check(k in out and out[k].data_ptr() == src[k].data_ptr(),
              f"non-block tensor {k} was not carried over unchanged")

    # --- write (atomically: no partial output on failure) -----------------
    os.makedirs(OUT_DIR, exist_ok=True)
    if os.path.exists(OUT_PATH):  # a failed attempt must not leave a stale result
        os.remove(OUT_PATH)
    tmp = OUT_PATH + ".tmp"
    save_file({k: v.contiguous() for k, v in out.items()}, tmp, metadata={"format": "pt"})

    back = load_file(tmp)
    check(len(back) == EXPECTED_OUT_TENSORS,
          f"written file has {len(back)} tensors, expected {EXPECTED_OUT_TENSORS}")
    check(set(back) == expected_keys, "written file has the wrong key set")
    for k, t in back.items():
        check(t.dtype == out[k].dtype and t.shape == out[k].shape and bool((t == out[k]).all()),
              f"written tensor {k} does not match")
    os.replace(tmp, OUT_PATH)

    print(f"OK: wrote {OUT_PATH} with {len(back)} tensors, "
          f"{N_BLOCKS_OUT} blocks (dropped {DROP}, remapped "
          f"{ {o: n for o, n in mapping.items()} })")


if __name__ == "__main__":
    try:
        main()
    except CheckFailed as e:
        print(f"CHECK FAILED: {e}", file=sys.stderr)
        if os.path.exists(OUT_PATH + ".tmp"):
            os.remove(OUT_PATH + ".tmp")
        sys.exit(1)
