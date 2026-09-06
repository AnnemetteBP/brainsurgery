#!/usr/bin/env python
"""T1: depth-prune Pythia-1B from 16 to 12 blocks, renumbering survivors.

Route: torch-state-bridge rule-based key rewriting (collision detection on)
over a lazily loaded safetensors state dict, then safetensors save.

The renumbering is done in two rule stages through a placeholder token
(`gpt_neox.__block__<new>.`) so that no rule's destination can be re-matched
by a later rule's source -- that cascade is exactly how a naive
"old -> new" rule list silently overwrites surviving blocks.

Every required check runs before anything is written; the file is written to
a temporary path, re-read and compared bit-exactly against the input, and
only then moved into place. Any failure exits non-zero with no output file.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch_state_bridge import state_bridge, state_bridge_preview

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent  # sandbox root
SRC = ROOT / "inputs" / "base" / "model.safetensors"
DST = HERE / "model.safetensors"
TMP = HERE / "model.safetensors.tmp"

DROP = [2, 6, 10, 14]
N_OLD = 16
N_NEW = 12
EXPECTED_TENSORS = 184
PER_BLOCK = 15

BLOCK_RE = re.compile(r"^gpt_neox\.layers\.(\d+)\.")


def die(msg: str) -> None:
    print(f"FAIL: {msg}", file=sys.stderr)
    if TMP.exists():
        TMP.unlink()
    raise SystemExit(1)


def main() -> None:
    keep = [i for i in range(N_OLD) if i not in DROP]
    if len(keep) != N_NEW:
        die(f"expected {N_NEW} surviving blocks, got {len(keep)}")
    remap = {old: new for new, old in enumerate(keep)}

    with safe_open(str(SRC), framework="pt") as f:
        src_keys = list(f.keys())
        if len(src_keys) != N_OLD * PER_BLOCK + 4:
            die(f"input has {len(src_keys)} tensors, expected {N_OLD * PER_BLOCK + 4}")

        # --- select: drop every tensor of the pruned blocks -------------------
        pruned = {}
        for k in src_keys:
            m = BLOCK_RE.match(k)
            if m is not None and int(m.group(1)) in DROP:
                continue
            pruned[k] = f.get_tensor(k)

    dropped = len(src_keys) - len(pruned)
    if dropped != len(DROP) * PER_BLOCK:
        die(f"dropped {dropped} tensors, expected {len(DROP) * PER_BLOCK}")

    # --- rename: two staged rule sets, no destination re-matched by a source --
    stage1 = "\n".join(
        f"gpt_neox.layers.{old}., gpt_neox.__block__{new}." for old, new in remap.items()
    )
    stage2 = "\n".join(f"gpt_neox.__block__{n}., gpt_neox.layers.{n}." for n in range(N_NEW))

    _, _, collisions = state_bridge_preview(pruned, stage1)
    if collisions:
        die(f"rule collisions while renumbering: {sorted(collisions)}")

    staged = state_bridge(pruned, stage1, detect_collision=True)
    new_sd = state_bridge(staged, stage2, detect_collision=True)

    # --- required checks (all before any write) ---------------------------
    for k in new_sd:
        if "__block__" in k:
            die(f"placeholder token survived renaming: {k}")

    indices = sorted({int(m.group(1)) for k in new_sd if (m := BLOCK_RE.match(k))})

    stale = [i for i in indices if i >= N_NEW]
    if stale:
        die(f"tensors of removed block indices remain: {stale}")

    qkv = [k for k in new_sd if BLOCK_RE.match(k) and k.endswith("attention.query_key_value.weight")]
    if len(qkv) != N_NEW:
        die(f"{len(qkv)} query_key_value.weight tensors, expected exactly {N_NEW}")

    if indices != list(range(N_NEW)):
        die(f"block indices are not contiguous 0..{N_NEW - 1}: {indices}")

    for i in indices:
        n = sum(1 for k in new_sd if k.startswith(f"gpt_neox.layers.{i}."))
        if n != PER_BLOCK:
            die(f"block {i} has {n} tensors, expected {PER_BLOCK}")

    non_block = sorted(k for k in new_sd if not BLOCK_RE.match(k))
    expected_non_block = sorted(k for k in src_keys if not BLOCK_RE.match(k))
    if non_block != expected_non_block:
        die(f"non-block tensors changed: {non_block} != {expected_non_block}")

    if len(new_sd) != EXPECTED_TENSORS:
        die(f"output would have {len(new_sd)} tensors, expected exactly {EXPECTED_TENSORS}")

    # explicit expected key set, derived independently of the rule engine
    expected = set(expected_non_block)
    for old, new in remap.items():
        for k in src_keys:
            if k.startswith(f"gpt_neox.layers.{old}."):
                expected.add(f"gpt_neox.layers.{new}." + k[len(f"gpt_neox.layers.{old}.") :])
    if set(new_sd) != expected:
        die("output key set does not match the expected renumbering")

    # --- write, then verify the file on disk bit-exactly ------------------
    save_file({k: v.contiguous() for k, v in new_sd.items()}, str(TMP))

    inv = {}
    for old, new in remap.items():
        inv[f"gpt_neox.layers.{new}."] = f"gpt_neox.layers.{old}."

    with safe_open(str(TMP), framework="pt") as g, safe_open(str(SRC), framework="pt") as f:
        got = list(g.keys())
        if len(got) != EXPECTED_TENSORS:
            die(f"written file has {len(got)} tensors, expected {EXPECTED_TENSORS}")
        for k in got:
            m = BLOCK_RE.match(k)
            if m is None:
                origin = k
            else:
                pre = f"gpt_neox.layers.{m.group(1)}."
                origin = inv[pre] + k[len(pre) :]
            a, b = g.get_tensor(k), f.get_tensor(origin)
            if a.dtype != b.dtype or a.shape != b.shape or not torch.equal(a, b):
                die(f"value/shape/dtype mismatch for {k} (from {origin})")

    os.replace(TMP, DST)
    print(f"OK: wrote {DST} with {EXPECTED_TENSORS} tensors, {N_NEW} blocks 0..{N_NEW - 1}")


if __name__ == "__main__":
    main()
