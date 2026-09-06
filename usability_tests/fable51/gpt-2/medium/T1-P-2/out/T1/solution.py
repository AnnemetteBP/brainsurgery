"""T1: depth-prune GPT-2 (124M) from 12 to 9 blocks, removing blocks 2, 5, 8.

Loads inputs/base/model.safetensors, drops every tensor of the removed
blocks, renumbers survivors contiguously, verifies, then writes
out/T1/model.safetensors. Any failed check raises before anything is written.
"""

import os
import re
import sys

from safetensors.torch import load_file, save_file

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
SRC = os.path.join(ROOT, "inputs", "base", "model.safetensors")
DST = os.path.join(HERE, "model.safetensors")

REMOVE = {2, 5, 8}
N_OLD = 12
N_NEW = N_OLD - len(REMOVE)
TENSORS_PER_BLOCK = 13
NON_BLOCK = {"wte.weight", "wpe.weight", "ln_f.weight", "ln_f.bias"}
EXPECTED_TOTAL = N_NEW * TENSORS_PER_BLOCK + len(NON_BLOCK)  # 121

BLOCK_RE = re.compile(r"^h\.(\d+)\.(.+)$")


def fail(msg: str) -> None:
    print(f"FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    sd = load_file(SRC)
    if len(sd) != 160:
        fail(f"expected 160 input tensors, got {len(sd)}")

    # old index -> new index, preserving original order.
    survivors = [i for i in range(N_OLD) if i not in REMOVE]
    remap = {old: new for new, old in enumerate(survivors)}

    out: dict = {}
    for name, t in sd.items():
        m = BLOCK_RE.match(name)
        if m is None:
            if name not in NON_BLOCK:
                fail(f"unexpected non-block tensor {name!r}")
            out[name] = t
            continue
        old = int(m.group(1))
        if old >= N_OLD:
            fail(f"block index {old} out of range in {name!r}")
        if old in REMOVE:
            continue
        new_name = f"h.{remap[old]}.{m.group(2)}"
        if new_name in out:
            fail(f"collision: {new_name!r} already written (from {name!r})")
        out[new_name] = t

    # ---- Required checks -------------------------------------------------
    stale = [k for k in out if (m := BLOCK_RE.match(k)) and int(m.group(1)) >= N_NEW]
    if stale:
        fail(f"tensors of blocks >= {N_NEW} remain: {stale[:5]}")

    c_attn = sorted(int(BLOCK_RE.match(k).group(1)) for k in out if k.endswith(".attn.c_attn.weight"))
    if c_attn != list(range(N_NEW)):
        fail(f"expected c_attn.weight for blocks 0..{N_NEW - 1}, got {c_attn}")

    if len(out) != EXPECTED_TOTAL:
        fail(f"expected {EXPECTED_TOTAL} output tensors, got {len(out)}")

    # ---- Extra sanity: each surviving block is complete and bit-identical ---
    for old, new in remap.items():
        old_keys = {k for k in sd if BLOCK_RE.match(k) and int(BLOCK_RE.match(k).group(1)) == old}
        if len(old_keys) != TENSORS_PER_BLOCK:
            fail(f"input block {old} has {len(old_keys)} tensors, expected {TENSORS_PER_BLOCK}")
        for k in old_keys:
            nk = f"h.{new}." + BLOCK_RE.match(k).group(2)
            if nk not in out or out[nk].data_ptr() != sd[k].data_ptr():
                fail(f"{k!r} -> {nk!r} not carried over unchanged")
    for k in NON_BLOCK:
        if k not in out or out[k].data_ptr() != sd[k].data_ptr():
            fail(f"non-block tensor {k!r} changed or missing")

    save_file({k: v.contiguous() for k, v in out.items()}, DST)

    # Verify the written file.
    back = load_file(DST)
    if set(back) != set(out) or len(back) != EXPECTED_TOTAL:
        fail("written file does not match expected key set")
    for k, v in back.items():
        if v.shape != out[k].shape or v.dtype != out[k].dtype or not (v == out[k]).all():
            fail(f"written tensor {k!r} differs from source")
    print(f"OK: wrote {DST} with {len(back)} tensors, blocks 0..{N_NEW - 1}")


if __name__ == "__main__":
    main()
