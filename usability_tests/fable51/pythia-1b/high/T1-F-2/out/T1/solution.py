"""T1: depth-prune Pythia-1B from 16 to 12 blocks, renumbering contiguously.

Plain safetensors + torch script. All checks run on the in-memory result
before anything is written; the file is written to a temp path and renamed,
so a failed check leaves no output behind.
"""
import os
import re
import sys

import torch
from safetensors.torch import load_file, save_file

SRC = "inputs/base/model.safetensors"
DST = "out/T1/model.safetensors"
DROP = {2, 6, 10, 14}
N_OLD, N_NEW = 16, 12
EXPECTED_TENSORS = 184
BLOCK_RE = re.compile(r"^gpt_neox\.layers\.(\d+)\.(.+)$")


def fail(msg):
    print(f"FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


def block_of(name):
    m = BLOCK_RE.match(name)
    return int(m.group(1)) if m else None


src = load_file(SRC)
if len(src) != 244:
    fail(f"expected 244 input tensors, got {len(src)}")

old_blocks = sorted({b for b in map(block_of, src) if b is not None})
if old_blocks != list(range(N_OLD)):
    fail(f"input blocks are not 0..{N_OLD - 1}: {old_blocks}")

# Surviving blocks in original order -> new contiguous indices.
survivors = [b for b in range(N_OLD) if b not in DROP]
remap = {old: new for new, old in enumerate(survivors)}
assert len(remap) == N_NEW

# Build a fresh dict (no in-place renames -> no collision hazard). Still
# guard against duplicate destination names explicitly.
out = {}
for name, t in src.items():
    b = block_of(name)
    if b is None:
        new_name = name  # non-block tensor, unchanged
    elif b in DROP:
        continue
    else:
        new_name = BLOCK_RE.sub(lambda m: f"gpt_neox.layers.{remap[b]}.{m.group(2)}", name)
    if new_name in out:
        fail(f"destination collision: {new_name}")
    out[new_name] = t.contiguous()

# ---- Required checks (on the in-memory result, before writing) ----
new_blocks = sorted({b for b in map(block_of, out) if b is not None})
bad = [b for b in new_blocks if b >= N_NEW]
if bad:
    fail(f"tensors of blocks >= {N_NEW} remain: {bad}")
if new_blocks != list(range(N_NEW)):
    fail(f"blocks are not exactly 0..{N_NEW - 1}: {new_blocks}")
qkv = [n for n in out if BLOCK_RE.match(n) and n.endswith(".attention.query_key_value.weight")]
if len(qkv) != N_NEW:
    fail(f"expected {N_NEW} query_key_value.weight tensors, got {len(qkv)}")
if len(out) != EXPECTED_TENSORS:
    fail(f"expected {EXPECTED_TENSORS} output tensors, got {len(out)}")

# ---- Extra sanity: every kept tensor is bit-identical to its source ----
for old, new in remap.items():
    for name, t in src.items():
        if block_of(name) == old:
            nn = BLOCK_RE.sub(lambda m: f"gpt_neox.layers.{new}.{m.group(2)}", name)
        elif block_of(name) is None:
            nn = name
        else:
            continue
        o = out[nn]
        if o.shape != t.shape or o.dtype != t.dtype or not torch.equal(o, t):
            fail(f"value/shape/dtype mismatch for {name} -> {nn}")

# ---- Write atomically ----
tmp = DST + ".tmp"
os.makedirs(os.path.dirname(DST), exist_ok=True)
save_file(out, tmp, metadata={"format": "pt"})
os.replace(tmp, DST)

# ---- Verify the written file ----
chk = load_file(DST)
if len(chk) != EXPECTED_TENSORS or set(chk) != set(out):
    fail("written file does not match the in-memory result")
print(f"OK: wrote {DST} with {len(chk)} tensors, blocks {new_blocks[0]}..{new_blocks[-1]}")
