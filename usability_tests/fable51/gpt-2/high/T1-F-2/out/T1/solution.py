"""T1: depth-prune GPT-2 (124M) from 12 to 9 blocks by removing blocks 2, 5, 8
and renumbering the survivors contiguously. Plain safetensors + torch.

Fails loudly (non-zero exit, nothing left under out/T1) if any check fails.
"""
import os
import re
import sys

import torch
from safetensors.torch import load_file, save_file

SRC = "inputs/base/model.safetensors"
DST = "out/T1/model.safetensors"
TMP = DST + ".tmp"

REMOVE = {2, 5, 8}
N_IN, N_OUT = 12, 9
BLOCK_RE = re.compile(r"^h\.(\d+)\.(.+)$")


def fail(msg):
    print(f"FAIL: {msg}", file=sys.stderr)
    if os.path.exists(TMP):
        os.remove(TMP)
    sys.exit(1)


src = load_file(SRC)
if len(src) != 160:
    fail(f"expected 160 input tensors, got {len(src)}")

survivors = [i for i in range(N_IN) if i not in REMOVE]
if len(survivors) != N_OUT:
    fail(f"expected {N_OUT} surviving blocks, got {len(survivors)}")
renum = {old: new for new, old in enumerate(survivors)}  # old index -> new index

out = {}
for name, t in src.items():
    m = BLOCK_RE.match(name)
    if m is None:
        out[name] = t  # non-block tensor, unchanged
        continue
    old = int(m.group(1))
    if old >= N_IN:
        fail(f"unexpected block index in input: {name}")
    if old in REMOVE:
        continue
    new_name = f"h.{renum[old]}.{m.group(2)}"
    if new_name in out:
        fail(f"collision: {new_name} already written (from {name})")
    out[new_name] = t

# ---- Required checks on the result (before writing anything) ----
for name in out:
    m = BLOCK_RE.match(name)
    if m and int(m.group(1)) >= N_OUT:
        fail(f"tensor of block {m.group(1)} remains: {name}")
c_attn = [n for n in out if re.fullmatch(r"h\.\d+\.attn\.c_attn\.weight", n)]
if len(c_attn) != N_OUT:
    fail(f"expected exactly {N_OUT} blocks (c_attn.weight), got {len(c_attn)}")
blocks = sorted({int(BLOCK_RE.match(n).group(1)) for n in out if BLOCK_RE.match(n)})
if blocks != list(range(N_OUT)):
    fail(f"block indices not contiguous 0..{N_OUT - 1}: {blocks}")
if len(out) != 121:
    fail(f"expected 121 output tensors, got {len(out)}")
for i in range(N_OUT):
    if sum(1 for n in out if n.startswith(f"h.{i}.")) != 13:
        fail(f"block {i} does not have exactly 13 tensors")

# Value check: every output block tensor is bit-identical to its source tensor.
for name, t in out.items():
    m = BLOCK_RE.match(name)
    src_name = f"h.{survivors[int(m.group(1))]}.{m.group(2)}" if m else name
    s = src[src_name]
    if s.shape != t.shape or s.dtype != t.dtype or not torch.equal(s, t):
        fail(f"{name} differs from source {src_name}")

# ---- Write atomically, then verify by reloading ----
save_file({k: v.contiguous() for k, v in out.items()}, TMP, metadata={"format": "pt"})
chk = load_file(TMP)
if len(chk) != 121 or set(chk) != set(out):
    fail("reloaded output has wrong key set")
for k in out:
    if not torch.equal(chk[k], out[k]):
        fail(f"reloaded {k} differs")
os.replace(TMP, DST)
print(f"OK: wrote {DST} with {len(chk)} tensors, blocks {blocks}, removed {sorted(REMOVE)}")
