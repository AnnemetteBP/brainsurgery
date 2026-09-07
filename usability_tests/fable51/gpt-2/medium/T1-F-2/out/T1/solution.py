"""T1: depth-prune GPT-2 (124M) from 12 to 9 blocks, dropping blocks 2, 5, 8.

Builds a fresh state dict from an explicit old->new index map (no in-place
renames, so no collision hazard), verifies the required checks, then saves.
Fails with a non-zero exit and no output written if any check fails.
"""
import os
import re
import sys

from safetensors.torch import load_file, save_file

SRC = "inputs/base/model.safetensors"
DST = "out/T1/model.safetensors"
REMOVE = {2, 5, 8}
N_OLD = 12
BLOCK_RE = re.compile(r"^h\.(\d+)\.(.+)$")


def fail(msg):
    print(f"FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


src = load_file(SRC)
if len(src) != 160:
    fail(f"expected 160 input tensors, got {len(src)}")

keep = [i for i in range(N_OLD) if i not in REMOVE]
old_to_new = {old: new for new, old in enumerate(keep)}

out = {}
for name, t in src.items():
    m = BLOCK_RE.match(name)
    if m is None:
        out[name] = t  # non-block tensor, unchanged
        continue
    old = int(m.group(1))
    if old in REMOVE:
        continue
    if old not in old_to_new:
        fail(f"unexpected block index {old} in {name}")
    new_name = f"h.{old_to_new[old]}.{m.group(2)}"
    if new_name in out:
        fail(f"collision: {new_name} already present")
    out[new_name] = t

# Required checks.
n_keep = len(keep)
for name in out:
    m = BLOCK_RE.match(name)
    if m and int(m.group(1)) >= n_keep:
        fail(f"tensor of block >= {n_keep} remains: {name}")
attn_w = [k for k in out if re.fullmatch(r"h\.\d+\.attn\.c_attn\.weight", k)]
if len(attn_w) != n_keep:
    fail(f"expected {n_keep} blocks, found {len(attn_w)} c_attn.weight tensors")
block_ids = sorted({int(BLOCK_RE.match(k).group(1)) for k in out if BLOCK_RE.match(k)})
if block_ids != list(range(n_keep)):
    fail(f"block indices not contiguous 0..{n_keep - 1}: {block_ids}")
if len(out) != 121:
    fail(f"expected 121 output tensors, got {len(out)}")

# Value/shape/dtype fidelity against the source under the mapping.
for old, new in old_to_new.items():
    for k in src:
        m = BLOCK_RE.match(k)
        if m and int(m.group(1)) == old:
            a, b = src[k], out[f"h.{new}.{m.group(2)}"]
            if a.shape != b.shape or a.dtype != b.dtype or not (a == b).all():
                fail(f"mismatch mapping {k} -> h.{new}.{m.group(2)}")
for k in ("wte.weight", "wpe.weight", "ln_f.weight", "ln_f.bias"):
    if k not in out or not (out[k] == src[k]).all():
        fail(f"non-block tensor changed or missing: {k}")

os.makedirs(os.path.dirname(DST), exist_ok=True)
save_file({k: v.contiguous() for k, v in out.items()}, DST)
print(f"OK: wrote {DST} with {len(out)} tensors, blocks {block_ids}")
