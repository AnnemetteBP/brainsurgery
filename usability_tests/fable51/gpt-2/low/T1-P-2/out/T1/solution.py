"""T1: depth-prune GPT-2 (124M) from 12 to 9 blocks, renumbering contiguously."""
import os
import re
import sys

from safetensors.torch import load_file, save_file

SRC = "inputs/base/model.safetensors"
DST = "out/T1/model.safetensors"
DROP = {2, 5, 8}
N_OLD, N_NEW = 12, 9
BLOCK_RE = re.compile(r"^h\.(\d+)\.(.+)$")


def fail(msg):
    print(f"FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


sd = load_file(SRC)
if len(sd) != 160:
    fail(f"expected 160 input tensors, got {len(sd)}")

keep = [i for i in range(N_OLD) if i not in DROP]
remap = {old: new for new, old in enumerate(keep)}  # old index -> new index

out = {}
for name, t in sd.items():
    m = BLOCK_RE.match(name)
    if m is None:
        out[name] = t  # non-block tensor, unchanged
        continue
    old = int(m.group(1))
    if old in DROP:
        continue
    new_name = f"h.{remap[old]}.{m.group(2)}"
    if new_name in out:
        fail(f"collision: {new_name} already present")
    out[new_name] = t

# Required checks
for name in out:
    m = BLOCK_RE.match(name)
    if m and int(m.group(1)) >= N_NEW:
        fail(f"tensor of block >= {N_NEW} remains: {name}")
n_blocks = sum(1 for n in out if re.fullmatch(r"h\.\d+\.attn\.c_attn\.weight", n))
if n_blocks != N_NEW:
    fail(f"expected {N_NEW} blocks, found {n_blocks}")
block_ids = sorted({int(BLOCK_RE.match(n).group(1)) for n in out if BLOCK_RE.match(n)})
if block_ids != list(range(N_NEW)):
    fail(f"block indices not contiguous 0..{N_NEW - 1}: {block_ids}")
if len(out) != 121:
    fail(f"expected 121 output tensors, got {len(out)}")
# Each surviving block has exactly 13 tensors with values identical to the source.
for old, new in remap.items():
    for name, t in sd.items():
        if name.startswith(f"h.{old}."):
            nn_ = f"h.{new}." + name[len(f"h.{old}."):]
            if nn_ not in out or out[nn_].data_ptr() != t.data_ptr():
                fail(f"mapping mismatch {name} -> {nn_}")

os.makedirs(os.path.dirname(DST), exist_ok=True)
save_file({k: v.contiguous() for k, v in out.items()}, DST)
print(f"wrote {DST} with {len(out)} tensors; blocks {block_ids}")
