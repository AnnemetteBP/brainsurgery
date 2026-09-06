"""T1: drop GPT-2 blocks 2, 5, 8 and renumber the rest contiguously."""
import os
import re
import sys

from safetensors.torch import load_file, save_file

SRC = "inputs/base/model.safetensors"
DST = "out/T1/model.safetensors"
DROP = {2, 5, 8}
N_OLD, N_NEW = 12, 9
BLOCK = re.compile(r"^h\.(\d+)\.(.+)$")


def fail(msg):
    print(f"FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


sd = load_file(SRC)
if len(sd) != 160:
    fail(f"expected 160 input tensors, got {len(sd)}")

keep = [i for i in range(N_OLD) if i not in DROP]
remap = {old: new for new, old in enumerate(keep)}  # old->new, order preserved

out = {}
for k, v in sd.items():
    m = BLOCK.match(k)
    if m is None:
        out[k] = v  # non-block tensor, unchanged
        continue
    old = int(m.group(1))
    if old in DROP:
        continue
    nk = f"h.{remap[old]}.{m.group(2)}"
    if nk in out:
        fail(f"collision on {nk}")
    out[nk] = v

# Required checks (all before writing anything).
stale = [k for k in out if (m := BLOCK.match(k)) and int(m.group(1)) >= N_NEW]
if stale:
    fail(f"tensors of blocks >= {N_NEW} remain: {stale[:5]}")
blocks = sorted({int(m.group(1)) for k in out if (m := BLOCK.match(k))})
if blocks != list(range(N_NEW)):
    fail(f"expected blocks 0..{N_NEW-1}, got {blocks}")
n_attn = sum(1 for k in out if re.fullmatch(r"h\.\d+\.attn\.c_attn\.weight", k))
if n_attn != N_NEW:
    fail(f"expected {N_NEW} c_attn.weight tensors, got {n_attn}")
if len(out) != 121:
    fail(f"expected 121 output tensors, got {len(out)}")
for old, new in remap.items():
    for k in sd:
        m = BLOCK.match(k)
        if m and int(m.group(1)) == old:
            nk = f"h.{new}.{m.group(2)}"
            if out[nk].shape != sd[k].shape or out[nk].dtype != sd[k].dtype:
                fail(f"shape/dtype changed for {k} -> {nk}")

os.makedirs(os.path.dirname(DST), exist_ok=True)
save_file(out, DST, metadata={"format": "pt"})
print(f"wrote {DST}: {len(out)} tensors, blocks {blocks}")
