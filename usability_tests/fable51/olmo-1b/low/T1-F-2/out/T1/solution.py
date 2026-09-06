"""T1: drop blocks 2, 6, 10, 14 from OLMo-1B (16 layers), renumber to 0..11, save one file."""
import json, os, re, sys
from safetensors import safe_open
from safetensors.torch import save_file

BASE = "inputs/base"
OUT = "out/T1/model.safetensors"
DROP = {2, 6, 10, 14}
PAT = re.compile(r"^model\.layers\.(\d+)\.(.+)$")

def fail(msg):
    print("FAIL:", msg, file=sys.stderr)
    sys.exit(1)

# Load all tensors from the shards (index tells us shard membership).
index = json.load(open(os.path.join(BASE, "model.safetensors.index.json")))
shards = sorted(set(index["weight_map"].values()))
src = {}
for shard in shards:
    with safe_open(os.path.join(BASE, shard), framework="pt") as f:
        for k in f.keys():
            src[k] = f.get_tensor(k)
if len(src) != 114:
    fail(f"expected 114 input tensors, got {len(src)}")

# Build the old->new block mapping from the surviving blocks in original order.
old_blocks = sorted({int(m.group(1)) for k in src if (m := PAT.match(k))})
keep = [b for b in old_blocks if b not in DROP]
remap = {old: new for new, old in enumerate(keep)}

# Build the output dict fresh (no in-place moves, so no collision hazard).
dst = {}
for k, t in src.items():
    m = PAT.match(k)
    if m is None:
        nk = k
    else:
        old = int(m.group(1))
        if old in DROP:
            continue
        nk = f"model.layers.{remap[old]}.{m.group(2)}"
    if nk in dst:
        fail(f"collision on {nk}")
    dst[nk] = t.contiguous()

# Required checks.
blocks = {int(m.group(1)) for k in dst if (m := PAT.match(k))}
if blocks & {12, 13, 14, 15}:
    fail(f"blocks >= 12 remain: {sorted(blocks & {12,13,14,15})}")
if blocks != set(range(12)):
    fail(f"expected blocks 0..11, got {sorted(blocks)}")
q = [k for k in dst if k.endswith(".self_attn.q_proj.weight")]
if len(q) != 12:
    fail(f"expected 12 q_proj tensors, got {len(q)}")
if len(dst) != 86:
    fail(f"expected 86 output tensors, got {len(dst)}")
# Values/shapes/dtypes unchanged versus the source.
for old, new in remap.items():
    for k in src:
        m = PAT.match(k)
        if m and int(m.group(1)) == old:
            nk = f"model.layers.{new}.{m.group(2)}"
            s, d = src[k], dst[nk]
            if s.shape != d.shape or s.dtype != d.dtype or not (s == d).all():
                fail(f"mismatch {k} -> {nk}")

os.makedirs(os.path.dirname(OUT), exist_ok=True)
save_file(dst, OUT, metadata={"format": "pt"})
print(f"wrote {OUT} with {len(dst)} tensors; block map {remap}")
