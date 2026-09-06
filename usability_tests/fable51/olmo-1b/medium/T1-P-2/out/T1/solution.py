"""T1: depth-prune OLMo-1B-0724-hf from 16 to 12 layers, renumbering contiguously."""
import json
import os
import re
import sys

from safetensors.torch import load_file, save_file

BASE = "inputs/base"
OUT_DIR = "out/T1"
OUT = os.path.join(OUT_DIR, "model.safetensors")
REMOVE = {2, 6, 10, 14}
N_OLD, N_NEW, PER_BLOCK, N_OTHER = 16, 12, 7, 2
LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.(.+)$")


def fail(msg):
    print(f"FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


# Load all shards listed in the index.
with open(os.path.join(BASE, "model.safetensors.index.json")) as f:
    index = json.load(f)
shards = sorted(set(index["weight_map"].values()))
sd = {}
for shard in shards:
    part = load_file(os.path.join(BASE, shard))
    if set(part) & set(sd):
        fail(f"duplicate keys across shards in {shard}")
    sd.update(part)
if len(sd) != N_OLD * PER_BLOCK + N_OTHER:
    fail(f"expected {N_OLD * PER_BLOCK + N_OTHER} input tensors, got {len(sd)}")

# Build old->new block mapping from surviving blocks in original order.
survivors = [i for i in range(N_OLD) if i not in REMOVE]
remap = {old: new for new, old in enumerate(survivors)}

# Build the new state dict into a fresh dict, so no rename can collide.
out = {}
for name, t in sd.items():
    m = LAYER_RE.match(name)
    if m is None:
        new_name = name
    else:
        old = int(m.group(1))
        if old >= N_OLD:
            fail(f"unexpected block index in {name}")
        if old in REMOVE:
            continue
        new_name = f"model.layers.{remap[old]}.{m.group(2)}"
    if new_name in out:
        fail(f"collision on {new_name}")
    out[new_name] = t

# Required checks.
for name in out:
    m = LAYER_RE.match(name)
    if m and int(m.group(1)) >= N_NEW:
        fail(f"block index >= {N_NEW} remains: {name}")
q_keys = [n for n in out if re.fullmatch(r"model\.layers\.\d+\.self_attn\.q_proj\.weight", n)]
if len(q_keys) != N_NEW:
    fail(f"expected {N_NEW} q_proj tensors, got {len(q_keys)}")
blocks = sorted({int(LAYER_RE.match(n).group(1)) for n in out if LAYER_RE.match(n)})
if blocks != list(range(N_NEW)):
    fail(f"block indices not contiguous 0..{N_NEW - 1}: {blocks}")
for b in range(N_NEW):
    n_b = sum(1 for n in out if n.startswith(f"model.layers.{b}."))
    if n_b != PER_BLOCK:
        fail(f"block {b} has {n_b} tensors, expected {PER_BLOCK}")
if len(out) != N_NEW * PER_BLOCK + N_OTHER:
    fail(f"expected {N_NEW * PER_BLOCK + N_OTHER} output tensors, got {len(out)}")
# Values/shapes/dtypes unchanged (same tensor objects, verify identity of source).
for old, new in remap.items():
    for rest in ("self_attn.q_proj.weight", "mlp.down_proj.weight"):
        a, b = sd[f"model.layers.{old}.{rest}"], out[f"model.layers.{new}.{rest}"]
        if a.shape != b.shape or a.dtype != b.dtype or a.data_ptr() != b.data_ptr():
            fail(f"tensor mismatch old {old} -> new {new} ({rest})")

os.makedirs(OUT_DIR, exist_ok=True)
save_file({k: v.contiguous() for k, v in out.items()}, OUT, metadata={"format": "pt"})

# Post-write verification.
written = load_file(OUT)
if len(written) != N_NEW * PER_BLOCK + N_OTHER or set(written) != set(out):
    os.remove(OUT)
    fail("written file does not match expected key set")
print(f"OK: wrote {OUT} with {len(written)} tensors; blocks {blocks}")
