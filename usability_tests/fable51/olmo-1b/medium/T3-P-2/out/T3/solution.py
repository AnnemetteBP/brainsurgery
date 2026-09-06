"""T3: mixed-precision export with sharding for OLMo-1B-0724-hf."""
import json
import os
import re
import sys

import torch
from safetensors.torch import load_file, save_file

IN_DIR = "inputs/base"
OUT_DIR = "out/T3"
MAX_SHARD = 256 * 1024 * 1024  # 268,435,456 bytes of tensor data per shard

PROJ_RE = re.compile(
    r"^model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)\.weight$"
)


def fail(msg):
    print(f"CHECK FAILED: {msg}", file=sys.stderr)
    sys.exit(1)


# ---- load all shards ----
with open(os.path.join(IN_DIR, "model.safetensors.index.json")) as f:
    index = json.load(f)
shard_files = sorted(set(index["weight_map"].values()))
state = {}
for sf in shard_files:
    state.update(load_file(os.path.join(IN_DIR, sf)))
print(f"loaded {len(state)} tensors from {len(shard_files)} shards")
if len(state) != 114:
    fail(f"expected 114 input tensors, got {len(state)}")

# ---- transform ----
out = {}
n_bf16 = 0
for name, t in state.items():
    if PROJ_RE.match(name):
        out[name] = t.to(torch.bfloat16).contiguous()
        n_bf16 += 1
    else:
        if t.dtype != torch.float32:
            fail(f"{name} is {t.dtype}, expected float32 input")
        out[name] = t.contiguous()

# ---- required checks (before writing) ----
if n_bf16 != 112:
    fail(f"expected 112 bfloat16 tensors, got {n_bf16}")
if sum(1 for t in out.values() if t.dtype == torch.bfloat16) != 112:
    fail("bfloat16 count mismatch")
if out["model.layers.0.self_attn.q_proj.weight"].dtype != torch.bfloat16:
    fail("model.layers.0.self_attn.q_proj.weight is not bfloat16")
if out["model.embed_tokens.weight"].dtype != torch.float32:
    fail("model.embed_tokens.weight is not float32")
if len(out) != 114:
    fail(f"expected 114 output tensors, got {len(out)}")
if set(out) != set(state):
    fail("tensor name set changed")
for name, t in out.items():
    if t.dtype == torch.float32 and not torch.equal(t, state[name]):
        fail(f"{name} float32 values changed")

# ---- shard (greedy, index order; oversized tensors alone) ----
shards = []  # list of (names, nbytes)
cur, cur_bytes = [], 0
for name in out:
    nb = out[name].numel() * out[name].element_size()
    if nb > MAX_SHARD:
        if cur:
            shards.append(cur)
            cur, cur_bytes = [], 0
        shards.append([name])
        continue
    if cur_bytes + nb > MAX_SHARD:
        shards.append(cur)
        cur, cur_bytes = [], 0
    cur.append(name)
    cur_bytes += nb
if cur:
    shards.append(cur)

n = len(shards)
weight_map = {}
total_size = 0
os.makedirs(OUT_DIR, exist_ok=True)
for i, names in enumerate(shards, 1):
    fname = f"model-{i:05d}-of-{n:05d}.safetensors"
    tensors = {k: out[k] for k in names}
    nb = sum(t.numel() * t.element_size() for t in tensors.values())
    if nb > MAX_SHARD and len(tensors) != 1:
        fail(f"shard {fname} exceeds budget with {len(tensors)} tensors")
    total_size += nb
    save_file(tensors, os.path.join(OUT_DIR, fname), metadata={"format": "pt"})
    for k in names:
        weight_map[k] = fname
    print(f"{fname}: {len(names)} tensors, {nb} bytes")

with open(os.path.join(OUT_DIR, "model.safetensors.index.json"), "w") as f:
    json.dump({"metadata": {"total_size": total_size}, "weight_map": weight_map}, f, indent=2)

# ---- verify written output ----
reloaded = {}
for sf in sorted(set(weight_map.values())):
    reloaded.update(load_file(os.path.join(OUT_DIR, sf)))
if len(reloaded) != 114 or len(weight_map) != 114:
    fail(f"written output has {len(reloaded)} tensors / {len(weight_map)} index entries")
if sum(1 for t in reloaded.values() if t.dtype == torch.bfloat16) != 112:
    fail("reloaded bfloat16 count != 112")
print(f"done: {n} shards, {len(reloaded)} tensors, total_size={total_size}")
