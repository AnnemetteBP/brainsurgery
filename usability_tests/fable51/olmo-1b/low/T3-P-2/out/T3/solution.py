"""T3: mixed-precision sharded export of OLMo-1B-0724-hf."""
import json
import os
import re
import sys

import torch
from safetensors.torch import load_file, save_file

BASE = "inputs/base"
OUT = "out/T3"
SHARD_BUDGET = 256 * 1024 * 1024  # 268,435,456 bytes of tensor data

PROJ_RE = re.compile(
    r"^model\.layers\.\d+\.(self_attn\.[qkvo]_proj|mlp\.(gate|up|down)_proj)\.weight$"
)


def fail(msg):
    print("CHECK FAILED: " + msg, file=sys.stderr)
    sys.exit(1)


# ---- load all shards ----
with open(os.path.join(BASE, "model.safetensors.index.json")) as f:
    in_index = json.load(f)
state = {}
for shard in sorted(set(in_index["weight_map"].values())):
    state.update(load_file(os.path.join(BASE, shard)))
if len(state) != 114:
    fail(f"expected 114 input tensors, got {len(state)}")

# ---- cast exactly the projection matrices ----
out = {}
n_cast = 0
for name, t in state.items():
    if PROJ_RE.match(name):
        out[name] = t.to(torch.bfloat16).contiguous()
        n_cast += 1
    else:
        if t.dtype != torch.float32:
            fail(f"{name} expected float32 input, got {t.dtype}")
        out[name] = t.contiguous()

# ---- required checks ----
n_bf16 = sum(1 for t in out.values() if t.dtype == torch.bfloat16)
if n_bf16 != 112:
    fail(f"expected 112 bfloat16 tensors, got {n_bf16}")
if n_cast != 112:
    fail(f"pattern matched {n_cast} tensors, expected 112")
if out["model.layers.0.self_attn.q_proj.weight"].dtype != torch.bfloat16:
    fail("q_proj layer 0 is not bfloat16")
if out["model.embed_tokens.weight"].dtype != torch.float32:
    fail("embed_tokens is not float32")
if out["lm_head.weight"].dtype != torch.float32:
    fail("lm_head is not float32")
if len(out) != 114:
    fail(f"expected 114 output tensors, got {len(out)}")
if set(out) != set(state):
    fail("tensor name set changed")
for name in out:
    if out[name].shape != state[name].shape:
        fail(f"shape changed for {name}")

# ---- shard: greedy in original order; oversized tensors go alone ----
shards = []  # list of (names, bytes)
cur, cur_bytes = [], 0
for name in state:  # original key order
    nb = out[name].numel() * out[name].element_size()
    if nb > SHARD_BUDGET:
        if cur:
            shards.append(cur)
            cur, cur_bytes = [], 0
        shards.append([name])
        continue
    if cur_bytes + nb > SHARD_BUDGET:
        shards.append(cur)
        cur, cur_bytes = [], 0
    cur.append(name)
    cur_bytes += nb
if cur:
    shards.append(cur)

n = len(shards)
os.makedirs(OUT, exist_ok=True)
weight_map = {}
total_size = 0
for i, names in enumerate(shards, 1):
    fname = f"model-{i:05d}-of-{n:05d}.safetensors"
    part = {k: out[k] for k in names}
    size = sum(t.numel() * t.element_size() for t in part.values())
    if size > SHARD_BUDGET and len(names) > 1:
        fail(f"shard {fname} exceeds budget: {size}")
    total_size += size
    save_file(part, os.path.join(OUT, fname), metadata={"format": "pt"})
    for k in names:
        weight_map[k] = fname

if len(weight_map) != 114:
    fail(f"weight_map has {len(weight_map)} entries")
with open(os.path.join(OUT, "model.safetensors.index.json"), "w") as f:
    json.dump({"metadata": {"total_size": total_size}, "weight_map": weight_map}, f, indent=2)

print(f"wrote {n} shards, {len(weight_map)} tensors, {n_bf16} bf16, total {total_size} bytes")
