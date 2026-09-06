"""T3: mixed-precision export of GPT-2 (124M) with sharding."""
import json
import os
import re

import torch
from safetensors.torch import load_file, save_file

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "..", "inputs", "base", "model.safetensors")
OUT = HERE
MAX_SHARD = 64 * 1024 * 1024

PROJ_RE = re.compile(
    r"^h\.(\d+)\.(attn\.c_attn|attn\.c_proj|mlp\.c_fc|mlp\.c_proj)\.weight$"
)
BUFFER_RE = re.compile(r"^h\.(\d+)\.attn\.bias$")

sd = load_file(SRC)
assert len(sd) == 160, len(sd)

out = {}
n_bf16 = 0
n_dropped = 0
for name, t in sd.items():
    if BUFFER_RE.match(name):
        n_dropped += 1
        continue
    if PROJ_RE.match(name):
        assert t.dtype == torch.float32 and t.ndim == 2, name
        out[name] = t.to(torch.bfloat16).contiguous()
        n_bf16 += 1
    else:
        assert t.dtype == torch.float32, (name, t.dtype)
        out[name] = t.contiguous()

# Required checks
assert n_bf16 == 48, n_bf16
assert sum(v.dtype == torch.bfloat16 for v in out.values()) == 48
assert out["h.0.attn.c_attn.weight"].dtype == torch.bfloat16
assert out["wte.weight"].dtype == torch.float32
assert n_dropped == 12, n_dropped
assert len(out) == 148, len(out)

# Sharding: greedy in original key order, each shard <= 64 MiB of tensor data;
# an oversized tensor goes alone in its own shard.
shards = []
cur, cur_size = [], 0
for name in out:
    size = out[name].numel() * out[name].element_size()
    if cur and cur_size + size > MAX_SHARD:
        shards.append(cur)
        cur, cur_size = [], 0
    cur.append(name)
    cur_size += size
if cur:
    shards.append(cur)

n = len(shards)
weight_map = {}
total_size = 0
for i, names in enumerate(shards, 1):
    fname = f"model-{i:05d}-of-{n:05d}.safetensors"
    tensors = {k: out[k] for k in names}
    data = sum(v.numel() * v.element_size() for v in tensors.values())
    assert data <= MAX_SHARD or len(tensors) == 1, (fname, data)
    total_size += data
    save_file(tensors, os.path.join(OUT, fname), metadata={"format": "pt"})
    for k in names:
        weight_map[k] = fname

assert len(weight_map) == 148
with open(os.path.join(OUT, "model.safetensors.index.json"), "w") as f:
    json.dump({"metadata": {"total_size": total_size}, "weight_map": weight_map}, f, indent=2)

print(f"wrote {n} shards, {len(weight_map)} tensors, {total_size} bytes")
