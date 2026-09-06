"""T3: mixed-precision sharded export of GPT-2 (124M) using torch + safetensors."""
import json
import os
import re

import torch
from safetensors.torch import load_file, save_file

IN = "inputs/base/model.safetensors"
OUT = "out/T3"
SHARD_BUDGET = 64 * 1024 * 1024  # bytes of tensor data per shard

PROJ = re.compile(r"^h\.\d+\.(attn\.c_attn|attn\.c_proj|mlp\.c_fc|mlp\.c_proj)\.weight$")
BUFFER = re.compile(r"^h\.\d+\.attn\.bias$")

sd = load_file(IN)
assert len(sd) == 160, len(sd)

out = {}
for name, t in sd.items():
    if BUFFER.match(name):
        continue  # causal-mask buffer, not a parameter
    if PROJ.match(name):
        out[name] = t.to(torch.bfloat16).contiguous()
    else:
        assert t.dtype == torch.float32, (name, t.dtype)
        out[name] = t.contiguous()

# Required checks (fail loudly before writing).
n_bf16 = sum(v.dtype == torch.bfloat16 for v in out.values())
assert n_bf16 == 48, f"expected 48 bfloat16 tensors, got {n_bf16}"
assert out["h.0.attn.c_attn.weight"].dtype == torch.bfloat16
assert out["wte.weight"].dtype == torch.float32
assert len(out) == 148, f"expected 148 tensors, got {len(out)}"
assert not any(BUFFER.match(k) for k in out)

# Shard: greedy in original key order; a tensor over budget gets its own shard.
shards, cur, cur_bytes = [], {}, 0
for name, t in out.items():
    nbytes = t.numel() * t.element_size()
    if cur and cur_bytes + nbytes > SHARD_BUDGET:
        shards.append(cur)
        cur, cur_bytes = {}, 0
    cur[name] = t
    cur_bytes += nbytes
    if nbytes > SHARD_BUDGET:  # oversized tensor stored alone
        shards.append(cur)
        cur, cur_bytes = {}, 0
if cur:
    shards.append(cur)

os.makedirs(OUT, exist_ok=True)
n = len(shards)
weight_map = {}
total = 0
for i, shard in enumerate(shards, 1):
    fname = f"model-{i:05d}-of-{n:05d}.safetensors"
    size = sum(t.numel() * t.element_size() for t in shard.values())
    assert size <= SHARD_BUDGET or len(shard) == 1, (fname, size)
    total += size
    save_file(shard, os.path.join(OUT, fname), metadata={"format": "pt"})
    for k in shard:
        weight_map[k] = fname

with open(os.path.join(OUT, "model.safetensors.index.json"), "w") as f:
    json.dump({"metadata": {"total_size": total}, "weight_map": weight_map}, f, indent=2)

assert len(weight_map) == 148
print(f"wrote {n} shards, {len(weight_map)} tensors, {total} bytes")
