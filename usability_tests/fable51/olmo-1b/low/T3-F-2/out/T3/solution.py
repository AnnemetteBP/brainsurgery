"""T3: mixed-precision export of OLMo-1B with sharding (plain torch + safetensors)."""
import json
import os
import re
import sys

import torch
from safetensors.torch import load_file, save_file

SRC = "inputs/base"
DST = "out/T3"
MAX_SHARD = 256 * 1024 * 1024  # bytes of tensor data
PROJ = re.compile(
    r"^model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)\.weight$"
)

# ---- load ----
index = json.load(open(os.path.join(SRC, "model.safetensors.index.json")))
sd = {}
for shard in sorted(set(index["weight_map"].values())):
    sd.update(load_file(os.path.join(SRC, shard)))
assert len(sd) == 114, f"expected 114 input tensors, got {len(sd)}"

# ---- transform ----
out = {}
for name, t in sd.items():
    if PROJ.match(name):
        out[name] = t.to(torch.bfloat16).contiguous()
    else:
        assert t.dtype == torch.float32, f"{name} unexpectedly {t.dtype}"
        out[name] = t.contiguous()

# ---- required checks (fail before writing) ----
n_bf16 = sum(t.dtype == torch.bfloat16 for t in out.values())
assert n_bf16 == 112, f"expected 112 bfloat16 tensors, got {n_bf16}"
assert out["model.layers.0.self_attn.q_proj.weight"].dtype == torch.bfloat16
assert out["model.embed_tokens.weight"].dtype == torch.float32
assert out["lm_head.weight"].dtype == torch.float32
assert len(out) == 114, f"expected 114 output tensors, got {len(out)}"
assert set(out) == set(sd), "tensor names changed"

# ---- shard (greedy, in original order; oversized tensors get their own shard) ----
shards, cur, cur_size = [], {}, 0
for name in sd:  # keep original key order
    t = out[name]
    nbytes = t.numel() * t.element_size()
    if cur and cur_size + nbytes > MAX_SHARD:
        shards.append(cur)
        cur, cur_size = {}, 0
    cur[name] = t
    cur_size += nbytes
if cur:
    shards.append(cur)

# ---- write ----
os.makedirs(DST, exist_ok=True)
n = len(shards)
weight_map, total = {}, 0
for i, shard in enumerate(shards, 1):
    fname = f"model-{i:05d}-of-{n:05d}.safetensors"
    size = sum(t.numel() * t.element_size() for t in shard.values())
    assert size <= MAX_SHARD or len(shard) == 1, f"{fname} over budget: {size}"
    save_file(shard, os.path.join(DST, fname), metadata={"format": "pt"})
    for name in shard:
        weight_map[name] = fname
    total += size
with open(os.path.join(DST, "model.safetensors.index.json"), "w") as f:
    json.dump({"metadata": {"total_size": total}, "weight_map": weight_map}, f, indent=2)
assert len(weight_map) == 114
print(f"wrote {n} shards, {len(weight_map)} tensors, {total} bytes", file=sys.stderr)
