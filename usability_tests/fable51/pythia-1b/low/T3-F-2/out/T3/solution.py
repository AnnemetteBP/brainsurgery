"""T3: mixed-precision sharded export of Pythia-1B (plain torch + safetensors)."""
import json
import os
import re

import torch
from safetensors import safe_open
from safetensors.torch import save_file

SRC = "inputs/base/model.safetensors"
OUT = "out/T3"
MAX_SHARD = 256 * 1024 * 1024

PROJ_RE = re.compile(
    r"^gpt_neox\.layers\.\d+\.(attention\.(query_key_value|dense)"
    r"|mlp\.(dense_h_to_4h|dense_4h_to_h))\.weight$"
)
BUF_RE = re.compile(
    r"^gpt_neox\.layers\.\d+\.attention\.(bias|masked_bias|rotary_emb\.inv_freq)$"
)

out = {}
with safe_open(SRC, framework="pt") as f:
    keys = list(f.keys())
    assert len(keys) == 244, len(keys)
    for k in keys:
        if BUF_RE.match(k):
            continue
        t = f.get_tensor(k)
        out[k] = t.to(torch.bfloat16 if PROJ_RE.match(k) else torch.float32).contiguous()

# Required checks (fail before writing)
n_bf16 = sum(t.dtype == torch.bfloat16 for t in out.values())
assert n_bf16 == 64, f"expected 64 bfloat16 tensors, got {n_bf16}"
assert out["gpt_neox.layers.0.attention.query_key_value.weight"].dtype == torch.bfloat16
assert out["gpt_neox.embed_in.weight"].dtype == torch.float32
assert len(out) == 196, f"expected 196 tensors, got {len(out)}"
assert all(t.dtype in (torch.bfloat16, torch.float32) for t in out.values())
assert not any(BUF_RE.match(k) for k in out)

# Shard: greedy in original key order, oversized tensors alone.
shards, cur, cur_size = [], {}, 0
for k, t in out.items():
    nb = t.numel() * t.element_size()
    if cur and cur_size + nb > MAX_SHARD:
        shards.append(cur)
        cur, cur_size = {}, 0
    cur[k] = t
    cur_size += nb
    if nb > MAX_SHARD:  # oversized tensor stays alone
        shards.append(cur)
        cur, cur_size = {}, 0
if cur:
    shards.append(cur)

for s in shards:
    assert len(s) == 1 or sum(t.numel() * t.element_size() for t in s.values()) <= MAX_SHARD

os.makedirs(OUT, exist_ok=True)
for fn in os.listdir(OUT):
    if fn.endswith(".safetensors") or fn == "model.safetensors.index.json":
        os.remove(os.path.join(OUT, fn))

weight_map, total = {}, 0
n = len(shards)
for i, s in enumerate(shards, 1):
    fn = f"model-{i:05d}-of-{n:05d}.safetensors"
    save_file(s, os.path.join(OUT, fn), metadata={"format": "pt"})
    for k, t in s.items():
        weight_map[k] = fn
        total += t.numel() * t.element_size()
with open(os.path.join(OUT, "model.safetensors.index.json"), "w") as fh:
    json.dump({"metadata": {"total_size": total}, "weight_map": weight_map}, fh, indent=2)
assert len(weight_map) == 196
print(f"wrote {n} shards, {len(weight_map)} tensors, {total} bytes")
