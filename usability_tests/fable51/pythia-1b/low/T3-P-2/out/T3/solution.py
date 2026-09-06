"""T3: mixed-precision export of Pythia-1B with sharding."""
import json
import os
import re

import torch
from safetensors.torch import load_file, save_file

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
SRC = os.path.join(ROOT, "inputs", "base", "model.safetensors")
OUT = HERE
MAX_SHARD = 256 * 1024 * 1024

PROJ_RE = re.compile(
    r"^gpt_neox\.layers\.\d+\."
    r"(attention\.query_key_value|attention\.dense|mlp\.dense_h_to_4h|mlp\.dense_4h_to_h)\.weight$"
)
BUF_RE = re.compile(
    r"^gpt_neox\.layers\.\d+\.attention\.(bias|masked_bias|rotary_emb\.inv_freq)$"
)

sd = load_file(SRC)
assert len(sd) == 244, len(sd)

out = {}
n_proj = n_buf = 0
for name, t in sd.items():
    if BUF_RE.match(name):
        n_buf += 1
        continue
    if PROJ_RE.match(name):
        n_proj += 1
        out[name] = t.to(torch.bfloat16).contiguous()
    else:
        out[name] = t.to(torch.float32).contiguous()

# Required checks
n_bf16 = sum(1 for t in out.values() if t.dtype == torch.bfloat16)
assert n_bf16 == 64, f"expected 64 bfloat16 tensors, got {n_bf16}"
assert n_proj == 64 and n_buf == 48, (n_proj, n_buf)
assert out["gpt_neox.layers.0.attention.query_key_value.weight"].dtype == torch.bfloat16
assert out["gpt_neox.embed_in.weight"].dtype == torch.float32
assert all(t.dtype == torch.float32 for k, t in out.items() if not PROJ_RE.match(k))
assert len(out) == 196, f"expected 196 tensors, got {len(out)}"

# Shard: greedy in original key order; an oversized tensor goes alone.
shards, cur, cur_size = [], {}, 0
for name, t in out.items():
    size = t.numel() * t.element_size()
    if cur and cur_size + size > MAX_SHARD:
        shards.append(cur)
        cur, cur_size = {}, 0
    cur[name] = t
    cur_size += size
    if size > MAX_SHARD:
        shards.append(cur)
        cur, cur_size = {}, 0
if cur:
    shards.append(cur)

n = len(shards)
weight_map = {}
total = 0
for i, shard in enumerate(shards, 1):
    fname = f"model-{i:05d}-of-{n:05d}.safetensors"
    shard_bytes = sum(t.numel() * t.element_size() for t in shard.values())
    assert shard_bytes <= MAX_SHARD or len(shard) == 1, (fname, shard_bytes)
    total += shard_bytes
    path = os.path.join(OUT, fname)
    assert not os.path.exists(path), f"destination exists: {path}"
    save_file(shard, path, metadata={"format": "pt"})
    for k in shard:
        weight_map[k] = fname

assert len(weight_map) == 196
with open(os.path.join(OUT, "model.safetensors.index.json"), "w") as f:
    json.dump({"metadata": {"total_size": total}, "weight_map": weight_map}, f, indent=2)
print(f"wrote {n} shards, {len(weight_map)} tensors, {total} bytes")
