"""T2: remove attention head 5 from every layer of OLMo-1B-0724-hf."""
import json
import os
import re
import sys

import torch
from safetensors.torch import load_file, save_file

BASE = "inputs/base"
OUT = "out/T2/model.safetensors"
N_LAYERS, N_HEADS, HEAD_DIM, HIDDEN = 16, 16, 128, 2048
PRUNE_HEAD = 5

# 1. Load all shards into one state dict.
with open(os.path.join(BASE, "model.safetensors.index.json")) as f:
    index = json.load(f)
shards = sorted(set(index["weight_map"].values()))
sd = {}
for shard in shards:
    part = load_file(os.path.join(BASE, shard))
    assert not (set(part) & set(sd)), f"duplicate keys across shards in {shard}"
    sd.update(part)
assert len(sd) == 114, f"expected 114 input tensors, got {len(sd)}"

# 2. Prune the head slice from each head-bearing projection.
keep = torch.cat([
    torch.arange(0, PRUNE_HEAD * HEAD_DIM),
    torch.arange((PRUNE_HEAD + 1) * HEAD_DIM, N_HEADS * HEAD_DIM),
])
assert keep.tolist() == list(range(0, 640)) + list(range(768, 2048))
pat = re.compile(r"^model\.layers\.(\d+)\.self_attn\.([qkvo]_proj)\.weight$")
touched = 0
for name in list(sd):
    m = pat.match(name)
    if not m:
        continue
    layer, proj = int(m.group(1)), m.group(2)
    assert 0 <= layer < N_LAYERS, name
    w = sd[name]
    assert w.shape == (HIDDEN, HIDDEN), (name, w.shape)
    if proj == "o_proj":
        sd[name] = w.index_select(1, keep).contiguous()   # heads are column blocks
    else:
        sd[name] = w.index_select(0, keep).contiguous()   # heads are row blocks
    touched += 1
assert touched == N_LAYERS * 4, f"touched {touched} tensors, expected {N_LAYERS * 4}"

# 3. Required checks (fail loudly before writing).
def check(name, shape):
    got = tuple(sd[name].shape)
    if got != shape:
        sys.exit(f"CHECK FAILED: {name} has shape {got}, expected {shape}")

check("model.layers.0.self_attn.q_proj.weight", (1920, 2048))
check("model.layers.0.self_attn.k_proj.weight", (1920, 2048))
check("model.layers.0.self_attn.v_proj.weight", (1920, 2048))
check("model.layers.0.self_attn.o_proj.weight", (2048, 1920))
if len(sd) != 114:
    sys.exit(f"CHECK FAILED: output has {len(sd)} tensors, expected 114")
# Every layer, and dtype preserved.
for i in range(N_LAYERS):
    for p in ("q_proj", "k_proj", "v_proj"):
        check(f"model.layers.{i}.self_attn.{p}.weight", (1920, 2048))
    check(f"model.layers.{i}.self_attn.o_proj.weight", (2048, 1920))
assert all(t.dtype == torch.float32 for t in sd.values()), "dtype changed"

# 4. Write a single file.
os.makedirs(os.path.dirname(OUT), exist_ok=True)
save_file(sd, OUT, metadata={"format": "pt"})

# 5. Verify what was written.
written = load_file(OUT)
assert len(written) == 114
assert set(written) == set(sd)
w0 = written["model.layers.0.self_attn.q_proj.weight"]
assert w0.shape == (1920, 2048)
print(f"OK: wrote {OUT} with {len(written)} tensors; pruned head {PRUNE_HEAD} in {N_LAYERS} layers")
