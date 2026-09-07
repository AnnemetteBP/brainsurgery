"""T2: prune attention head 5 from every layer of OLMo-1B-0724-hf."""
import json
import os
import sys

import torch
from safetensors.torch import load_file, save_file

BASE = "inputs/base"
OUT = "out/T2/model.safetensors"
N_LAYERS = 16
HEAD_DIM = 128
PRUNE_HEAD = 5
KEEP = list(range(0, PRUNE_HEAD * HEAD_DIM)) + list(range((PRUNE_HEAD + 1) * HEAD_DIM, 2048))
KEEP_IDX = torch.tensor(KEEP, dtype=torch.long)


def fail(msg):
    print("FAIL:", msg, file=sys.stderr)
    sys.exit(1)


with open(os.path.join(BASE, "model.safetensors.index.json")) as f:
    index = json.load(f)
shards = sorted(set(index["weight_map"].values()))

state = {}
for shard in shards:
    state.update(load_file(os.path.join(BASE, shard)))
if len(state) != 114:
    fail(f"expected 114 input tensors, got {len(state)}")

for i in range(N_LAYERS):
    p = f"model.layers.{i}.self_attn."
    for name in ("q_proj", "k_proj", "v_proj"):
        key = p + name + ".weight"
        w = state[key]
        if tuple(w.shape) != (2048, 2048):
            fail(f"{key} unexpected shape {tuple(w.shape)}")
        state[key] = w.index_select(0, KEEP_IDX).contiguous()
    key = p + "o_proj.weight"
    w = state[key]
    if tuple(w.shape) != (2048, 2048):
        fail(f"{key} unexpected shape {tuple(w.shape)}")
    state[key] = w.index_select(1, KEEP_IDX).contiguous()

# Required checks
checks = {
    "model.layers.0.self_attn.q_proj.weight": (1920, 2048),
    "model.layers.0.self_attn.k_proj.weight": (1920, 2048),
    "model.layers.0.self_attn.v_proj.weight": (1920, 2048),
    "model.layers.0.self_attn.o_proj.weight": (2048, 1920),
}
for key, shape in checks.items():
    if tuple(state[key].shape) != shape:
        fail(f"{key} has shape {tuple(state[key].shape)}, expected {shape}")
if len(state) != 114:
    fail(f"output has {len(state)} tensors, expected 114")

save_file(state, OUT)
reloaded = load_file(OUT)
if len(reloaded) != 114:
    fail(f"written file has {len(reloaded)} tensors")
print(f"wrote {OUT} with {len(reloaded)} tensors")
