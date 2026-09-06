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
LO, HI = PRUNE_HEAD * HEAD_DIM, (PRUNE_HEAD + 1) * HEAD_DIM  # 640, 768


def fail(msg):
    print(f"FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


# Load all shards.
with open(os.path.join(BASE, "model.safetensors.index.json")) as f:
    index = json.load(f)
state = {}
for shard in sorted(set(index["weight_map"].values())):
    state.update(load_file(os.path.join(BASE, shard)))
if len(state) != 114:
    fail(f"expected 114 input tensors, got {len(state)}")

# Prune.
for i in range(N_LAYERS):
    p = f"model.layers.{i}.self_attn."
    for name in ("q_proj", "k_proj", "v_proj"):
        k = p + name + ".weight"
        w = state[k]
        if tuple(w.shape) != (2048, 2048):
            fail(f"{k} unexpected shape {tuple(w.shape)}")
        state[k] = torch.cat([w[:LO, :], w[HI:, :]], dim=0).contiguous()
    k = p + "o_proj.weight"
    w = state[k]
    if tuple(w.shape) != (2048, 2048):
        fail(f"{k} unexpected shape {tuple(w.shape)}")
    state[k] = torch.cat([w[:, :LO], w[:, HI:]], dim=1).contiguous()

# Required checks.
checks = {
    "model.layers.0.self_attn.q_proj.weight": (1920, 2048),
    "model.layers.0.self_attn.k_proj.weight": (1920, 2048),
    "model.layers.0.self_attn.v_proj.weight": (1920, 2048),
    "model.layers.0.self_attn.o_proj.weight": (2048, 1920),
}
for k, shape in checks.items():
    if tuple(state[k].shape) != shape:
        fail(f"{k} has shape {tuple(state[k].shape)}, expected {shape}")
if len(state) != 114:
    fail(f"output has {len(state)} tensors, expected 114")

save_file(state, OUT, metadata={"format": "pt"})
print(f"wrote {OUT} with {len(state)} tensors")
