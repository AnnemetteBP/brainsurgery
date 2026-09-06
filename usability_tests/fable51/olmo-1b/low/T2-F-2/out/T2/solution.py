"""T2: remove attention head 5 from every layer of OLMo-1B-0724-hf.

Plain torch + safetensors: slice head blocks out of q/k/v (rows) and o_proj
(columns), leave everything else untouched, write one safetensors file.
"""
import json
import os
import sys

import torch
from safetensors.torch import load_file, save_file

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, "..", "..", "inputs", "base")
OUT = os.path.join(HERE, "model.safetensors")

NUM_LAYERS, NUM_HEADS, HEAD_DIM, HIDDEN = 16, 16, 128, 2048
PRUNE_HEAD = 5
KEEP = [h for h in range(NUM_HEADS) if h != PRUNE_HEAD]
KEEP_IDX = torch.cat([torch.arange(h * HEAD_DIM, (h + 1) * HEAD_DIM) for h in KEEP])
NEW_DIM = len(KEEP) * HEAD_DIM  # 1920


def fail(msg):
    print("CHECK FAILED:", msg, file=sys.stderr)
    sys.exit(1)


# Load all shards into one state dict.
with open(os.path.join(BASE, "model.safetensors.index.json")) as f:
    index = json.load(f)
shards = sorted(set(index["weight_map"].values()))
state = {}
for shard in shards:
    state.update(load_file(os.path.join(BASE, shard)))
if len(state) != 114:
    fail(f"expected 114 input tensors, got {len(state)}")

for i in range(NUM_LAYERS):
    pre = f"model.layers.{i}.self_attn."
    for name in ("q_proj", "k_proj", "v_proj"):
        key = pre + name + ".weight"
        w = state[key]
        if tuple(w.shape) != (HIDDEN, HIDDEN):
            fail(f"{key} unexpected shape {tuple(w.shape)}")
        state[key] = w.index_select(0, KEEP_IDX).contiguous()
    key = pre + "o_proj.weight"
    w = state[key]
    if tuple(w.shape) != (HIDDEN, HIDDEN):
        fail(f"{key} unexpected shape {tuple(w.shape)}")
    state[key] = w.index_select(1, KEEP_IDX).contiguous()

# Required checks (before writing).
expected = {
    "model.layers.0.self_attn.q_proj.weight": (NEW_DIM, HIDDEN),
    "model.layers.0.self_attn.k_proj.weight": (NEW_DIM, HIDDEN),
    "model.layers.0.self_attn.v_proj.weight": (NEW_DIM, HIDDEN),
    "model.layers.0.self_attn.o_proj.weight": (HIDDEN, NEW_DIM),
}
for key, shape in expected.items():
    if tuple(state[key].shape) != shape:
        fail(f"{key} has shape {tuple(state[key].shape)}, expected {shape}")
if len(state) != 114:
    fail(f"output has {len(state)} tensors, expected 114")
# Extra: all layers, and dtype preserved.
for i in range(NUM_LAYERS):
    pre = f"model.layers.{i}.self_attn."
    for name in ("q_proj", "k_proj", "v_proj"):
        assert tuple(state[pre + name + ".weight"].shape) == (NEW_DIM, HIDDEN), i
    assert tuple(state[pre + "o_proj.weight"].shape) == (HIDDEN, NEW_DIM), i
assert all(t.dtype == torch.float32 for t in state.values())

save_file(state, OUT, metadata={"format": "pt"})
print(f"wrote {OUT} with {len(state)} tensors")
