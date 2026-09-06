"""T2: prune attention head 5 from every layer of Pythia-1B (GPT-NeoX layout)."""
import os
import sys

import torch
from safetensors.torch import load_file, save_file

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "..", "inputs", "base", "model.safetensors")
DST = os.path.join(HERE, "model.safetensors")

N_LAYERS, N_HEADS, HEAD_DIM, HIDDEN = 16, 8, 256, 2048
PRUNE = 5
BLOCK = 3 * HEAD_DIM  # q,k,v rows per head in the fused projection

state = load_file(SRC)
if len(state) != 244:
    sys.exit(f"expected 244 input tensors, got {len(state)}")

qkv_keep = torch.cat([torch.arange(0, BLOCK * PRUNE), torch.arange(BLOCK * (PRUNE + 1), BLOCK * N_HEADS)])
dense_keep = torch.cat([torch.arange(0, HEAD_DIM * PRUNE), torch.arange(HEAD_DIM * (PRUNE + 1), HEAD_DIM * N_HEADS)])

for i in range(N_LAYERS):
    p = f"gpt_neox.layers.{i}.attention."
    w, b, d = p + "query_key_value.weight", p + "query_key_value.bias", p + "dense.weight"
    assert state[w].shape == (BLOCK * N_HEADS, HIDDEN), (w, state[w].shape)
    assert state[b].shape == (BLOCK * N_HEADS,), (b, state[b].shape)
    assert state[d].shape == (HIDDEN, HEAD_DIM * N_HEADS), (d, state[d].shape)
    state[w] = state[w].index_select(0, qkv_keep).contiguous()
    state[b] = state[b].index_select(0, qkv_keep).contiguous()
    state[d] = state[d].index_select(1, dense_keep).contiguous()

# Required checks
l0 = "gpt_neox.layers.0.attention."
assert state[l0 + "query_key_value.weight"].shape == (5376, 2048), state[l0 + "query_key_value.weight"].shape
assert state[l0 + "query_key_value.bias"].shape == (5376,), state[l0 + "query_key_value.bias"].shape
assert state[l0 + "dense.weight"].shape == (2048, 1792), state[l0 + "dense.weight"].shape
assert len(state) == 244, len(state)

save_file(state, DST, metadata={"format": "pt"})
print(f"wrote {DST} with {len(state)} tensors")
