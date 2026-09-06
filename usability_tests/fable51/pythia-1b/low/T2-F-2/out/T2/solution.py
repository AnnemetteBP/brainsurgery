"""T2: remove attention head 5 from every layer of Pythia-1B (safetensors -> safetensors)."""
import os
import torch
from safetensors.torch import load_file, save_file

SRC = "inputs/base/model.safetensors"
DST = "out/T2/model.safetensors"
N_LAYERS, N_HEADS, HEAD_DIM, HEAD = 16, 8, 256, 5
QKV_BLOCK = 3 * HEAD_DIM  # 768 rows per head (q|k|v interleaved per head)

sd = load_file(SRC)
assert len(sd) == 244, len(sd)

def drop_rows(t, start, size):
    return torch.cat([t[:start], t[start + size:]], dim=0).contiguous()

def drop_cols(t, start, size):
    return torch.cat([t[:, :start], t[:, start + size:]], dim=1).contiguous()

for i in range(N_LAYERS):
    p = f"gpt_neox.layers.{i}.attention."
    w, b, d = p + "query_key_value.weight", p + "query_key_value.bias", p + "dense.weight"
    assert sd[w].shape == (N_HEADS * QKV_BLOCK, 2048), sd[w].shape
    assert sd[b].shape == (N_HEADS * QKV_BLOCK,), sd[b].shape
    assert sd[d].shape == (2048, N_HEADS * HEAD_DIM), sd[d].shape
    sd[w] = drop_rows(sd[w], HEAD * QKV_BLOCK, QKV_BLOCK)
    sd[b] = drop_rows(sd[b], HEAD * QKV_BLOCK, QKV_BLOCK)
    sd[d] = drop_cols(sd[d], HEAD * HEAD_DIM, HEAD_DIM)

# Required checks (fail loudly before writing).
assert sd["gpt_neox.layers.0.attention.query_key_value.weight"].shape == (5376, 2048)
assert sd["gpt_neox.layers.0.attention.query_key_value.bias"].shape == (5376,)
assert sd["gpt_neox.layers.0.attention.dense.weight"].shape == (2048, 1792)
assert len(sd) == 244, len(sd)
for i in range(N_LAYERS):
    p = f"gpt_neox.layers.{i}.attention."
    assert sd[p + "query_key_value.weight"].shape == (5376, 2048)
    assert sd[p + "query_key_value.bias"].shape == (5376,)
    assert sd[p + "dense.weight"].shape == (2048, 1792)

os.makedirs(os.path.dirname(DST), exist_ok=True)
save_file(sd, DST, metadata={"format": "pt"})
print("wrote", DST, len(sd), "tensors")
