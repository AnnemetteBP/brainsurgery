"""T2: prune attention head 5 from every layer of Pythia-1B (checkpoint level)."""

import torch
from safetensors import safe_open
from safetensors.torch import save_file

SRC = "inputs/base/model.safetensors"
DST = "out/T2/model.safetensors"

NUM_LAYERS = 16
NUM_HEADS = 8
HEAD_DIM = 256
PRUNE = 5
QKV_BLOCK = 3 * HEAD_DIM  # 768 fused rows per head (q|k|v interleaved per head)

# rows/cols to keep
qkv_keep = torch.tensor(
    [r for r in range(NUM_HEADS * QKV_BLOCK)
     if not (PRUNE * QKV_BLOCK <= r < (PRUNE + 1) * QKV_BLOCK)]
)
dense_keep = torch.tensor(
    [c for c in range(NUM_HEADS * HEAD_DIM)
     if not (PRUNE * HEAD_DIM <= c < (PRUNE + 1) * HEAD_DIM)]
)

out = {}
src_dtypes = {}
with safe_open(SRC, framework="pt") as f:
    keys = list(f.keys())
    for k in keys:
        t = f.get_tensor(k)
        if k.endswith("attention.query_key_value.weight"):
            assert t.shape == (6144, 2048), (k, t.shape)
            t = t[qkv_keep, :]
        elif k.endswith("attention.query_key_value.bias"):
            assert t.shape == (6144,), (k, t.shape)
            t = t[qkv_keep]
        elif k.endswith("attention.dense.weight"):
            assert t.shape == (2048, 2048), (k, t.shape)
            t = t[:, dense_keep]
        out[k] = t.contiguous()
        src_dtypes[k] = t.dtype

# --- required checks, before writing ---
def check(name, shape):
    got = tuple(out[name].shape)
    if got != shape:
        raise AssertionError(f"{name}: expected {shape}, got {got}")

check("gpt_neox.layers.0.attention.query_key_value.weight", (5376, 2048))
check("gpt_neox.layers.0.attention.query_key_value.bias", (5376,))
check("gpt_neox.layers.0.attention.dense.weight", (2048, 1792))
if len(out) != 244:
    raise AssertionError(f"expected 244 tensors, got {len(out)}")

# per-layer sanity: every layer got the same treatment, dtypes preserved
for i in range(NUM_LAYERS):
    check(f"gpt_neox.layers.{i}.attention.query_key_value.weight", (5376, 2048))
    check(f"gpt_neox.layers.{i}.attention.query_key_value.bias", (5376,))
    check(f"gpt_neox.layers.{i}.attention.dense.weight", (2048, 1792))
    assert tuple(out[f"gpt_neox.layers.{i}.attention.dense.bias"].shape) == (2048,)
for k in out:  # dtypes must be preserved exactly (buffers are not fp16)
    assert out[k].dtype == src_dtypes[k], (k, out[k].dtype, src_dtypes[k])

save_file(out, DST, metadata={"format": "pt"})
print(f"wrote {DST}: {len(out)} tensors")
