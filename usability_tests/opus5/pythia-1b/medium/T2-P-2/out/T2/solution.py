"""T2: structured attention-head pruning of Pythia-1B (remove head 5 in every layer)."""

from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

HERE = Path(__file__).resolve().parent
SRC = HERE.parents[1] / "inputs" / "base" / "model.safetensors"
DST = HERE / "model.safetensors"

N_LAYERS = 16
N_HEADS = 8
HEAD_DIM = 256
HIDDEN = 2048
DROP = 5

# Fused qkv is interleaved per head: head h owns rows 768*h .. 768*h+767.
QKV_BLOCK = 3 * HEAD_DIM
qkv_keep = torch.cat(
    [
        torch.arange(0, DROP * QKV_BLOCK),
        torch.arange((DROP + 1) * QKV_BLOCK, N_HEADS * QKV_BLOCK),
    ]
)
# Output projection consumes heads as 256-wide column blocks.
dense_keep = torch.cat(
    [
        torch.arange(0, DROP * HEAD_DIM),
        torch.arange((DROP + 1) * HEAD_DIM, N_HEADS * HEAD_DIM),
    ]
)

with safe_open(SRC, framework="pt") as f:
    metadata = f.metadata()
    tensors = {k: f.get_tensor(k) for k in f.keys()}

n_in = len(tensors)
if n_in != 244:
    raise AssertionError(f"input has {n_in} tensors, expected 244")

out = {}
for name, t in tensors.items():
    parts = name.split(".")
    is_layer = len(parts) > 2 and parts[0] == "gpt_neox" and parts[1] == "layers"
    suffix = ".".join(parts[3:]) if is_layer else None

    if suffix == "attention.query_key_value.weight":
        if tuple(t.shape) != (N_HEADS * QKV_BLOCK, HIDDEN):
            raise AssertionError(f"{name}: unexpected input shape {tuple(t.shape)}")
        new = t.index_select(0, qkv_keep)
    elif suffix == "attention.query_key_value.bias":
        if tuple(t.shape) != (N_HEADS * QKV_BLOCK,):
            raise AssertionError(f"{name}: unexpected input shape {tuple(t.shape)}")
        new = t.index_select(0, qkv_keep)
    elif suffix == "attention.dense.weight":
        if tuple(t.shape) != (HIDDEN, HIDDEN):
            raise AssertionError(f"{name}: unexpected input shape {tuple(t.shape)}")
        new = t.index_select(1, dense_keep)
    else:
        new = t

    if new.dtype != t.dtype:
        raise AssertionError(f"{name}: dtype changed {t.dtype} -> {new.dtype}")
    out[name] = new.contiguous()

# Every layer must have been touched exactly on its three head-bearing tensors.
touched = sum(
    1
    for i in range(N_LAYERS)
    for s in (
        "attention.query_key_value.weight",
        "attention.query_key_value.bias",
        "attention.dense.weight",
    )
    if f"gpt_neox.layers.{i}.{s}" in out
)
if touched != 3 * N_LAYERS:
    raise AssertionError(f"expected {3 * N_LAYERS} head-bearing tensors, found {touched}")

# Required checks.
expected = {
    "gpt_neox.layers.0.attention.query_key_value.weight": (5376, 2048),
    "gpt_neox.layers.0.attention.query_key_value.bias": (5376,),
    "gpt_neox.layers.0.attention.dense.weight": (2048, 1792),
}
for key, shape in expected.items():
    if key not in out:
        raise AssertionError(f"missing {key}")
    if tuple(out[key].shape) != shape:
        raise AssertionError(f"{key}: shape {tuple(out[key].shape)}, expected {shape}")
if len(out) != 244:
    raise AssertionError(f"output has {len(out)} tensors, expected 244")

# Spot-check the slicing against a direct row-range concatenation for every layer.
for i in range(N_LAYERS):
    src = tensors[f"gpt_neox.layers.{i}.attention.query_key_value.weight"]
    ref = torch.cat([src[0:3840], src[4608:6144]], dim=0)
    if not torch.equal(out[f"gpt_neox.layers.{i}.attention.query_key_value.weight"], ref):
        raise AssertionError(f"layer {i}: qkv weight rows do not match the required ranges")
    d = tensors[f"gpt_neox.layers.{i}.attention.dense.weight"]
    dref = torch.cat([d[:, 0:1280], d[:, 1536:2048]], dim=1)
    if not torch.equal(out[f"gpt_neox.layers.{i}.attention.dense.weight"], dref):
        raise AssertionError(f"layer {i}: dense weight columns do not match the required ranges")

DST.parent.mkdir(parents=True, exist_ok=True)
save_file(out, str(DST), metadata=metadata)
print(f"wrote {DST} with {len(out)} tensors")
