"""T2: remove attention head 5 from every layer of OLMo-1B-0724-hf.

Plain safetensors + torch script. Reads the sharded input via the index file,
slices the head-bearing projections, verifies the required shapes and tensor
count, then writes a single out/T2/model.safetensors.
"""
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "inputs" / "base"
OUT = ROOT / "out" / "T2" / "model.safetensors"

NUM_LAYERS = 16
NUM_HEADS = 16
HEAD_DIM = 128
HIDDEN = NUM_HEADS * HEAD_DIM
PRUNE_HEAD = 5
EXPECTED_TENSORS = 114

# Rows/columns to keep: everything except head 5's block [640, 768).
keep = torch.cat(
    [
        torch.arange(0, PRUNE_HEAD * HEAD_DIM),
        torch.arange((PRUNE_HEAD + 1) * HEAD_DIM, HIDDEN),
    ]
)
assert keep.numel() == HIDDEN - HEAD_DIM == 1920
assert keep[PRUNE_HEAD * HEAD_DIM - 1] == 639 and keep[PRUNE_HEAD * HEAD_DIM] == 768


def fail(msg: str) -> None:
    print(f"FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


# ---- load all shards -------------------------------------------------------
index = json.loads((SRC / "model.safetensors.index.json").read_text())
shards = sorted(set(index["weight_map"].values()))
state: dict[str, torch.Tensor] = {}
for shard in shards:
    part = load_file(str(SRC / shard))
    dup = set(part) & set(state)
    if dup:
        fail(f"duplicate keys across shards: {sorted(dup)[:5]}")
    state.update(part)
if len(state) != EXPECTED_TENSORS:
    fail(f"input has {len(state)} tensors, expected {EXPECTED_TENSORS}")

# ---- prune -----------------------------------------------------------------
for i in range(NUM_LAYERS):
    pre = f"model.layers.{i}.self_attn."
    for name in ("q_proj", "k_proj", "v_proj"):
        key = pre + name + ".weight"
        w = state[key]
        if tuple(w.shape) != (HIDDEN, HIDDEN):
            fail(f"{key} unexpected shape {tuple(w.shape)}")
        state[key] = w.index_select(0, keep).contiguous()  # rows = output = heads
    key = pre + "o_proj.weight"
    w = state[key]
    if tuple(w.shape) != (HIDDEN, HIDDEN):
        fail(f"{key} unexpected shape {tuple(w.shape)}")
    state[key] = w.index_select(1, keep).contiguous()  # columns = input = heads

# ---- required checks (fail loudly before writing) --------------------------
checks = {
    "model.layers.0.self_attn.q_proj.weight": (1920, 2048),
    "model.layers.0.self_attn.k_proj.weight": (1920, 2048),
    "model.layers.0.self_attn.v_proj.weight": (1920, 2048),
    "model.layers.0.self_attn.o_proj.weight": (2048, 1920),
}
for key, shape in checks.items():
    if key not in state:
        fail(f"missing tensor {key}")
    if tuple(state[key].shape) != shape:
        fail(f"{key} has shape {tuple(state[key].shape)}, expected {shape}")
if len(state) != EXPECTED_TENSORS:
    fail(f"output has {len(state)} tensors, expected {EXPECTED_TENSORS}")

# Extra sanity across all layers: same shapes everywhere, values untouched.
for i in range(NUM_LAYERS):
    pre = f"model.layers.{i}.self_attn."
    for name in ("q_proj", "k_proj", "v_proj"):
        if tuple(state[pre + name + ".weight"].shape) != (1920, 2048):
            fail(f"layer {i} {name} wrong shape")
    if tuple(state[pre + "o_proj.weight"].shape) != (2048, 1920):
        fail(f"layer {i} o_proj wrong shape")
for k, v in state.items():
    if v.dtype != torch.float32:
        fail(f"{k} dtype {v.dtype} is not float32")

# ---- write -----------------------------------------------------------------
if OUT.exists():
    fail(f"destination already exists: {OUT}")
OUT.parent.mkdir(parents=True, exist_ok=True)
save_file(state, str(OUT), metadata={"format": "pt"})
print(f"wrote {OUT} with {len(state)} tensors")
