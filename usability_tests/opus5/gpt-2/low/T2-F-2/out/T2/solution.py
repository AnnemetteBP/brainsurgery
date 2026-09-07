"""T2: remove attention head 5 from every GPT-2 layer at the checkpoint level.

Plain torch + safetensors script (see REPORT.md for why).
"""

import torch
from safetensors.torch import load_file, save_file

IN_PATH = "inputs/base/model.safetensors"
OUT_PATH = "out/T2/model.safetensors"

N_LAYERS = 12
N_HEADS = 12
HEAD_DIM = 64
HIDDEN = N_HEADS * HEAD_DIM  # 768
PRUNE = 5

# Indices to keep inside one 768-wide head-partitioned segment.
keep_seg = [i for i in range(HIDDEN) if i // HEAD_DIM != PRUNE]
# c_attn is [q | k | v], three consecutive 768-wide segments.
keep_qkv = torch.tensor(
    [s * HIDDEN + i for s in range(3) for i in keep_seg], dtype=torch.long
)
keep_out = torch.tensor(keep_seg, dtype=torch.long)

sd = load_file(IN_PATH)
n_in = len(sd)

for i in range(N_LAYERS):
    w = f"h.{i}.attn.c_attn.weight"
    b = f"h.{i}.attn.c_attn.bias"
    p = f"h.{i}.attn.c_proj.weight"
    for k in (w, b, p):
        if k not in sd:
            raise KeyError(f"missing expected tensor {k}")
    if tuple(sd[w].shape) != (HIDDEN, 3 * HIDDEN):
        raise ValueError(f"{w}: unexpected input shape {tuple(sd[w].shape)}")
    if tuple(sd[b].shape) != (3 * HIDDEN,):
        raise ValueError(f"{b}: unexpected input shape {tuple(sd[b].shape)}")
    if tuple(sd[p].shape) != (HIDDEN, HIDDEN):
        raise ValueError(f"{p}: unexpected input shape {tuple(sd[p].shape)}")
    sd[w] = sd[w].index_select(1, keep_qkv).contiguous()  # heads are columns
    sd[b] = sd[b].index_select(0, keep_qkv).contiguous()
    sd[p] = sd[p].index_select(0, keep_out).contiguous()  # heads are rows

# Required checks: fail loudly before writing.
expected = {
    "h.0.attn.c_attn.weight": (768, 2112),
    "h.0.attn.c_attn.bias": (2112,),
    "h.0.attn.c_proj.weight": (704, 768),
}
for k, shape in expected.items():
    got = tuple(sd[k].shape)
    if got != shape:
        raise AssertionError(f"{k}: expected shape {shape}, got {got}")
if len(sd) != 160:
    raise AssertionError(f"expected 160 tensors, got {len(sd)}")
if len(sd) != n_in:
    raise AssertionError(f"tensor count changed: {n_in} -> {len(sd)}")

# Shape check on every head-bearing projection, not just layer 0.
for i in range(N_LAYERS):
    for k, shape in (
        (f"h.{i}.attn.c_attn.weight", (768, 2112)),
        (f"h.{i}.attn.c_attn.bias", (2112,)),
        (f"h.{i}.attn.c_proj.weight", (704, 768)),
        (f"h.{i}.attn.c_proj.bias", (768,)),
        (f"h.{i}.attn.bias", (1, 1, 1024, 1024)),
    ):
        got = tuple(sd[k].shape)
        if got != shape:
            raise AssertionError(f"{k}: expected shape {shape}, got {got}")

save_file(sd, OUT_PATH, metadata={"format": "pt"})
print(f"wrote {OUT_PATH}: {len(sd)} tensors")
