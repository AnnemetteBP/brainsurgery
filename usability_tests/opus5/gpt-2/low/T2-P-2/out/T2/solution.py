"""T2: remove attention head 5 from every layer of GPT-2 (124M)."""

from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

HERE = Path(__file__).resolve().parent
SANDBOX = HERE.parent.parent
SRC = SANDBOX / "inputs" / "base" / "model.safetensors"
DST = HERE / "model.safetensors"

N_LAYERS = 12
N_HEADS = 12
HEAD_DIM = 64
HIDDEN = N_HEADS * HEAD_DIM  # 768
PRUNE = 5

# Indices to keep within one 768-wide q/k/v segment: all heads except head 5.
keep_in_segment = [i for i in range(HIDDEN) if not (PRUNE * HEAD_DIM <= i < (PRUNE + 1) * HEAD_DIM)]
keep_head = torch.tensor(keep_in_segment, dtype=torch.long)
# c_attn is the fused [q | k | v] projection: repeat the pattern per segment.
keep_qkv = torch.cat([keep_head + seg * HIDDEN for seg in range(3)])

state = load_file(str(SRC))
n_in = len(state)
out = dict(state)

for i in range(N_LAYERS):
    w = f"h.{i}.attn.c_attn.weight"
    b = f"h.{i}.attn.c_attn.bias"
    p = f"h.{i}.attn.c_proj.weight"
    for name in (w, b, p):
        if name not in state:
            raise SystemExit(f"missing tensor: {name}")
    if state[w].shape != (HIDDEN, 3 * HIDDEN):
        raise SystemExit(f"{w}: unexpected shape {tuple(state[w].shape)}")
    if state[b].shape != (3 * HIDDEN,):
        raise SystemExit(f"{b}: unexpected shape {tuple(state[b].shape)}")
    if state[p].shape != (HIDDEN, HIDDEN):
        raise SystemExit(f"{p}: unexpected shape {tuple(state[p].shape)}")

    # c_attn: heads are column blocks (Conv1D [in, out]).
    out[w] = state[w][:, keep_qkv].contiguous()
    out[b] = state[b][keep_qkv].contiguous()
    # c_proj: heads are row blocks (the input side of the output projection).
    out[p] = state[p][keep_head, :].contiguous()

# Required checks: fail loudly before writing.
exp_cols = 3 * (HIDDEN - HEAD_DIM)  # 2112
exp_rows = HIDDEN - HEAD_DIM  # 704
checks = [
    ("h.0.attn.c_attn.weight", (HIDDEN, exp_cols)),
    ("h.0.attn.c_attn.bias", (exp_cols,)),
    ("h.0.attn.c_proj.weight", (exp_rows, HIDDEN)),
]
for name, want in checks:
    got = tuple(out[name].shape)
    if got != want:
        raise SystemExit(f"check failed: {name} has shape {got}, expected {want}")
if len(out) != 160:
    raise SystemExit(f"check failed: output has {len(out)} tensors, expected 160")
if len(out) != n_in:
    raise SystemExit(f"check failed: tensor count changed {n_in} -> {len(out)}")

save_file(out, str(DST))
print(f"wrote {DST} with {len(out)} tensors")
for name, want in checks:
    print(f"  {name}: {tuple(out[name].shape)}")
