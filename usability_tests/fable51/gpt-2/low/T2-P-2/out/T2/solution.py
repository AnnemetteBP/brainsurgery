"""T2: prune attention head 5 from every layer of GPT-2 (124M)."""
import os
import torch
from safetensors.torch import load_file, save_file

IN = "inputs/base/model.safetensors"
OUT = "out/T2/model.safetensors"
N_LAYERS, N_HEADS, HEAD_DIM, HIDDEN = 12, 12, 64, 768
PRUNE = 5

keep_heads = [h for h in range(N_HEADS) if h != PRUNE]
# column/row indices to keep within one 768-wide head-bearing segment
seg_idx = torch.tensor([h * HEAD_DIM + d for h in keep_heads for d in range(HEAD_DIM)])
# fused [q | k | v]: same pattern in each of the three segments
qkv_idx = torch.cat([seg_idx + s * HIDDEN for s in range(3)])

sd = load_file(IN)
assert len(sd) == 160, f"expected 160 input tensors, got {len(sd)}"

for i in range(N_LAYERS):
    p = f"h.{i}.attn."
    w = sd[p + "c_attn.weight"]
    b = sd[p + "c_attn.bias"]
    o = sd[p + "c_proj.weight"]
    assert w.shape == (HIDDEN, 3 * HIDDEN), w.shape
    assert b.shape == (3 * HIDDEN,), b.shape
    assert o.shape == (HIDDEN, HIDDEN), o.shape
    sd[p + "c_attn.weight"] = w[:, qkv_idx].contiguous()
    sd[p + "c_attn.bias"] = b[qkv_idx].contiguous()
    sd[p + "c_proj.weight"] = o[seg_idx, :].contiguous()

# Required checks
assert sd["h.0.attn.c_attn.weight"].shape == (768, 2112), sd["h.0.attn.c_attn.weight"].shape
assert sd["h.0.attn.c_attn.bias"].shape == (2112,), sd["h.0.attn.c_attn.bias"].shape
assert sd["h.0.attn.c_proj.weight"].shape == (704, 768), sd["h.0.attn.c_proj.weight"].shape
assert len(sd) == 160, len(sd)
for i in range(N_LAYERS):
    assert sd[f"h.{i}.attn.c_attn.weight"].shape == (768, 2112)
    assert sd[f"h.{i}.attn.c_attn.bias"].shape == (2112,)
    assert sd[f"h.{i}.attn.c_proj.weight"].shape == (704, 768)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
save_file(sd, OUT, metadata={"format": "pt"})
print(f"wrote {OUT} with {len(sd)} tensors")
