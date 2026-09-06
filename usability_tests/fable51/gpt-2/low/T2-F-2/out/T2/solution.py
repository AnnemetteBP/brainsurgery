"""T2: remove head 5 from every GPT-2 layer at the checkpoint level."""
import os
import torch
from safetensors.torch import load_file, save_file

SRC = "inputs/base/model.safetensors"
DST = "out/T2/model.safetensors"
N_LAYERS, N_HEADS, HEAD_DIM, HIDDEN = 12, 12, 64, 768
PRUNE = 5

keep = [h for h in range(N_HEADS) if h != PRUNE]
head_idx = torch.cat([torch.arange(h * HEAD_DIM, (h + 1) * HEAD_DIM) for h in keep])  # 704
qkv_idx = torch.cat([head_idx + s * HIDDEN for s in range(3)])  # 2112

sd = load_file(SRC)
assert len(sd) == 160, len(sd)
out = {}
for k, v in sd.items():
    if k.endswith("attn.c_attn.weight"):
        out[k] = v[:, qkv_idx].contiguous()
    elif k.endswith("attn.c_attn.bias"):
        out[k] = v[qkv_idx].contiguous()
    elif k.endswith("attn.c_proj.weight"):
        out[k] = v[head_idx, :].contiguous()
    else:
        out[k] = v

# Required checks (fail loudly before writing).
assert tuple(out["h.0.attn.c_attn.weight"].shape) == (768, 2112), out["h.0.attn.c_attn.weight"].shape
assert tuple(out["h.0.attn.c_attn.bias"].shape) == (2112,), out["h.0.attn.c_attn.bias"].shape
assert tuple(out["h.0.attn.c_proj.weight"].shape) == (704, 768), out["h.0.attn.c_proj.weight"].shape
assert len(out) == 160, len(out)
for i in range(N_LAYERS):
    assert tuple(out[f"h.{i}.attn.c_attn.weight"].shape) == (768, 2112)
    assert tuple(out[f"h.{i}.attn.c_attn.bias"].shape) == (2112,)
    assert tuple(out[f"h.{i}.attn.c_proj.weight"].shape) == (704, 768)
    assert out[f"h.{i}.attn.c_proj.bias"].shape == sd[f"h.{i}.attn.c_proj.bias"].shape
    assert out[f"h.{i}.attn.bias"].shape == sd[f"h.{i}.attn.bias"].shape
# Spot-check block order on layer 0 against the spec's explicit column ranges.
w0, w1 = sd["h.0.attn.c_attn.weight"], out["h.0.attn.c_attn.weight"]
assert torch.equal(w1[:, 320:704], w0[:, 384:768])
assert torch.equal(w1[:, 704:1024], w0[:, 768:1088])
assert torch.equal(w1[:, 1408:1728], w0[:, 1536:1856])

os.makedirs(os.path.dirname(DST), exist_ok=True)
save_file(out, DST)
chk = load_file(DST)
assert len(chk) == 160
print("OK: wrote", DST, "with", len(chk), "tensors")
