"""T2: remove attention head 5 from every layer of GPT-2 (124M) at checkpoint level.

Uses safetensors + torch only. Slices the fused Conv1D c_attn [q|k|v] columns/bias
entries and the c_proj rows for the pruned head; all other tensors pass through
untouched (same dtype, same names). Fails loudly if the required shapes or the
tensor count do not hold before writing.
"""
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

SRC = Path("inputs/base/model.safetensors")
DST = Path("out/T2/model.safetensors")

N_LAYERS, N_HEADS, HEAD_DIM, HIDDEN = 12, 12, 64, 768
PRUNE_HEAD = 5
EXPECTED_TENSORS = 160

keep_heads = [h for h in range(N_HEADS) if h != PRUNE_HEAD]
# Index of retained columns within one 768-wide segment: heads are 64-wide blocks.
seg_keep = torch.cat([torch.arange(h * HEAD_DIM, (h + 1) * HEAD_DIM) for h in keep_heads])
assert seg_keep.numel() == (N_HEADS - 1) * HEAD_DIM == 704
# Same index for q, k, v segments of the fused projection, offset by 768 each.
qkv_keep = torch.cat([seg_keep + s * HIDDEN for s in range(3)])
assert qkv_keep.numel() == 2112

sd = load_file(str(SRC))
assert len(sd) == EXPECTED_TENSORS, f"input has {len(sd)} tensors, expected {EXPECTED_TENSORS}"

out = {}
for name, t in sd.items():
    out[name] = t

for i in range(N_LAYERS):
    w = sd[f"h.{i}.attn.c_attn.weight"]
    b = sd[f"h.{i}.attn.c_attn.bias"]
    p = sd[f"h.{i}.attn.c_proj.weight"]
    assert w.shape == (HIDDEN, 3 * HIDDEN), (i, w.shape)
    assert b.shape == (3 * HIDDEN,), (i, b.shape)
    assert p.shape == (HIDDEN, HIDDEN), (i, p.shape)
    out[f"h.{i}.attn.c_attn.weight"] = w.index_select(1, qkv_keep).contiguous()
    out[f"h.{i}.attn.c_attn.bias"] = b.index_select(0, qkv_keep).contiguous()
    out[f"h.{i}.attn.c_proj.weight"] = p.index_select(0, seg_keep).contiguous()

# Required checks (fail loudly before writing).
assert tuple(out["h.0.attn.c_attn.weight"].shape) == (768, 2112), out["h.0.attn.c_attn.weight"].shape
assert tuple(out["h.0.attn.c_attn.bias"].shape) == (2112,), out["h.0.attn.c_attn.bias"].shape
assert tuple(out["h.0.attn.c_proj.weight"].shape) == (704, 768), out["h.0.attn.c_proj.weight"].shape
assert len(out) == EXPECTED_TENSORS, f"output has {len(out)} tensors, expected {EXPECTED_TENSORS}"

# Extra sanity: every layer, dtype preserved, untouched tensors bit-identical.
for i in range(N_LAYERS):
    assert tuple(out[f"h.{i}.attn.c_attn.weight"].shape) == (768, 2112)
    assert tuple(out[f"h.{i}.attn.c_attn.bias"].shape) == (2112,)
    assert tuple(out[f"h.{i}.attn.c_proj.weight"].shape) == (704, 768)
    # spot-check block boundaries: retained block 5 (output) must equal source head 6
    assert torch.equal(out[f"h.{i}.attn.c_attn.weight"][:, 320:384], sd[f"h.{i}.attn.c_attn.weight"][:, 384:448])
    assert torch.equal(out[f"h.{i}.attn.c_proj.weight"][320:384], sd[f"h.{i}.attn.c_proj.weight"][384:448])
for name, t in out.items():
    assert t.dtype == sd[name].dtype, name
    if not name.endswith(("attn.c_attn.weight", "attn.c_attn.bias", "attn.c_proj.weight")):
        assert torch.equal(t, sd[name]), name
assert set(out) == set(sd)

DST.parent.mkdir(parents=True, exist_ok=True)
save_file(out, str(DST), metadata={"format": "pt"})

# Post-write verification.
chk = load_file(str(DST))
assert len(chk) == EXPECTED_TENSORS
assert tuple(chk["h.0.attn.c_attn.weight"].shape) == (768, 2112)
print(f"wrote {DST} with {len(chk)} tensors")
