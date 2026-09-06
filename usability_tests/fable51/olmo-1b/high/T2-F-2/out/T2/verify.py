"""Independent verification of out/T2/model.safetensors against inputs/base."""
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "inputs" / "base"
index = json.loads((SRC / "model.safetensors.index.json").read_text())
src = {}
for shard in sorted(set(index["weight_map"].values())):
    src.update(load_file(str(SRC / shard)))
out = load_file(str(ROOT / "out" / "T2" / "model.safetensors"))

assert set(src) == set(out), "key set differs"
assert len(out) == 114
pruned = 0
for k, a in src.items():
    b = out[k]
    assert a.dtype == b.dtype, k
    if ".self_attn." in k and k.endswith(("q_proj.weight", "k_proj.weight", "v_proj.weight")):
        ref = torch.cat([a[:640], a[768:]], dim=0)
        pruned += 1
    elif ".self_attn.o_proj.weight" in k:
        ref = torch.cat([a[:, :640], a[:, 768:]], dim=1)
        pruned += 1
    else:
        ref = a
    assert ref.shape == b.shape, (k, ref.shape, b.shape)
    assert torch.equal(ref, b), f"values differ: {k}"
assert pruned == 64, pruned
print("verify: all 114 tensors match expected (64 pruned, 50 untouched)")
