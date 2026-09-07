"""T4: task-vector merge of two GPT-2 fine-tunes onto the base."""

import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

LAMBDA = 0.4
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "out" / "T4" / "model.safetensors"

MLP = {
    f"h.{i}.mlp.{proj}.{kind}"
    for i in range(12)
    for proj in ("c_fc", "c_proj")
    for kind in ("weight", "bias")
}


def fail(msg):
    raise SystemExit(f"T4 FAILED: {msg}")


base = load_file(ROOT / "inputs" / "base" / "model.safetensors")
ft1 = load_file(ROOT / "inputs" / "ft1" / "model.safetensors")
ft2 = load_file(ROOT / "inputs" / "ft2" / "model.safetensors")

# Step 1: same key sets, and everything outside the MLP tensors identical.
if not (set(base) == set(ft1) == set(ft2)):
    fail(
        "tensor names differ: "
        f"ft1 only={sorted(set(ft1) ^ set(base))[:5]} ft2 only={sorted(set(ft2) ^ set(base))[:5]}"
    )
missing = MLP - set(base)
if missing:
    fail(f"expected MLP tensors absent from base: {sorted(missing)}")
if len(MLP) != 48:
    fail(f"MLP tensor list has {len(MLP)} names, expected 48")

for name, b in base.items():
    for tag, other in (("ft1", ft1[name]), ("ft2", ft2[name])):
        if b.shape != other.shape or b.dtype != other.dtype:
            fail(f"{name}: {tag} has shape/dtype {other.shape}/{other.dtype}, base {b.shape}/{b.dtype}")
        if name not in MLP and not torch.equal(b, other):
            fail(f"non-MLP tensor {name} differs between base and {tag}")

# Step 2/3: task-vector merge, each vector taken against the untouched base.
out = {}
merged = 0
for name, b in base.items():
    if name in MLP:
        b32 = b.to(torch.float32)
        merged_t = b32 + LAMBDA * (ft1[name].to(torch.float32) - b32) + LAMBDA * (
            ft2[name].to(torch.float32) - b32
        )
        out[name] = merged_t.to(b.dtype).contiguous()
        merged += 1
    else:
        out[name] = b.clone()

if merged != 48:
    fail(f"merged {merged} tensors, expected 48")
if len(out) != 160:
    fail(f"output has {len(out)} tensors, expected 160")

OUT.parent.mkdir(parents=True, exist_ok=True)
save_file(out, str(OUT))

# Post-write verification of the required checks against the file on disk.
check = load_file(OUT)
if len(check) != 160:
    fail(f"written file has {len(check)} tensors, expected 160")
if set(check) != set(base):
    fail("written file key set differs from base")
changed = sum(1 for n in check if not torch.equal(check[n], base[n]))
if set(n for n in check if not torch.equal(check[n], base[n])) - MLP:
    fail("a non-MLP tensor changed in the output")
print(f"wrote {OUT} : {len(check)} tensors, {merged} merged, {changed} differ from base")
