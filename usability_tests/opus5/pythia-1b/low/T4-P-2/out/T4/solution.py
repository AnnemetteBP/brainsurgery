"""T4: task-vector merge of two Pythia-1B fine-tunes."""

import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

LAMBDA = 0.4
N_LAYERS = 16
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
BASE = ROOT / "inputs" / "base" / "model.safetensors"
FT1 = ROOT / "inputs" / "ft1" / "model.safetensors"
FT2 = ROOT / "inputs" / "ft2" / "model.safetensors"
OUT = HERE / "model.safetensors"

MLP_KEYS = [
    f"gpt_neox.layers.{i}.mlp.{proj}.{kind}"
    for i in range(N_LAYERS)
    for proj in ("dense_h_to_4h", "dense_4h_to_h")
    for kind in ("weight", "bias")
]


def fail(msg):
    raise SystemExit(f"ERROR: {msg}")


base = load_file(str(BASE))
ft1 = load_file(str(FT1))
ft2 = load_file(str(FT2))

# --- step 1: same names everywhere -----------------------------------------
if set(ft1) != set(base):
    fail(f"ft1 key set differs from base: {sorted(set(ft1) ^ set(base))[:10]}")
if set(ft2) != set(base):
    fail(f"ft2 key set differs from base: {sorted(set(ft2) ^ set(base))[:10]}")

missing = [k for k in MLP_KEYS if k not in base]
if missing:
    fail(f"expected MLP tensors absent from base: {missing}")
if len(MLP_KEYS) != 64:
    fail(f"expected 64 MLP tensor names, built {len(MLP_KEYS)}")

mlp = set(MLP_KEYS)

# --- step 1 (cont.): everything outside the MLP set is bit-identical --------
for name in sorted(base):
    if name in mlp:
        continue
    b = base[name]
    for tag, other in (("ft1", ft1[name]), ("ft2", ft2[name])):
        if other.shape != b.shape or other.dtype != b.dtype:
            fail(f"{name}: {tag} shape/dtype {tuple(other.shape)}/{other.dtype} "
                 f"!= base {tuple(b.shape)}/{b.dtype}")
        if not torch.equal(other, b):
            fail(f"{name}: {tag} differs from base but is outside the MLP set")

# shapes/dtypes of the MLP tensors themselves must line up too
for name in MLP_KEYS:
    b = base[name]
    for tag, other in (("ft1", ft1[name]), ("ft2", ft2[name])):
        if other.shape != b.shape or other.dtype != b.dtype:
            fail(f"{name}: {tag} shape/dtype {tuple(other.shape)}/{other.dtype} "
                 f"!= base {tuple(b.shape)}/{b.dtype}")

# --- step 2/3: merge, taking both task vectors against the untouched base ---
out = {}
merged = 0
for name, b in base.items():
    if name in mlp:
        b32 = b.to(torch.float32)
        tv1 = ft1[name].to(torch.float32) - b32
        tv2 = ft2[name].to(torch.float32) - b32
        out[name] = (b32 + LAMBDA * tv1 + LAMBDA * tv2).to(b.dtype).contiguous()
        merged += 1
    else:
        out[name] = b.clone().contiguous()

if merged != 64:
    fail(f"merged {merged} tensors, expected 64")
if len(out) != 244:
    fail(f"output has {len(out)} tensors, expected 244")

save_file(out, str(OUT))

check = load_file(str(OUT))
if len(check) != 244:
    fail(f"written file has {len(check)} tensors, expected 244")

print(f"wrote {OUT} ({len(check)} tensors, {merged} merged, lambda={LAMBDA})")
