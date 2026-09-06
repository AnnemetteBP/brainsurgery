"""T4: task-vector merge of two Pythia-1B fine-tunes (lambda = 0.4).

Plain torch + safetensors. Steps:
  1. Verify the three checkpoints share the same tensor names and that every
     non-MLP tensor is bit-identical in all three (abort otherwise).
  2. For the 64 MLP tensors compute base + l*(ft1-base) + l*(ft2-base) in
     float32 against the unmodified base, cast back to the base dtype.
  3. Copy everything else from the base unchanged; save 244 tensors.
"""
import re
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

LAMBDA = 0.4
N_LAYERS = 16
EXPECTED_TOTAL = 244
EXPECTED_MLP = 64
MLP_RE = re.compile(r"^gpt_neox\.layers\.(\d+)\.mlp\.dense_(h_to_4h|4h_to_h)\.(weight|bias)$")

root = Path(__file__).resolve().parents[2]
base = load_file(root / "inputs/base/model.safetensors")
ft1 = load_file(root / "inputs/ft1/model.safetensors")
ft2 = load_file(root / "inputs/ft2/model.safetensors")

# --- Step 1: shared-tensor verification, before touching anything ---------
names = set(base)
if not (names == set(ft1) == set(ft2)):
    sys.exit(f"ERROR: tensor name sets differ: base={len(base)} ft1={len(ft1)} ft2={len(ft2)} "
             f"(base-ft1={sorted(names ^ set(ft1))[:5]}, base-ft2={sorted(names ^ set(ft2))[:5]})")
if len(base) != EXPECTED_TOTAL:
    sys.exit(f"ERROR: expected {EXPECTED_TOTAL} tensors, base has {len(base)}")

mlp_names = {n for n in names if MLP_RE.match(n) and int(MLP_RE.match(n).group(1)) < N_LAYERS}
if len(mlp_names) != EXPECTED_MLP:
    sys.exit(f"ERROR: expected {EXPECTED_MLP} MLP tensors, matched {len(mlp_names)}")

for n in sorted(names):
    b, f1, f2 = base[n], ft1[n], ft2[n]
    for tag, t in (("ft1", f1), ("ft2", f2)):
        if t.shape != b.shape or t.dtype != b.dtype:
            sys.exit(f"ERROR: {n}: {tag} shape/dtype {tuple(t.shape)}/{t.dtype} "
                     f"!= base {tuple(b.shape)}/{b.dtype}")
    if n not in mlp_names:
        if not (torch.equal(b, f1) and torch.equal(b, f2)):
            sys.exit(f"ERROR: non-MLP tensor differs between checkpoints: {n}")

# --- Step 2/3: merge against the unmodified base ------------------------
out = {}
merged = 0
for n in sorted(names):
    b = base[n]
    if n in mlp_names:
        b32 = b.float()
        r = b32 + LAMBDA * (ft1[n].float() - b32) + LAMBDA * (ft2[n].float() - b32)
        out[n] = r.to(b.dtype).contiguous()
        merged += 1
    else:
        out[n] = b.contiguous()

# --- Required checks -------------------------------------------------------
if merged != EXPECTED_MLP:
    sys.exit(f"ERROR: merged {merged} tensors, expected {EXPECTED_MLP}")
if len(out) != EXPECTED_TOTAL:
    sys.exit(f"ERROR: output has {len(out)} tensors, expected {EXPECTED_TOTAL}")
for n in names - mlp_names:
    assert torch.equal(out[n], base[n]), n

dest = root / "out/T4/model.safetensors"
save_file(out, str(dest), metadata={"format": "pt"})
reloaded = load_file(dest)
if len(reloaded) != EXPECTED_TOTAL:
    sys.exit(f"ERROR: saved file has {len(reloaded)} tensors, expected {EXPECTED_TOTAL}")
print(f"OK: wrote {dest} with {len(reloaded)} tensors, {merged} merged, lambda={LAMBDA}")
