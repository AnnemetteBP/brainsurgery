"""T4: task-vector merge of two GPT-2 fine-tunes (lambda = 0.4), MLP tensors only."""
import re
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parents[2]
LAMBDA = 0.4
MLP_RE = re.compile(r"^h\.(\d+)\.mlp\.(c_fc|c_proj)\.(weight|bias)$")


def fail(msg):
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


base = load_file(ROOT / "inputs/base/model.safetensors")
ft1 = load_file(ROOT / "inputs/ft1/model.safetensors")
ft2 = load_file(ROOT / "inputs/ft2/model.safetensors")

# Step 1: same names, and everything outside the MLP set identical in all three.
if not (set(base) == set(ft1) == set(ft2)):
    fail(f"tensor name sets differ: base={len(base)} ft1={len(ft1)} ft2={len(ft2)}")
if len(base) != 160:
    fail(f"expected 160 tensors in base, got {len(base)}")

mlp_names = sorted(n for n in base if MLP_RE.match(n))
if len(mlp_names) != 48:
    fail(f"expected 48 MLP tensors, matched {len(mlp_names)}")

for n in base:
    for tag, other in (("ft1", ft1), ("ft2", ft2)):
        if base[n].shape != other[n].shape or base[n].dtype != other[n].dtype:
            fail(f"{n}: shape/dtype mismatch between base and {tag}")
    if n in mlp_names:
        continue
    if not (torch.equal(base[n], ft1[n]) and torch.equal(base[n], ft2[n])):
        fail(f"non-MLP tensor {n} differs between checkpoints")

# Step 2/3: merge MLP tensors against the unmodified base; copy the rest.
out = {}
merged = 0
for n, b in base.items():
    if n in mlp_names:
        if b.dtype != torch.float32:
            fail(f"{n}: expected float32, got {b.dtype}")
        out[n] = (b + LAMBDA * (ft1[n] - b) + LAMBDA * (ft2[n] - b)).contiguous()
        merged += 1
    else:
        out[n] = b.contiguous()

if merged != 48:
    fail(f"merged {merged} tensors, expected 48")
if len(out) != 160:
    fail(f"output has {len(out)} tensors, expected 160")

dest = ROOT / "out/T4/model.safetensors"
save_file(out, dest)

# Re-open and confirm.
check = load_file(dest)
if len(check) != 160 or set(check) != set(base):
    fail("re-loaded output does not have the expected 160 tensor names")
print(f"OK: wrote {dest} with {len(check)} tensors, {merged} merged (lambda={LAMBDA})")
