"""T4: task-vector merge of two Pythia-1B fine-tunes into the base."""
import re
import sys

import torch
from safetensors.torch import load_file, save_file

LAMBDA = 0.4
MLP_RE = re.compile(r"^gpt_neox\.layers\.\d+\.mlp\.dense_(h_to_4h|4h_to_h)\.(weight|bias)$")


def fail(msg):
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


base = load_file("inputs/base/model.safetensors")
ft1 = load_file("inputs/ft1/model.safetensors")
ft2 = load_file("inputs/ft2/model.safetensors")

# Step 1: verify layout and shared tensors before touching anything.
if not (base.keys() == ft1.keys() == ft2.keys()):
    fail("tensor name sets differ between base, ft1 and ft2")
if len(base) != 244:
    fail(f"expected 244 tensors in base, found {len(base)}")

mlp_names = sorted(k for k in base if MLP_RE.match(k))
if len(mlp_names) != 64:
    fail(f"expected 64 MLP tensors, matched {len(mlp_names)}")

for name in base:
    b, f1, f2 = base[name], ft1[name], ft2[name]
    if not (b.shape == f1.shape == f2.shape and b.dtype == f1.dtype == f2.dtype):
        fail(f"shape/dtype mismatch for {name}")
    if name in mlp_names:
        continue
    if not (torch.equal(b, f1) and torch.equal(b, f2)):
        fail(f"non-MLP tensor differs between checkpoints: {name}")

# Step 2/3: merge MLP tensors against the unmodified base; copy everything else.
out = {}
merged = 0
for name in base:
    b = base[name]
    if name in mlp_names:
        b32 = b.float()
        tv = LAMBDA * (ft1[name].float() - b32) + LAMBDA * (ft2[name].float() - b32)
        out[name] = (b32 + tv).to(b.dtype).contiguous()
        merged += 1
    else:
        out[name] = b.clone().contiguous()

if merged != 64:
    fail(f"merged {merged} tensors, expected 64")
if len(out) != 244:
    fail(f"output has {len(out)} tensors, expected 244")

save_file(out, "out/T4/model.safetensors")
check = load_file("out/T4/model.safetensors")
if len(check) != 244:
    fail(f"saved file has {len(check)} tensors, expected 244")
print(f"OK: merged {merged} MLP tensors, wrote {len(check)} tensors to out/T4/model.safetensors")
