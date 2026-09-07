"""T4: task-vector merge of two fine-tunes (Pythia-1B), plain torch + safetensors."""
import re
import sys
import torch
from safetensors.torch import load_file, save_file

LAMBDA = 0.4
MLP_RE = re.compile(r"^gpt_neox\.layers\.\d+\.mlp\.dense_(h_to_4h|4h_to_h)\.(weight|bias)$")
OUT = "out/T4/model.safetensors"

def die(msg):
    print("ERROR:", msg, file=sys.stderr)
    sys.exit(1)

base = load_file("inputs/base/model.safetensors")
ft1 = load_file("inputs/ft1/model.safetensors")
ft2 = load_file("inputs/ft2/model.safetensors")

# Step 1: same names, and everything outside the MLP set identical in all three.
if not (base.keys() == ft1.keys() == ft2.keys()):
    die("tensor name sets differ between base/ft1/ft2")
mlp_names = {k for k in base if MLP_RE.match(k)}
if len(mlp_names) != 64:
    die(f"expected 64 MLP tensors, matched {len(mlp_names)}")
for k in base:
    for name, ft in (("ft1", ft1), ("ft2", ft2)):
        if ft[k].shape != base[k].shape or ft[k].dtype != base[k].dtype:
            die(f"{name}[{k}] shape/dtype differs from base")
        if k not in mlp_names and not torch.equal(base[k], ft[k]):
            die(f"shared tensor {k} differs between base and {name}")
print("verification passed: names match, 180 shared tensors identical")

# Step 2/3: merge MLP tensors against the unmodified base; copy the rest.
out = {}
merged = 0
for k, b in base.items():
    if k in mlp_names:
        b32 = b.float()
        m = b32 + LAMBDA * (ft1[k].float() - b32) + LAMBDA * (ft2[k].float() - b32)
        out[k] = m.to(b.dtype).contiguous()
        merged += 1
    else:
        out[k] = b
assert merged == 64, f"merged {merged} tensors, expected 64"
assert len(out) == 244, f"output has {len(out)} tensors, expected 244"
save_file(out, OUT, metadata={"format": "pt"})

# Post-write check.
chk = load_file(OUT)
assert len(chk) == 244, f"written file has {len(chk)} tensors"
assert chk.keys() == base.keys()
print(f"wrote {OUT}: {len(chk)} tensors, {merged} merged with lambda={LAMBDA}")
