"""T4: task-vector merge of two fine-tunes into OLMo-1B base (lambda = 0.4)."""
import json
import os
import re
import sys

import torch
from safetensors.torch import load_file, save_file

ROOT = os.path.dirname(os.path.abspath(__file__))
SANDBOX = os.path.abspath(os.path.join(ROOT, "..", ".."))
BASE_DIR = os.path.join(SANDBOX, "inputs", "base")
FT1 = os.path.join(SANDBOX, "inputs", "ft1", "model.safetensors")
FT2 = os.path.join(SANDBOX, "inputs", "ft2", "model.safetensors")
OUT = os.path.join(ROOT, "model.safetensors")
LAMBDA = 0.4
MLP_RE = re.compile(r"^model\.layers\.(\d+)\.mlp\.(gate_proj|up_proj|down_proj)\.weight$")


def fail(msg):
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def load_base():
    with open(os.path.join(BASE_DIR, "model.safetensors.index.json")) as f:
        index = json.load(f)
    sd = {}
    for shard in sorted(set(index["weight_map"].values())):
        part = load_file(os.path.join(BASE_DIR, shard))
        if set(part) & set(sd):
            fail(f"duplicate tensor names across shards in {shard}")
        sd.update(part)
    if set(sd) != set(index["weight_map"]):
        fail("base shard contents do not match index weight_map")
    return sd


base = load_base()
ft1 = load_file(FT1)
ft2 = load_file(FT2)

# Step 1: same names, and all non-MLP tensors bit-identical across the three.
if not (set(base) == set(ft1) == set(ft2)):
    fail(f"tensor name sets differ: base={len(base)} ft1={len(ft1)} ft2={len(ft2)}")
if len(base) != 114:
    fail(f"expected 114 tensors in base, got {len(base)}")

mlp_names = sorted(n for n in base if MLP_RE.match(n))
if len(mlp_names) != 48:
    fail(f"expected 48 MLP tensors, found {len(mlp_names)}")

for name in base:
    for tag, other in (("ft1", ft1), ("ft2", ft2)):
        b, o = base[name], other[name]
        if b.shape != o.shape or b.dtype != o.dtype:
            fail(f"{name}: shape/dtype mismatch between base and {tag}")
        if name not in mlp_names and not torch.equal(b, o):
            fail(f"{name}: differs between base and {tag} but is not an MLP tensor")

# Step 2/3: merge against the unmodified base; copy everything else.
out = {}
merged = 0
for name, b in base.items():
    if name in mlp_names:
        if b.dtype != torch.float32:
            fail(f"{name}: expected float32, got {b.dtype}")
        tv1 = ft1[name].float() - b
        tv2 = ft2[name].float() - b
        out[name] = (b + LAMBDA * tv1 + LAMBDA * tv2).to(torch.float32).contiguous()
        merged += 1
    else:
        out[name] = b.contiguous()

if merged != 48:
    fail(f"merged {merged} tensors, expected 48")
if len(out) != 114:
    fail(f"output has {len(out)} tensors, expected 114")

save_file(out, OUT, metadata={"format": "pt"})

# Post-write verification.
check = load_file(OUT)
if len(check) != 114 or set(check) != set(base):
    fail("written file does not contain exactly the 114 expected tensors")
for name in base:
    if name not in mlp_names and not torch.equal(check[name], base[name]):
        fail(f"{name}: unchanged tensor was altered in output")
print(f"OK: wrote {OUT} with {len(check)} tensors, {merged} merged (lambda={LAMBDA})")
