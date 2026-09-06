"""T4: task-vector merge of two GPT-2 fine-tunes."""

from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

HERE = Path(__file__).resolve().parents[2]
BASE = HERE / "inputs/base/model.safetensors"
FT1 = HERE / "inputs/ft1/model.safetensors"
FT2 = HERE / "inputs/ft2/model.safetensors"
OUT = HERE / "out/T4/model.safetensors"

LAMBDA = 0.4

MLP_NAMES = [
    f"h.{i}.mlp.{mod}.{kind}"
    for i in range(12)
    for mod in ("c_fc", "c_proj")
    for kind in ("weight", "bias")
]
assert len(MLP_NAMES) == 48

base = load_file(str(BASE))
ft1 = load_file(str(FT1))
ft2 = load_file(str(FT2))

# 1. same tensor names everywhere
if not (set(base) == set(ft1) == set(ft2)):
    raise SystemExit(
        "tensor name sets differ: "
        f"base^ft1={sorted(set(base) ^ set(ft1))} base^ft2={sorted(set(base) ^ set(ft2))}"
    )

missing = [n for n in MLP_NAMES if n not in base]
if missing:
    raise SystemExit(f"expected MLP tensors missing from base: {missing}")

# 1b. everything outside the 48 MLP tensors must be identical in all three
mlp = set(MLP_NAMES)
for name in sorted(base):
    if name in mlp:
        continue
    b, a, c = base[name], ft1[name], ft2[name]
    if b.shape != a.shape or b.shape != c.shape:
        raise SystemExit(f"shape mismatch on shared tensor {name}: {b.shape} {a.shape} {c.shape}")
    if b.dtype != a.dtype or b.dtype != c.dtype:
        raise SystemExit(f"dtype mismatch on shared tensor {name}: {b.dtype} {a.dtype} {c.dtype}")
    if not torch.equal(b, a) or not torch.equal(b, c):
        raise SystemExit(f"non-MLP tensor differs between checkpoints: {name}")

# 2/3. merge; task vectors are always taken against the *unmodified* base
out = {}
merged = 0
for name in sorted(base):
    b = base[name]
    if name in mlp:
        f1, f2 = ft1[name], ft2[name]
        if b.shape != f1.shape or b.shape != f2.shape:
            raise SystemExit(f"shape mismatch on MLP tensor {name}")
        if b.dtype != torch.float32:
            raise SystemExit(f"expected float32 for {name}, got {b.dtype}")
        b32 = b.to(torch.float32)
        merged_t = b32 + LAMBDA * (f1.to(torch.float32) - b32) + LAMBDA * (f2.to(torch.float32) - b32)
        out[name] = merged_t.to(b.dtype).contiguous()
        merged += 1
    else:
        out[name] = b.clone().contiguous()

if merged != 48:
    raise SystemExit(f"expected 48 merged tensors, got {merged}")
if len(out) != 160:
    raise SystemExit(f"expected 160 output tensors, got {len(out)}")

OUT.parent.mkdir(parents=True, exist_ok=True)
save_file(out, str(OUT))

check = load_file(str(OUT))
if len(check) != 160:
    raise SystemExit(f"output file has {len(check)} tensors, expected 160")
print(f"wrote {OUT} with {len(check)} tensors, {merged} merged (lambda={LAMBDA})")
