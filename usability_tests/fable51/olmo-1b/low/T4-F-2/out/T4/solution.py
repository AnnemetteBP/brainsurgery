"""T4: task-vector merge of two fine-tunes (OLMo-1B). Plain torch + safetensors."""
import json
import re
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[2]
BASE_DIR = ROOT / "inputs" / "base"
FT1 = ROOT / "inputs" / "ft1" / "model.safetensors"
FT2 = ROOT / "inputs" / "ft2" / "model.safetensors"
OUT = ROOT / "out" / "T4" / "model.safetensors"
LAMBDA = 0.4
MLP_RE = re.compile(r"^model\.layers\.(\d+)\.mlp\.(gate_proj|up_proj|down_proj)\.weight$")


def fail(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def load_dir(d: Path) -> dict[str, torch.Tensor]:
    index = json.loads((d / "model.safetensors.index.json").read_text())
    out: dict[str, torch.Tensor] = {}
    for shard in sorted(set(index["weight_map"].values())):
        with safe_open(d / shard, framework="pt") as f:
            for k in f.keys():
                if k in out:
                    fail(f"duplicate key across shards: {k}")
                out[k] = f.get_tensor(k)
    return out


def load_file(p: Path) -> dict[str, torch.Tensor]:
    with safe_open(p, framework="pt") as f:
        return {k: f.get_tensor(k) for k in f.keys()}


base, ft1, ft2 = load_dir(BASE_DIR), load_file(FT1), load_file(FT2)

# Step 1: verify names and non-MLP identity BEFORE any arithmetic.
if not (set(base) == set(ft1) == set(ft2)):
    fail(f"tensor name sets differ: base={len(base)} ft1={len(ft1)} ft2={len(ft2)}")
if len(base) != 114:
    fail(f"expected 114 tensors in base, got {len(base)}")
mlp_keys = sorted(k for k in base if MLP_RE.match(k))
if len(mlp_keys) != 48:
    fail(f"expected 48 MLP tensors, matched {len(mlp_keys)}")
for k in base:
    for name, ft in (("ft1", ft1), ("ft2", ft2)):
        if ft[k].shape != base[k].shape or ft[k].dtype != base[k].dtype:
            fail(f"{name}[{k}] shape/dtype mismatch: {ft[k].shape}/{ft[k].dtype} vs {base[k].shape}/{base[k].dtype}")
    if k in mlp_keys:
        continue
    if not torch.equal(base[k], ft1[k]):
        fail(f"non-MLP tensor differs between base and ft1: {k}")
    if not torch.equal(base[k], ft2[k]):
        fail(f"non-MLP tensor differs between base and ft2: {k}")
print(f"verified: 114 shared names, {114 - len(mlp_keys)} non-MLP tensors identical in all three")

# Step 2: merge, each task vector against the unmodified base.
out: dict[str, torch.Tensor] = {}
merged = 0
for k, b in base.items():
    if k in mlp_keys:
        if b.dtype != torch.float32:
            fail(f"{k} is {b.dtype}, expected float32")
        out[k] = (b + LAMBDA * (ft1[k] - b) + LAMBDA * (ft2[k] - b)).contiguous()
        merged += 1
    else:
        out[k] = b
if merged != 48:
    fail(f"merged {merged} tensors, expected 48")
if len(out) != 114:
    fail(f"output has {len(out)} tensors, expected 114")

OUT.parent.mkdir(parents=True, exist_ok=True)
save_file(out, str(OUT), metadata={"format": "pt"})

with safe_open(OUT, framework="pt") as f:
    n = len(list(f.keys()))
if n != 114:
    fail(f"written file has {n} tensors, expected 114")
print(f"OK: merged {merged} MLP tensors, wrote {n} tensors to {OUT}")
