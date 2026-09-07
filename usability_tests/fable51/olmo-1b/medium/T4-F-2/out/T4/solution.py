"""T4: task-vector merge of two fine-tunes (OLMo-1B-0724-hf).

out[X] = base[X] + lam*(ft1[X]-base[X]) + lam*(ft2[X]-base[X])  for the 48 MLP tensors,
everything else copied from the base unchanged. Plain safetensors + torch.
"""
import json
import os
import re
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE_DIR = os.path.join(ROOT, "inputs", "base")
FT1 = os.path.join(ROOT, "inputs", "ft1", "model.safetensors")
FT2 = os.path.join(ROOT, "inputs", "ft2", "model.safetensors")
OUT = os.path.join(ROOT, "out", "T4", "model.safetensors")

LAMBDA = 0.4
MLP_RE = re.compile(r"^model\.layers\.(\d+)\.mlp\.(gate_proj|up_proj|down_proj)\.weight$")
EXPECTED_MLP = 48
EXPECTED_TOTAL = 114


def fail(msg):
    raise SystemExit(f"ERROR: {msg}")


def load_dict(path_or_dir):
    """Load every tensor of a safetensors file or sharded directory (via index) into a dict."""
    if os.path.isdir(path_or_dir):
        with open(os.path.join(path_or_dir, "model.safetensors.index.json")) as f:
            index = json.load(f)
        files = sorted(set(index["weight_map"].values()))
        files = [os.path.join(path_or_dir, fn) for fn in files]
    else:
        files = [path_or_dir]
    out = {}
    for fn in files:
        with safe_open(fn, framework="pt", device="cpu") as f:
            for k in f.keys():
                if k in out:
                    fail(f"duplicate tensor {k} across shards of {path_or_dir}")
                out[k] = f.get_tensor(k)
    return out


def main():
    base = load_dict(BASE_DIR)
    ft1 = load_dict(FT1)
    ft2 = load_dict(FT2)

    # Step 1: same names in all three, and everything outside the MLP set identical.
    if set(base) != set(ft1) or set(base) != set(ft2):
        fail(
            "tensor name sets differ: "
            f"base^ft1={sorted(set(base) ^ set(ft1))[:5]} base^ft2={sorted(set(base) ^ set(ft2))[:5]}"
        )
    if len(base) != EXPECTED_TOTAL:
        fail(f"expected {EXPECTED_TOTAL} tensors in base, found {len(base)}")

    mlp_names = sorted(k for k in base if MLP_RE.match(k))
    if len(mlp_names) != EXPECTED_MLP:
        fail(f"expected {EXPECTED_MLP} MLP tensors, matched {len(mlp_names)}")

    for k in base:
        for name, other in (("ft1", ft1), ("ft2", ft2)):
            if other[k].shape != base[k].shape or other[k].dtype != base[k].dtype:
                fail(f"{k}: shape/dtype mismatch base={base[k].shape}/{base[k].dtype} "
                     f"{name}={other[k].shape}/{other[k].dtype}")
        if k in mlp_names:
            continue
        for name, other in (("ft1", ft1), ("ft2", ft2)):
            if not torch.equal(base[k], other[k]):
                fail(f"shared tensor {k} differs between base and {name}; refusing to merge")

    # Step 2/3: merge MLP tensors against the unmodified base; copy the rest.
    out = {}
    merged = 0
    for k in base:
        b = base[k]
        if k in mlp_names:
            if b.dtype != torch.float32:
                fail(f"{k}: expected float32, got {b.dtype}")
            out[k] = (b + LAMBDA * (ft1[k] - b) + LAMBDA * (ft2[k] - b)).to(torch.float32).contiguous()
            merged += 1
        else:
            out[k] = b.contiguous()

    if merged != EXPECTED_MLP:
        fail(f"merged {merged} tensors, expected {EXPECTED_MLP}")
    if len(out) != EXPECTED_TOTAL:
        fail(f"output has {len(out)} tensors, expected {EXPECTED_TOTAL}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    save_file(out, OUT, metadata={"format": "pt"})

    # Post-write verification of the file on disk.
    with safe_open(OUT, framework="pt", device="cpu") as f:
        keys = list(f.keys())
        if len(keys) != EXPECTED_TOTAL or set(keys) != set(base):
            fail(f"written file has {len(keys)} tensors / wrong key set")
        for k in base:
            t = f.get_tensor(k)
            if t.shape != base[k].shape or t.dtype != base[k].dtype:
                fail(f"written {k}: shape/dtype mismatch")
            if k not in mlp_names and not torch.equal(t, base[k]):
                fail(f"written {k}: unchanged tensor differs from base")
    print(f"OK: wrote {OUT} with {len(keys)} tensors, {merged} merged (lambda={LAMBDA})")


if __name__ == "__main__":
    sys.exit(main())
