"""T4: task-vector merge of two fine-tunes into the OLMo-1B base (lambda = 0.4)."""
import json
import os
import re
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
BASE_DIR = os.path.join(ROOT, "inputs", "base")
FT1 = os.path.join(ROOT, "inputs", "ft1", "model.safetensors")
FT2 = os.path.join(ROOT, "inputs", "ft2", "model.safetensors")
OUT = os.path.join(HERE, "model.safetensors")

LAMBDA = 0.4
N_LAYERS = 16
MLP_RE = re.compile(r"^model\.layers\.(\d+)\.mlp\.(gate_proj|up_proj|down_proj)\.weight$")


def fail(msg):
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def load_dir(path):
    """Load a sharded or single-file safetensors checkpoint into a dict."""
    index = os.path.join(path, "model.safetensors.index.json")
    if os.path.isfile(index):
        with open(index) as f:
            weight_map = json.load(f)["weight_map"]
        shards = sorted(set(weight_map.values()))
        out = {}
        for shard in shards:
            with safe_open(os.path.join(path, shard), framework="pt") as f:
                for k in f.keys():
                    if k in out:
                        fail(f"duplicate tensor {k} across shards of {path}")
                    out[k] = f.get_tensor(k)
        if set(out) != set(weight_map):
            fail(f"index/shard mismatch in {path}")
        return out
    with safe_open(path, framework="pt") as f:
        return {k: f.get_tensor(k) for k in f.keys()}


def main():
    base = load_dir(BASE_DIR)
    ft1 = load_dir(FT1)
    ft2 = load_dir(FT2)

    # Step 1: same names, and everything outside the MLP set identical in all three.
    names = set(base)
    if len(names) != 114:
        fail(f"base has {len(names)} tensors, expected 114")
    if set(ft1) != names:
        fail(f"ft1 names differ from base: {sorted(set(ft1) ^ names)[:10]}")
    if set(ft2) != names:
        fail(f"ft2 names differ from base: {sorted(set(ft2) ^ names)[:10]}")

    mlp_names = {n for n in names if MLP_RE.match(n)}
    expected_mlp = {
        f"model.layers.{i}.mlp.{p}.weight"
        for i in range(N_LAYERS)
        for p in ("gate_proj", "up_proj", "down_proj")
    }
    if mlp_names != expected_mlp:
        fail(f"MLP tensor set mismatch: {sorted(mlp_names ^ expected_mlp)[:10]}")
    if len(mlp_names) != 48:
        fail(f"found {len(mlp_names)} MLP tensors, expected 48")

    for n in sorted(names):
        b, a1, a2 = base[n], ft1[n], ft2[n]
        for tag, t in (("ft1", a1), ("ft2", a2)):
            if t.shape != b.shape or t.dtype != b.dtype:
                fail(f"{tag}[{n}] shape/dtype {tuple(t.shape)}/{t.dtype} "
                     f"!= base {tuple(b.shape)}/{b.dtype}")
        if n not in mlp_names:
            if not torch.equal(b, a1):
                fail(f"non-MLP tensor differs between base and ft1: {n}")
            if not torch.equal(b, a2):
                fail(f"non-MLP tensor differs between base and ft2: {n}")
    print(f"verified: 114 shared names, {114 - len(mlp_names)} non-MLP tensors identical")

    # Step 2/3: merge MLP tensors against the unmodified base; copy the rest.
    out = {}
    merged = 0
    for n in sorted(names):
        b = base[n]
        if n in mlp_names:
            if b.dtype != torch.float32:
                fail(f"{n} is {b.dtype}, expected float32")
            tv1 = ft1[n] - b
            tv2 = ft2[n] - b
            out[n] = (b + LAMBDA * tv1 + LAMBDA * tv2).to(torch.float32).contiguous()
            merged += 1
        else:
            out[n] = b.contiguous()

    if merged != 48:
        fail(f"merged {merged} tensors, expected 48")
    if len(out) != 114:
        fail(f"output has {len(out)} tensors, expected 114")

    if os.path.exists(OUT):
        fail(f"output already exists: {OUT}")
    save_file(out, OUT)

    with safe_open(OUT, framework="pt") as f:
        keys = list(f.keys())
    if len(keys) != 114 or set(keys) != names:
        fail(f"written file has {len(keys)} tensors / wrong key set")
    print(f"wrote {OUT}: {len(keys)} tensors, {merged} merged with lambda={LAMBDA}")


if __name__ == "__main__":
    main()
