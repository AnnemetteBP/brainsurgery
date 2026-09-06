"""T4: task-vector merge of two fine-tunes into OLMo-1B-0724-hf base.

out[X] = base[X] + lam*(ft1[X]-base[X]) + lam*(ft2[X]-base[X]) for the 48 MLP
tensors; every other tensor is verified identical across base/ft1/ft2 and
copied from the base unchanged.
"""

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
FT1_PATH = os.path.join(ROOT, "inputs", "ft1", "model.safetensors")
FT2_PATH = os.path.join(ROOT, "inputs", "ft2", "model.safetensors")
OUT_PATH = os.path.join(HERE, "model.safetensors")

LAMBDA = 0.4
N_LAYERS = 16
EXPECTED_TOTAL = 114
EXPECTED_MLP = 48
MLP_RE = re.compile(r"^model\.layers\.(\d+)\.mlp\.(gate_proj|up_proj|down_proj)\.weight$")


def fail(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


class ShardedReader:
    """Read tensors by name from a sharded or single-file safetensors checkpoint."""

    def __init__(self, path: str):
        self.handles: dict[str, object] = {}
        self.where: dict[str, str] = {}
        if os.path.isdir(path):
            index = os.path.join(path, "model.safetensors.index.json")
            with open(index) as f:
                weight_map = json.load(f)["weight_map"]
            for name, shard in weight_map.items():
                self.where[name] = os.path.join(path, shard)
        else:
            self.where = {}
            with safe_open(path, framework="pt", device="cpu") as f:
                for name in f.keys():
                    self.where[name] = path
        for shard in sorted(set(self.where.values())):
            self.handles[shard] = safe_open(shard, framework="pt", device="cpu")
        # Cross-check the index against what the shards actually contain.
        actual = set()
        for h in self.handles.values():
            actual.update(h.keys())
        if actual != set(self.where):
            fail(f"{path}: index lists {len(self.where)} tensors, shards contain {len(actual)}")

    def keys(self) -> set[str]:
        return set(self.where)

    def get(self, name: str) -> torch.Tensor:
        return self.handles[self.where[name]].get_tensor(name)


def main() -> None:
    base = ShardedReader(BASE_DIR)
    ft1 = ShardedReader(FT1_PATH)
    ft2 = ShardedReader(FT2_PATH)

    # --- Step 1: same names in all three, identical outside the MLP tensors ---
    names = base.keys()
    if names != ft1.keys():
        fail(f"tensor names differ between base and ft1: "
             f"base-only={sorted(names - ft1.keys())[:5]} ft1-only={sorted(ft1.keys() - names)[:5]}")
    if names != ft2.keys():
        fail(f"tensor names differ between base and ft2: "
             f"base-only={sorted(names - ft2.keys())[:5]} ft2-only={sorted(ft2.keys() - names)[:5]}")
    if len(names) != EXPECTED_TOTAL:
        fail(f"expected {EXPECTED_TOTAL} tensors, found {len(names)}")

    mlp_names = {n for n in names if MLP_RE.match(n) and int(MLP_RE.match(n).group(1)) < N_LAYERS}
    expected_mlp = {
        f"model.layers.{i}.mlp.{p}.weight"
        for i in range(N_LAYERS)
        for p in ("gate_proj", "up_proj", "down_proj")
    }
    if mlp_names != expected_mlp:
        fail(f"MLP tensor set mismatch: missing={sorted(expected_mlp - mlp_names)} "
             f"extra={sorted(mlp_names - expected_mlp)}")
    shared_names = names - mlp_names

    out: dict[str, torch.Tensor] = {}
    mismatched = []
    for name in sorted(shared_names):
        b = base.get(name)
        t1 = ft1.get(name)
        t2 = ft2.get(name)
        for tag, t in (("ft1", t1), ("ft2", t2)):
            if t.shape != b.shape or t.dtype != b.dtype:
                mismatched.append(f"{name}: {tag} shape/dtype {tuple(t.shape)}/{t.dtype} "
                                  f"vs base {tuple(b.shape)}/{b.dtype}")
            elif not torch.equal(b, t):
                mismatched.append(f"{name}: {tag} values differ from base")
        out[name] = b
    if mismatched:
        fail("shared-tensor verification failed for "
             f"{len(mismatched)} tensor(s):\n  " + "\n  ".join(mismatched[:20]))
    print(f"verified {len(shared_names)} shared tensors identical across base/ft1/ft2")

    # --- Step 2: merge the MLP tensors against the unmodified base ---
    merged = 0
    for name in sorted(mlp_names):
        b = base.get(name)
        t1 = ft1.get(name)
        t2 = ft2.get(name)
        for tag, t in (("ft1", t1), ("ft2", t2)):
            if t.shape != b.shape or t.dtype != b.dtype:
                fail(f"{name}: {tag} shape/dtype {tuple(t.shape)}/{t.dtype} "
                     f"vs base {tuple(b.shape)}/{b.dtype}")
        if b.dtype != torch.float32:
            fail(f"{name}: expected float32, got {b.dtype}")
        b32 = b.to(torch.float32)
        tv1 = t1.to(torch.float32) - b32
        tv2 = t2.to(torch.float32) - b32
        out[name] = (b32 + LAMBDA * tv1 + LAMBDA * tv2).contiguous()
        merged += 1

    if merged != EXPECTED_MLP:
        fail(f"expected to merge {EXPECTED_MLP} tensors, merged {merged}")
    if len(out) != EXPECTED_TOTAL:
        fail(f"output has {len(out)} tensors, expected {EXPECTED_TOTAL}")
    if set(out) != names:
        fail("output tensor names differ from input names")
    print(f"merged {merged} MLP tensors with lambda={LAMBDA}")

    # --- Step 3: write and re-verify ---
    os.makedirs(HERE, exist_ok=True)
    save_file(out, OUT_PATH, metadata={"format": "pt"})
    with safe_open(OUT_PATH, framework="pt", device="cpu") as f:
        written = set(f.keys())
    if len(written) != EXPECTED_TOTAL or written != names:
        fail(f"written file has {len(written)} tensors, expected {EXPECTED_TOTAL}")
    print(f"wrote {OUT_PATH} with {len(written)} tensors")


if __name__ == "__main__":
    main()
