"""Task-vector merge of two fine-tunes into the OLMo-1B base checkpoint."""

import json
import re
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

HERE = Path(__file__).resolve().parent
SANDBOX = HERE.parent.parent
BASE_DIR = SANDBOX / "inputs" / "base"
FT1_PATH = SANDBOX / "inputs" / "ft1" / "model.safetensors"
FT2_PATH = SANDBOX / "inputs" / "ft2" / "model.safetensors"
OUT_PATH = HERE / "model.safetensors"

LAMBDA = 0.4
MLP_RE = re.compile(r"^model\.layers\.(\d+)\.mlp\.(gate_proj|up_proj|down_proj)\.weight$")


def load_base():
    index_path = BASE_DIR / "model.safetensors.index.json"
    with open(index_path) as f:
        index = json.load(f)
    weight_map = index["weight_map"]
    tensors = {}
    shard_names = sorted(set(weight_map.values()))
    for shard_name in shard_names:
        shard_path = BASE_DIR / shard_name
        with safe_open(str(shard_path), framework="pt") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)
    if set(tensors.keys()) != set(weight_map.keys()):
        raise RuntimeError("base shard contents do not match index.json weight_map")
    return tensors


def load_flat(path):
    tensors = {}
    with safe_open(str(path), framework="pt") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    return tensors


def main():
    base = load_base()
    ft1 = load_flat(FT1_PATH)
    ft2 = load_flat(FT2_PATH)

    base_keys = set(base.keys())
    ft1_keys = set(ft1.keys())
    ft2_keys = set(ft2.keys())

    if not (base_keys == ft1_keys == ft2_keys):
        only_base = base_keys - ft1_keys - ft2_keys
        only_ft1 = ft1_keys - base_keys
        only_ft2 = ft2_keys - base_keys
        raise RuntimeError(
            "Tensor name sets differ between checkpoints: "
            f"base-only(subset)={only_base}, ft1-only={only_ft1}, ft2-only={only_ft2}"
        )

    if len(base_keys) != 114:
        raise RuntimeError(f"Expected 114 tensors in base, found {len(base_keys)}")

    mlp_keys = {k for k in base_keys if MLP_RE.match(k)}
    if len(mlp_keys) != 48:
        raise RuntimeError(f"Expected 48 MLP tensors, found {len(mlp_keys)}: {sorted(mlp_keys)}")

    shared_keys = base_keys - mlp_keys

    for key in shared_keys:
        b, f1, f2 = base[key], ft1[key], ft2[key]
        if b.shape != f1.shape or b.shape != f2.shape:
            raise RuntimeError(f"Shape mismatch for shared tensor {key}")
        if b.dtype != f1.dtype or b.dtype != f2.dtype:
            raise RuntimeError(f"Dtype mismatch for shared tensor {key}")
        if not torch.equal(b, f1):
            raise RuntimeError(f"Shared tensor {key} differs between base and ft1")
        if not torch.equal(b, f2):
            raise RuntimeError(f"Shared tensor {key} differs between base and ft2")

    out = {}
    for key in shared_keys:
        out[key] = base[key].clone()

    merged_count = 0
    for key in mlp_keys:
        b = base[key].to(torch.float32)
        f1 = ft1[key].to(torch.float32)
        f2 = ft2[key].to(torch.float32)
        if b.dtype != torch.float32 or f1.dtype != torch.float32 or f2.dtype != torch.float32:
            raise RuntimeError(f"Expected float32 tensors for {key}")
        merged = b + LAMBDA * (f1 - b) + LAMBDA * (f2 - b)
        out[key] = merged
        merged_count += 1

    if merged_count != 48:
        raise RuntimeError(f"Expected to merge 48 tensors, merged {merged_count}")

    if len(out) != 114:
        raise RuntimeError(f"Expected 114 output tensors, got {len(out)}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_file(out, str(OUT_PATH))
    print(f"Wrote {len(out)} tensors ({merged_count} merged) to {OUT_PATH}")


if __name__ == "__main__":
    main()
