"""T4: task-vector merge of two OLMo-1B fine-tunes into the base."""

import json
import os

import torch
from safetensors import safe_open
from safetensors.torch import save_file

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
BASE_DIR = os.path.join(ROOT, "inputs", "base")
FT1 = os.path.join(ROOT, "inputs", "ft1", "model.safetensors")
FT2 = os.path.join(ROOT, "inputs", "ft2", "model.safetensors")
OUT = os.path.join(ROOT, "out", "T4", "model.safetensors")

LAMBDA = 0.4
N_LAYERS = 16
MLP_NAMES = {
    f"model.layers.{i}.mlp.{proj}.weight"
    for i in range(N_LAYERS)
    for proj in ("gate_proj", "up_proj", "down_proj")
}


def load_sharded(directory):
    index_path = os.path.join(directory, "model.safetensors.index.json")
    with open(index_path) as fh:
        weight_map = json.load(fh)["weight_map"]
    tensors = {}
    for shard in sorted(set(weight_map.values())):
        with safe_open(os.path.join(directory, shard), framework="pt") as fh:
            for name in fh.keys():
                tensors[name] = fh.get_tensor(name)
    missing = set(weight_map) - set(tensors)
    if missing:
        raise RuntimeError(f"index lists tensors absent from shards: {sorted(missing)}")
    return tensors


def load_single(path):
    with safe_open(path, framework="pt") as fh:
        return {name: fh.get_tensor(name) for name in fh.keys()}


def main():
    base = load_sharded(BASE_DIR)
    ft1 = load_single(FT1)
    ft2 = load_single(FT2)

    # Step 1: identical key sets across the three checkpoints.
    for label, ckpt in (("ft1", ft1), ("ft2", ft2)):
        if set(ckpt) != set(base):
            only_base = sorted(set(base) - set(ckpt))
            only_ckpt = sorted(set(ckpt) - set(base))
            raise RuntimeError(
                f"{label} key set differs from base; "
                f"missing={only_base[:5]} extra={only_ckpt[:5]}"
            )

    if len(base) != 114:
        raise RuntimeError(f"expected 114 tensors in the base, found {len(base)}")

    absent = MLP_NAMES - set(base)
    if absent:
        raise RuntimeError(f"expected MLP tensors absent from the base: {sorted(absent)}")

    # Step 1 (cont.): every non-MLP tensor must be bit-identical in all three.
    for name in sorted(set(base) - MLP_NAMES):
        b = base[name]
        for label, ckpt in (("ft1", ft1), ("ft2", ft2)):
            t = ckpt[name]
            if t.shape != b.shape or t.dtype != b.dtype:
                raise RuntimeError(
                    f"non-MLP tensor {name} differs in shape/dtype between base and {label}: "
                    f"{tuple(b.shape)}/{b.dtype} vs {tuple(t.shape)}/{t.dtype}"
                )
            if not torch.equal(t, b):
                raise RuntimeError(
                    f"non-MLP tensor {name} is not identical in base and {label}; "
                    "the frozen-backbone precondition does not hold"
                )

    # Step 2: merge, always against the unmodified base.
    out = dict(base)
    merged = 0
    for name in sorted(MLP_NAMES):
        b = base[name]
        t1 = ft1[name]
        t2 = ft2[name]
        for label, t in (("ft1", t1), ("ft2", t2)):
            if t.shape != b.shape or t.dtype != b.dtype:
                raise RuntimeError(
                    f"MLP tensor {name} differs in shape/dtype between base and {label}: "
                    f"{tuple(b.shape)}/{b.dtype} vs {tuple(t.shape)}/{t.dtype}"
                )
        b32 = b.to(torch.float32)
        merged_tensor = b32 + LAMBDA * (t1.to(torch.float32) - b32) + LAMBDA * (
            t2.to(torch.float32) - b32
        )
        out[name] = merged_tensor.to(b.dtype).contiguous()
        merged += 1

    if merged != 48:
        raise RuntimeError(f"expected to merge exactly 48 tensors, merged {merged}")
    if len(out) != 114:
        raise RuntimeError(f"expected 114 tensors in the output, have {len(out)}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    save_file({k: v.contiguous() for k, v in out.items()}, OUT)

    with safe_open(OUT, framework="pt") as fh:
        written = list(fh.keys())
    if len(written) != 114:
        raise RuntimeError(f"output file has {len(written)} tensors, expected 114")
    if set(written) != set(base):
        raise RuntimeError("output key set differs from the base key set")

    print(f"merged {merged} MLP tensors (lambda={LAMBDA}); wrote {len(written)} tensors to {OUT}")


if __name__ == "__main__":
    main()
