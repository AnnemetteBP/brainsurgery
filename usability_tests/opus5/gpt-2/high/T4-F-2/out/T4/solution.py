#!/usr/bin/env python
"""T4: task-vector merge of two GPT-2 fine-tunes onto the base.

    out[X] = base[X] + lambda*(ft1[X]-base[X]) + lambda*(ft2[X]-base[X])

for the 48 MLP tensors only; every other tensor is copied bit-exactly from
the base. All required checks are hard assertions: the script exits non-zero
and writes nothing if any of them fails.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

LAMBDA = 0.4
N_LAYERS = 12
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent  # sandbox root
INPUTS = ROOT / "inputs"
OUT = HERE / "model.safetensors"

MLP_SUFFIXES = ("mlp.c_fc.weight", "mlp.c_fc.bias", "mlp.c_proj.weight", "mlp.c_proj.bias")
MLP_KEYS = [f"h.{i}.{s}" for i in range(N_LAYERS) for s in MLP_SUFFIXES]
EXPECTED_SHAPES = {
    "mlp.c_fc.weight": (768, 3072),
    "mlp.c_fc.bias": (3072,),
    "mlp.c_proj.weight": (3072, 768),
    "mlp.c_proj.bias": (768,),
}


class CheckFailed(RuntimeError):
    pass


def check(cond: bool, msg: str) -> None:
    if not cond:
        raise CheckFailed(msg)


def main() -> None:
    base = load_file(INPUTS / "base" / "model.safetensors")
    ft1 = load_file(INPUTS / "ft1" / "model.safetensors")
    ft2 = load_file(INPUTS / "ft2" / "model.safetensors")

    # --- step 1: identical tensor names across the three checkpoints -------
    for name, sd in (("ft1", ft1), ("ft2", ft2)):
        missing = sorted(set(base) - set(sd))
        extra = sorted(set(sd) - set(base))
        check(not missing and not extra,
              f"{name} key set differs from base: missing={missing[:5]} extra={extra[:5]}")
    check(len(base) == 160, f"base has {len(base)} tensors, expected 160")

    # the 48 MLP tensors must exist, with the documented shapes and dtypes
    for key in MLP_KEYS:
        check(key in base, f"expected MLP tensor {key} missing from base")
        want = EXPECTED_SHAPES[key.split(".", 2)[2]]
        for name, sd in (("base", base), ("ft1", ft1), ("ft2", ft2)):
            check(tuple(sd[key].shape) == want,
                  f"{name}[{key}] has shape {tuple(sd[key].shape)}, expected {want}")
            check(sd[key].dtype == torch.float32,
                  f"{name}[{key}] has dtype {sd[key].dtype}, expected float32")
    mlp_keys = set(MLP_KEYS)
    check(len(mlp_keys) == 48, f"built {len(mlp_keys)} MLP keys, expected 48")

    # --- step 1 (cont.): every non-MLP tensor identical in all three ------
    shared = [k for k in base if k not in mlp_keys]
    check(len(shared) == 112, f"{len(shared)} non-MLP tensors, expected 112")
    for key in shared:
        for name, sd in (("ft1", ft1), ("ft2", ft2)):
            check(sd[key].shape == base[key].shape and sd[key].dtype == base[key].dtype,
                  f"{name}[{key}] shape/dtype differs from base "
                  f"({tuple(sd[key].shape)}/{sd[key].dtype} vs "
                  f"{tuple(base[key].shape)}/{base[key].dtype})")
            check(torch.equal(sd[key], base[key]),
                  f"frozen-backbone assumption violated: {name}[{key}] differs from base")

    # --- step 2: task-vector merge, both vectors against the *original* base
    out: dict[str, torch.Tensor] = {}
    merged = 0
    for key in base:
        if key in mlp_keys:
            b = base[key].to(torch.float32)
            tv1 = ft1[key].to(torch.float32) - b
            tv2 = ft2[key].to(torch.float32) - b
            merged_tensor = b + LAMBDA * tv1 + LAMBDA * tv2
            check(merged_tensor.dtype == torch.float32, f"{key} merged as {merged_tensor.dtype}")
            out[key] = merged_tensor.contiguous()
            merged += 1
        else:
            # step 3: unchanged tensors come from the base, bit-exact
            out[key] = base[key].clone().contiguous()

    check(merged == 48, f"merged {merged} tensors, expected exactly 48")
    check(len(out) == 160, f"output has {len(out)} tensors, expected exactly 160")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    save_file(out, OUT, metadata={"format": "pt"})

    # --- post-write verification of the file actually on disk -------------
    back = load_file(OUT)
    check(len(back) == 160, f"written file has {len(back)} tensors, expected 160")
    check(set(back) == set(base), "written file key set differs from base")
    n_changed = 0
    for key in base:
        check(back[key].shape == base[key].shape and back[key].dtype == base[key].dtype,
              f"written {key} shape/dtype drifted")
        if key in mlp_keys:
            expect = base[key] + LAMBDA * (ft1[key] - base[key]) + LAMBDA * (ft2[key] - base[key])
            check(torch.allclose(back[key], expect, rtol=0, atol=0),
                  f"written {key} does not match the merge formula")
            n_changed += 1
        else:
            check(torch.equal(back[key], base[key]),
                  f"written {key} is not bit-identical to the base")
    check(n_changed == 48, f"verified {n_changed} merged tensors, expected 48")

    print(f"OK: wrote {OUT} with {len(back)} tensors "
          f"({n_changed} merged at lambda={LAMBDA}, {len(back) - n_changed} copied from base)")


if __name__ == "__main__":
    try:
        main()
    except CheckFailed as exc:
        print(f"CHECK FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1)
