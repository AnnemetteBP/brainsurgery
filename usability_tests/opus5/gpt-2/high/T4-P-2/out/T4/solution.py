#!/usr/bin/env python
"""T4: task-vector merge of two GPT-2 fine-tunes.

out[X] = base[X] + lambda*(ft1[X] - base[X]) + lambda*(ft2[X] - base[X])
for the 48 MLP tensors; every other tensor is copied from the base verbatim.
Both task vectors are taken against the *unmodified* base.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent  # sandbox root
BASE = ROOT / "inputs" / "base" / "model.safetensors"
FT1 = ROOT / "inputs" / "ft1" / "model.safetensors"
FT2 = ROOT / "inputs" / "ft2" / "model.safetensors"
OUT_DIR = ROOT / "out" / "T4"
OUT = OUT_DIR / "model.safetensors"

LAMBDA = 0.4
N_TOTAL = 160
N_MLP = 48

# h.<i>.mlp.c_fc.{weight,bias} / h.<i>.mlp.c_proj.{weight,bias}; tolerate a
# leading module prefix such as "transformer." if the checkpoint carries one.
MLP_RE = re.compile(r"(?:^|\.)h\.(\d+)\.mlp\.c_(?:fc|proj)\.(?:weight|bias)$")


class CheckFailed(RuntimeError):
    pass


def check(cond: bool, msg: str) -> None:
    if not cond:
        raise CheckFailed(msg)


def main() -> int:
    for p in (BASE, FT1, FT2):
        check(p.is_file(), f"missing input checkpoint: {p}")

    with safe_open(BASE, framework="pt") as fb, \
         safe_open(FT1, framework="pt") as f1, \
         safe_open(FT2, framework="pt") as f2:

        keys_base = list(fb.keys())
        kb, k1, k2 = set(keys_base), set(f1.keys()), set(f2.keys())

        # --- step 1a: identical key sets across the three checkpoints -------
        check(
            kb == k1,
            "base and ft1 key sets differ: "
            f"only-in-base={sorted(kb - k1)[:5]} only-in-ft1={sorted(k1 - kb)[:5]}",
        )
        check(
            kb == k2,
            "base and ft2 key sets differ: "
            f"only-in-base={sorted(kb - k2)[:5]} only-in-ft2={sorted(k2 - kb)[:5]}",
        )
        check(
            len(keys_base) == N_TOTAL,
            f"expected {N_TOTAL} tensors in each checkpoint, base has {len(keys_base)}",
        )

        mlp_keys = sorted(k for k in keys_base if MLP_RE.search(k))
        check(
            len(mlp_keys) == N_MLP,
            f"expected {N_MLP} MLP tensors, matched {len(mlp_keys)}: {mlp_keys}",
        )
        layers = sorted({int(MLP_RE.search(k).group(1)) for k in mlp_keys})
        check(
            layers == list(range(12)),
            f"MLP tensors do not cover layers 0..11, got layers {layers}",
        )

        mlp_set = set(mlp_keys)
        out: dict[str, torch.Tensor] = {}
        n_merged = 0
        n_copied = 0

        for key in keys_base:
            tb = fb.get_tensor(key)
            t1 = f1.get_tensor(key)
            t2 = f2.get_tensor(key)

            check(
                tb.shape == t1.shape == t2.shape,
                f"shape mismatch for {key}: base={tuple(tb.shape)} "
                f"ft1={tuple(t1.shape)} ft2={tuple(t2.shape)}",
            )
            check(
                tb.dtype == t1.dtype == t2.dtype,
                f"dtype mismatch for {key}: base={tb.dtype} ft1={t1.dtype} ft2={t2.dtype}",
            )

            if key in mlp_set:
                # --- step 2: merge, both task vectors against the pristine base
                check(
                    tb.dtype == torch.float32,
                    f"MLP tensor {key} is {tb.dtype}, expected float32",
                )
                merged = tb + LAMBDA * (t1 - tb) + LAMBDA * (t2 - tb)
                check(
                    merged.dtype == torch.float32,
                    f"merged {key} has dtype {merged.dtype}, expected float32",
                )
                check(
                    torch.isfinite(merged).all().item(),
                    f"merged {key} contains non-finite values",
                )
                out[key] = merged.contiguous()
                n_merged += 1
            else:
                # --- step 1b: shared tensors must be identical in all three --
                check(
                    torch.equal(tb, t1),
                    f"non-MLP tensor {key} differs between base and ft1 "
                    "(backbone is not frozen); aborting",
                )
                check(
                    torch.equal(tb, t2),
                    f"non-MLP tensor {key} differs between base and ft2 "
                    "(backbone is not frozen); aborting",
                )
                # --- step 3: taken from the base unchanged -------------------
                out[key] = tb.clone().contiguous()
                n_copied += 1

    # --- required checks -----------------------------------------------------
    check(n_merged == N_MLP, f"merged {n_merged} tensors, expected exactly {N_MLP}")
    check(
        n_copied == N_TOTAL - N_MLP,
        f"copied {n_copied} tensors, expected exactly {N_TOTAL - N_MLP}",
    )
    check(len(out) == N_TOTAL, f"output holds {len(out)} tensors, expected {N_TOTAL}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    save_file(out, str(OUT), metadata={"format": "pt"})

    # --- verify what actually landed on disk ---------------------------------
    with safe_open(OUT, framework="pt") as fo:
        written = list(fo.keys())
        check(
            len(written) == N_TOTAL,
            f"written file has {len(written)} tensors, expected {N_TOTAL}",
        )
        check(set(written) == set(out), "written key set differs from the intended one")
        with safe_open(BASE, framework="pt") as fb:
            for key in written:
                tw = fo.get_tensor(key)
                tb = fb.get_tensor(key)
                check(
                    tw.shape == tb.shape and tw.dtype == tb.dtype,
                    f"written {key}: shape/dtype drifted from base "
                    f"({tuple(tw.shape)}/{tw.dtype} vs {tuple(tb.shape)}/{tb.dtype})",
                )
                if key not in mlp_set:
                    check(torch.equal(tw, tb), f"written {key} is not bit-identical to base")

    print(f"merged  : {n_merged} MLP tensors (lambda={LAMBDA})")
    print(f"copied  : {n_copied} tensors unchanged from base")
    print(f"written : {OUT} ({len(written)} tensors)")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except CheckFailed as exc:
        print(f"CHECK FAILED: {exc}", file=sys.stderr)
        sys.exit(1)
