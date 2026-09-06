"""T4: task-vector merge of two GPT-2 fine-tunes onto the base.

out[X] = base[X] + lambda*(ft1[X]-base[X]) + lambda*(ft2[X]-base[X]) for the
48 MLP tensors; every other tensor is copied from the base verbatim.
Each task vector is taken against the *unmodified* base.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

LAMBDA = 0.4
N_LAYERS = 12
N_MLP = 48
N_TOTAL = 160

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent  # sandbox root
IN = ROOT / "inputs"
OUT = HERE / "model.safetensors"

MLP_KEYS = [
    f"h.{i}.mlp.{mod}.{suf}"
    for i in range(N_LAYERS)
    for mod in ("c_fc", "c_proj")
    for suf in ("weight", "bias")
]


class CheckFailed(RuntimeError):
    pass


def require(cond: bool, msg: str) -> None:
    if not cond:
        raise CheckFailed(msg)


def main() -> None:
    base = load_file(IN / "base" / "model.safetensors")
    ft1 = load_file(IN / "ft1" / "model.safetensors")
    ft2 = load_file(IN / "ft2" / "model.safetensors")

    # --- step 1: verification, before touching anything -------------------
    require(
        set(base) == set(ft1) == set(ft2),
        "tensor name sets differ between base/ft1/ft2: "
        f"ft1 only={sorted(set(ft1) ^ set(base))[:5]} "
        f"ft2 only={sorted(set(ft2) ^ set(base))[:5]}",
    )
    require(
        len(base) == N_TOTAL,
        f"expected {N_TOTAL} tensors in the base, found {len(base)}",
    )

    missing = [k for k in MLP_KEYS if k not in base]
    require(not missing, f"expected MLP tensors are absent: {missing}")
    require(len(set(MLP_KEYS)) == N_MLP, f"MLP key list is not {N_MLP} names")

    for k in sorted(base):
        for name, other in (("ft1", ft1), ("ft2", ft2)):
            require(
                base[k].shape == other[k].shape and base[k].dtype == other[k].dtype,
                f"{k}: shape/dtype mismatch base {tuple(base[k].shape)}/{base[k].dtype} "
                f"vs {name} {tuple(other[k].shape)}/{other[k].dtype}",
            )

    shared = [k for k in sorted(base) if k not in set(MLP_KEYS)]
    require(
        len(shared) == N_TOTAL - N_MLP,
        f"expected {N_TOTAL - N_MLP} non-MLP tensors, found {len(shared)}",
    )
    for k in shared:
        for name, other in (("ft1", ft1), ("ft2", ft2)):
            require(
                torch.equal(base[k], other[k]),
                f"non-MLP tensor {k} differs between base and {name}; "
                "the frozen-backbone assumption does not hold",
            )

    # --- step 2/3: merge ---------------------------------------------------
    out: dict[str, torch.Tensor] = {}
    merged = 0
    for k in base:
        if k in set(MLP_KEYS):
            b = base[k].to(torch.float32)
            v1 = ft1[k].to(torch.float32) - b
            v2 = ft2[k].to(torch.float32) - b
            out[k] = (b + LAMBDA * v1 + LAMBDA * v2).to(base[k].dtype).contiguous()
            merged += 1
        else:
            out[k] = base[k].clone().contiguous()

    # --- required checks on the result -------------------------------------
    require(merged == N_MLP, f"merged {merged} tensors, expected {N_MLP}")
    require(len(out) == N_TOTAL, f"output has {len(out)} tensors, expected {N_TOTAL}")
    require(set(out) == set(base), "output key set drifted from the base")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    save_file(out, str(OUT))

    # --- post-write verification ------------------------------------------
    back = load_file(OUT)
    require(len(back) == N_TOTAL, f"written file has {len(back)} tensors, expected {N_TOTAL}")
    require(set(back) == set(base), "written file key set differs from the base")
    changed = 0
    for k in back:
        require(back[k].shape == base[k].shape, f"{k}: shape changed on write")
        require(back[k].dtype == base[k].dtype, f"{k}: dtype changed on write")
        if not torch.equal(back[k], base[k]):
            changed += 1
        if k not in set(MLP_KEYS):
            require(torch.equal(back[k], base[k]), f"non-MLP tensor {k} was modified")
    require(changed <= N_MLP, f"{changed} tensors changed, expected at most {N_MLP}")

    print(f"OK: verified {len(shared)} shared tensors, merged {merged}, wrote {len(back)} to {OUT}")
    print(f"    lambda={LAMBDA}, tensors actually differing from base: {changed}")


if __name__ == "__main__":
    try:
        main()
    except CheckFailed as exc:
        print(f"FAILED CHECK: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
