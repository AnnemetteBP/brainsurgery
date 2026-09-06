"""T4: task-vector merge of two Pythia-1B fine-tunes.

out[X] = base[X] + lam*(ft1[X] - base[X]) + lam*(ft2[X] - base[X])  for the 64 MLP
tensors, computed in float32 against the *unmodified* base and cast back to float16;
every other tensor is copied bit-exactly from the base.

Plain torch + safetensors: the task's precondition (everything outside the 64 MLP
tensors is bit-identical across the three checkpoints) has to be checked and has to
abort the run, which a mergekit task_arithmetic config cannot express.
"""

import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

LAMBDA = 0.4
N_LAYERS = 16
N_TENSORS = 244
N_MLP = 64

BASE = Path("inputs/base/model.safetensors")
FT1 = Path("inputs/ft1/model.safetensors")
FT2 = Path("inputs/ft2/model.safetensors")
OUT = Path("out/T4/model.safetensors")


class CheckFailed(Exception):
    """A required check did not hold; the run must abort."""


def require(cond, msg):
    if not cond:
        raise CheckFailed(msg)


def bits(t):
    """Bit pattern of a tensor, so equality is exact even for NaN / -0.0."""
    return t.contiguous().reshape(-1).view(torch.uint8)


def mlp_names():
    names = set()
    for i in range(N_LAYERS):
        for proj in ("dense_h_to_4h", "dense_4h_to_h"):
            for part in ("weight", "bias"):
                names.add(f"gpt_neox.layers.{i}.mlp.{proj}.{part}")
    return names


def main():
    targets = mlp_names()
    require(len(targets) == N_MLP, f"expected {N_MLP} MLP names, built {len(targets)}")

    with (
        safe_open(BASE, framework="pt") as base,
        safe_open(FT1, framework="pt") as ft1,
        safe_open(FT2, framework="pt") as ft2,
    ):
        kb, k1, k2 = set(base.keys()), set(ft1.keys()), set(ft2.keys())

        # --- step 1: same names in all three, and the 64 MLP tensors are among them ---
        require(kb == k1, f"base vs ft1 key mismatch: {sorted(kb ^ k1)[:5]}")
        require(kb == k2, f"base vs ft2 key mismatch: {sorted(kb ^ k2)[:5]}")
        require(len(kb) == N_TENSORS, f"expected {N_TENSORS} tensors, base has {len(kb)}")
        missing = targets - kb
        require(not missing, f"MLP tensors absent from the checkpoints: {sorted(missing)[:5]}")

        # --- step 1 (cont.): every tensor outside the 64 is identical in all three ---
        shared = sorted(kb - targets)
        require(
            len(shared) == N_TENSORS - N_MLP,
            f"expected {N_TENSORS - N_MLP} shared tensors, found {len(shared)}",
        )
        differing = []
        for name in shared:
            b = base.get_tensor(name)
            for tag, f in (("ft1", ft1), ("ft2", ft2)):
                o = f.get_tensor(name)
                if o.shape != b.shape or o.dtype != b.dtype:
                    differing.append(f"{name}: {tag} is {tuple(o.shape)}/{o.dtype}, "
                                     f"base is {tuple(b.shape)}/{b.dtype}")
                elif not torch.equal(bits(b), bits(o)):
                    differing.append(f"{name}: differs from {tag}")
        require(
            not differing,
            "frozen-backbone precondition violated, "
            f"{len(differing)} non-MLP tensor(s) differ:\n  " + "\n  ".join(differing[:10]),
        )
        print(f"[check] {len(shared)} non-MLP tensors identical across base/ft1/ft2")

        # --- step 2/3: merge the 64, copy the rest from the base ---
        merged_names = []
        out = {}
        for name in sorted(kb):
            b = base.get_tensor(name)
            if name not in targets:
                out[name] = b.clone()
                continue
            t1, t2 = ft1.get_tensor(name), ft2.get_tensor(name)
            for tag, o in (("ft1", t1), ("ft2", t2)):
                require(
                    o.shape == b.shape and o.dtype == b.dtype,
                    f"{name}: {tag} is {tuple(o.shape)}/{o.dtype}, "
                    f"base is {tuple(b.shape)}/{b.dtype}",
                )
            b32 = b.to(torch.float32)
            # both task vectors are taken against the unmodified base
            merged = b32 + LAMBDA * (t1.to(torch.float32) - b32) \
                         + LAMBDA * (t2.to(torch.float32) - b32)
            out[name] = merged.to(b.dtype)
            merged_names.append(name)

        require(
            len(merged_names) == N_MLP,
            f"merged {len(merged_names)} tensors, expected exactly {N_MLP}",
        )
        require(len(out) == N_TENSORS, f"built {len(out)} tensors, expected {N_TENSORS}")
        print(f"[check] merged exactly {len(merged_names)} MLP tensors with lambda={LAMBDA}")

        OUT.parent.mkdir(parents=True, exist_ok=True)
        save_file(out, str(OUT))

    # --- verify what actually landed on disk ---
    with safe_open(BASE, framework="pt") as base, safe_open(OUT, framework="pt") as res:
        kr = set(res.keys())
        require(len(kr) == N_TENSORS, f"output has {len(kr)} tensors, expected {N_TENSORS}")
        require(kr == set(base.keys()), "output key set differs from the base")
        changed = []
        for name in sorted(kr):
            b, r = base.get_tensor(name), res.get_tensor(name)
            require(
                r.shape == b.shape and r.dtype == b.dtype,
                f"{name}: output is {tuple(r.shape)}/{r.dtype}, base is "
                f"{tuple(b.shape)}/{b.dtype}",
            )
            if not torch.equal(bits(b), bits(r)):
                changed.append(name)
        require(
            set(changed) <= targets,
            f"output changed non-MLP tensors: {sorted(set(changed) - targets)[:5]}",
        )
        print(f"[check] output has {len(kr)} tensors; {len(changed)} differ from the base "
              f"(all MLP); {N_TENSORS - len(changed)} bit-identical")
    print(f"[ok] wrote {OUT}")


if __name__ == "__main__":
    try:
        main()
    except CheckFailed as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        sys.exit(1)
