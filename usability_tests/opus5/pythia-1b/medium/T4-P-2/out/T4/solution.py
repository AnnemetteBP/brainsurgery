"""T4: task-vector merge of two Pythia-1B fine-tunes onto the base."""

import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

BASE = Path("inputs/base/model.safetensors")
FT1 = Path("inputs/ft1/model.safetensors")
FT2 = Path("inputs/ft2/model.safetensors")
OUT = Path("out/T4/model.safetensors")

LAMBDA = 0.4
N_LAYERS = 16
N_TOTAL = 244
N_MLP = 64


def mlp_names() -> list[str]:
    names = []
    for i in range(N_LAYERS):
        for proj in ("dense_h_to_4h", "dense_4h_to_h"):
            for suffix in ("weight", "bias"):
                names.append(f"gpt_neox.layers.{i}.mlp.{proj}.{suffix}")
    return names


def fail(msg: str) -> None:
    raise SystemExit(f"ERROR: {msg}")


def main() -> None:
    with safe_open(BASE, framework="pt") as fb, \
         safe_open(FT1, framework="pt") as f1, \
         safe_open(FT2, framework="pt") as f2:

        kb, k1, k2 = set(fb.keys()), set(f1.keys()), set(f2.keys())

        # Step 1a: identical key sets across the three checkpoints.
        if kb != k1:
            fail(f"base/ft1 key sets differ: only-base={sorted(kb - k1)[:5]} "
                 f"only-ft1={sorted(k1 - kb)[:5]}")
        if kb != k2:
            fail(f"base/ft2 key sets differ: only-base={sorted(kb - k2)[:5]} "
                 f"only-ft2={sorted(k2 - kb)[:5]}")
        if len(kb) != N_TOTAL:
            fail(f"expected {N_TOTAL} tensors in the base, found {len(kb)}")

        merge_keys = mlp_names()
        if len(merge_keys) != N_MLP:
            fail(f"MLP name list has {len(merge_keys)} entries, expected {N_MLP}")
        missing = [k for k in merge_keys if k not in kb]
        if missing:
            fail(f"MLP tensors missing from the checkpoints: {missing}")
        merge_set = set(merge_keys)

        # Step 1b: every non-MLP tensor is bit-identical in all three.
        mismatched = []
        for name in sorted(kb - merge_set):
            tb = fb.get_tensor(name)
            t1 = f1.get_tensor(name)
            t2 = f2.get_tensor(name)
            for other, tag in ((t1, "ft1"), (t2, "ft2")):
                if other.shape != tb.shape or other.dtype != tb.dtype:
                    mismatched.append(f"{name} ({tag}: shape/dtype differs from base)")
                elif not torch.equal(other, tb):
                    mismatched.append(f"{name} ({tag}: values differ from base)")
        if mismatched:
            fail(f"{len(mismatched)} shared tensor(s) differ outside the MLP set; "
                 f"first: {mismatched[:5]}")

        # Step 2 / 3: merge the MLP tensors, copy everything else from the base.
        out: dict[str, torch.Tensor] = {}
        merged = 0
        for name in fb.keys():
            tb = fb.get_tensor(name)
            if name in merge_set:
                t1 = f1.get_tensor(name)
                t2 = f2.get_tensor(name)
                if t1.shape != tb.shape or t2.shape != tb.shape:
                    fail(f"{name}: MLP shape mismatch across checkpoints")
                if t1.dtype != tb.dtype or t2.dtype != tb.dtype:
                    fail(f"{name}: MLP dtype mismatch across checkpoints")
                b32 = tb.to(torch.float32)
                # Both task vectors are taken against the unmodified base.
                acc = b32 + LAMBDA * (t1.to(torch.float32) - b32) \
                          + LAMBDA * (t2.to(torch.float32) - b32)
                out[name] = acc.to(tb.dtype).contiguous()
                merged += 1
            else:
                out[name] = tb.clone().contiguous()

        metadata = fb.metadata() or {"format": "pt"}

    # Required checks.
    if merged != N_MLP:
        fail(f"merged {merged} tensors, expected exactly {N_MLP}")
    if len(out) != N_TOTAL:
        fail(f"output has {len(out)} tensors, expected exactly {N_TOTAL}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    save_file(out, str(OUT), metadata=metadata)

    with safe_open(OUT, framework="pt") as fo:
        if len(fo.keys()) != N_TOTAL:
            fail(f"written file has {len(fo.keys())} tensors, expected {N_TOTAL}")

    print(f"OK: verified {N_TOTAL - N_MLP} shared tensors, merged {merged} MLP "
          f"tensors with lambda={LAMBDA}, wrote {OUT} ({len(out)} tensors)")


if __name__ == "__main__":
    sys.exit(main())
