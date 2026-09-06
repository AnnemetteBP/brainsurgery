"""T4: task-vector merge of two frozen-backbone fine-tunes into a Pythia-1B base.

out[X] = base[X] + lam*(ft1[X]-base[X]) + lam*(ft2[X]-base[X])  for the 64 MLP tensors,
computed in float32 against the *unmodified* base, cast back to float16.
Everything else is copied bit-exactly from the base.
"""
import os
import re
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
BASE = os.path.join(ROOT, "inputs", "base", "model.safetensors")
FT1 = os.path.join(ROOT, "inputs", "ft1", "model.safetensors")
FT2 = os.path.join(ROOT, "inputs", "ft2", "model.safetensors")
OUT = os.path.join(ROOT, "out", "T4", "model.safetensors")

LAM = 0.4
N_LAYERS = 16
EXPECTED_TOTAL = 244
EXPECTED_MERGED = 64

MLP_RE = re.compile(
    r"^gpt_neox\.layers\.(\d+)\.mlp\.(dense_h_to_4h|dense_4h_to_h)\.(weight|bias)$"
)


def fail(msg):
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def main():
    base_f = safe_open(BASE, framework="pt", device="cpu")
    ft1_f = safe_open(FT1, framework="pt", device="cpu")
    ft2_f = safe_open(FT2, framework="pt", device="cpu")

    base_keys = set(base_f.keys())
    ft1_keys = set(ft1_f.keys())
    ft2_keys = set(ft2_f.keys())

    # --- Step 1: verification before touching anything -----------------------
    if len(base_keys) != EXPECTED_TOTAL:
        fail(f"base has {len(base_keys)} tensors, expected {EXPECTED_TOTAL}")
    if base_keys != ft1_keys:
        fail(f"ft1 key set differs from base: only-base={sorted(base_keys - ft1_keys)[:5]} "
             f"only-ft1={sorted(ft1_keys - base_keys)[:5]}")
    if base_keys != ft2_keys:
        fail(f"ft2 key set differs from base: only-base={sorted(base_keys - ft2_keys)[:5]} "
             f"only-ft2={sorted(ft2_keys - base_keys)[:5]}")

    mlp_keys = set()
    for k in base_keys:
        m = MLP_RE.match(k)
        if m and 0 <= int(m.group(1)) < N_LAYERS:
            mlp_keys.add(k)
    if len(mlp_keys) != EXPECTED_MERGED:
        fail(f"found {len(mlp_keys)} MLP tensors, expected {EXPECTED_MERGED}")

    shared_keys = base_keys - mlp_keys
    for k in sorted(shared_keys):
        b = base_f.get_tensor(k)
        t1 = ft1_f.get_tensor(k)
        t2 = ft2_f.get_tensor(k)
        for name, t in (("ft1", t1), ("ft2", t2)):
            if t.shape != b.shape or t.dtype != b.dtype:
                fail(f"shared tensor {k}: {name} shape/dtype {tuple(t.shape)}/{t.dtype} "
                     f"!= base {tuple(b.shape)}/{b.dtype}")
            # bit-exact comparison (torch.equal treats NaN != NaN, so compare raw bits)
            if not torch.equal(t.view(torch.int16) if t.dtype == torch.float16 else t,
                               b.view(torch.int16) if b.dtype == torch.float16 else b):
                fail(f"shared tensor {k} differs between base and {name}")
    print(f"verified {len(shared_keys)} shared tensors identical across base/ft1/ft2")

    # --- Step 2: merge the MLP tensors against the unmodified base ----------
    out = {}
    merged = 0
    for k in sorted(base_keys):
        b = base_f.get_tensor(k)
        if k in mlp_keys:
            t1 = ft1_f.get_tensor(k)
            t2 = ft2_f.get_tensor(k)
            if t1.shape != b.shape or t2.shape != b.shape:
                fail(f"MLP tensor {k}: shape mismatch base={tuple(b.shape)} "
                     f"ft1={tuple(t1.shape)} ft2={tuple(t2.shape)}")
            if t1.dtype != b.dtype or t2.dtype != b.dtype:
                fail(f"MLP tensor {k}: dtype mismatch base={b.dtype} ft1={t1.dtype} ft2={t2.dtype}")
            b32 = b.to(torch.float32)
            tv1 = t1.to(torch.float32) - b32
            tv2 = t2.to(torch.float32) - b32
            merged_t = (b32 + LAM * tv1 + LAM * tv2).to(b.dtype)
            out[k] = merged_t.contiguous()
            merged += 1
        else:
            out[k] = b.contiguous()

    # --- Required checks ------------------------------------------------------
    if merged != EXPECTED_MERGED:
        fail(f"merged {merged} tensors, expected {EXPECTED_MERGED}")
    if len(out) != EXPECTED_TOTAL:
        fail(f"output has {len(out)} tensors, expected {EXPECTED_TOTAL}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    save_file(out, OUT, metadata={"format": "pt"})

    # Re-open and confirm the written file.
    with safe_open(OUT, framework="pt", device="cpu") as f:
        written = list(f.keys())
        if len(written) != EXPECTED_TOTAL:
            fail(f"written file has {len(written)} tensors, expected {EXPECTED_TOTAL}")
        if set(written) != base_keys:
            fail("written key set differs from base key set")
        for k in written:
            w = f.get_tensor(k)
            b = base_f.get_tensor(k)
            if w.shape != b.shape or w.dtype != b.dtype:
                fail(f"written tensor {k}: shape/dtype {tuple(w.shape)}/{w.dtype} "
                     f"!= base {tuple(b.shape)}/{b.dtype}")
    print(f"merged {merged} MLP tensors (lambda={LAM}); wrote {len(written)} tensors to {OUT}")


if __name__ == "__main__":
    main()
