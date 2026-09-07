"""T4: task-vector merge of two Pythia-1B fine-tunes (lambda = 0.4).

out[X] = base[X] + lam*(ft1[X]-base[X]) + lam*(ft2[X]-base[X])  for the 64 MLP tensors,
computed in float32 and cast back to the base dtype; everything else copied bit-exact
from base. Aborts before writing anything if the three checkpoints do not share the
same tensor names or differ outside the MLP tensors.
"""
import os
import re
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INPUTS = os.path.join(ROOT, "inputs")
OUT_DIR = os.path.join(ROOT, "out", "T4")
OUT_FILE = os.path.join(OUT_DIR, "model.safetensors")

LAM = 0.4
N_LAYERS = 16
EXPECTED_TOTAL = 244
EXPECTED_MERGED = 64
MLP_RE = re.compile(r"^gpt_neox\.layers\.(\d+)\.mlp\.dense_(h_to_4h|4h_to_h)\.(weight|bias)$")


def fail(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    if os.path.exists(OUT_FILE):
        fail(f"destination already exists: {OUT_FILE}")

    handles = {
        name: safe_open(os.path.join(INPUTS, name, "model.safetensors"), "pt", device="cpu")
        for name in ("base", "ft1", "ft2")
    }
    base, ft1, ft2 = handles["base"], handles["ft1"], handles["ft2"]

    # ---- Step 1: precondition checks, before any arithmetic ---------------------------
    keys = {n: set(h.keys()) for n, h in handles.items()}
    if not (keys["base"] == keys["ft1"] == keys["ft2"]):
        fail(
            "tensor name sets differ: "
            f"base-ft1={sorted(keys['base'] ^ keys['ft1'])[:5]} "
            f"base-ft2={sorted(keys['base'] ^ keys['ft2'])[:5]}"
        )
    names = sorted(keys["base"])
    if len(names) != EXPECTED_TOTAL:
        fail(f"expected {EXPECTED_TOTAL} tensors in base, found {len(names)}")

    mlp_names = [n for n in names if MLP_RE.match(n)]
    layer_ids = {int(MLP_RE.match(n).group(1)) for n in mlp_names}
    if len(mlp_names) != EXPECTED_MERGED or layer_ids != set(range(N_LAYERS)):
        fail(f"expected {EXPECTED_MERGED} MLP tensors over layers 0..{N_LAYERS-1}, "
             f"found {len(mlp_names)} over layers {sorted(layer_ids)}")
    mlp_set = set(mlp_names)

    for n in names:
        b = base.get_tensor(n)
        t1 = ft1.get_tensor(n)
        t2 = ft2.get_tensor(n)
        for tag, t in (("ft1", t1), ("ft2", t2)):
            if t.shape != b.shape or t.dtype != b.dtype:
                fail(f"{n}: {tag} shape/dtype {tuple(t.shape)}/{t.dtype} "
                     f"!= base {tuple(b.shape)}/{b.dtype}")
        if n not in mlp_set:
            # bit-exact comparison (view as raw bits so NaN patterns also compare equal)
            for tag, t in (("ft1", t1), ("ft2", t2)):
                if not torch.equal(b.view(torch.int16), t.view(torch.int16)):
                    fail(f"shared tensor differs between base and {tag}: {n}")
    print(f"precondition OK: {len(names)} names shared, "
          f"{len(names) - len(mlp_names)} non-MLP tensors identical across all three")

    # ---- Step 2/3: merge, every task vector taken against the untouched base ---------
    out: dict[str, torch.Tensor] = {}
    merged = 0
    for n in names:
        b = base.get_tensor(n)
        if n in mlp_set:
            b32 = b.to(torch.float32)
            tv1 = ft1.get_tensor(n).to(torch.float32) - b32
            tv2 = ft2.get_tensor(n).to(torch.float32) - b32
            res = (b32 + LAM * tv1 + LAM * tv2).to(b.dtype)
            if not torch.isfinite(res.float()).all():
                fail(f"non-finite values after merge in {n}")
            out[n] = res.contiguous()
            merged += 1
        else:
            out[n] = b.contiguous()

    # ---- Required checks --------------------------------------------------------------
    if merged != EXPECTED_MERGED:
        fail(f"merged {merged} tensors, expected {EXPECTED_MERGED}")
    if len(out) != EXPECTED_TOTAL:
        fail(f"output has {len(out)} tensors, expected {EXPECTED_TOTAL}")

    os.makedirs(OUT_DIR, exist_ok=True)
    save_file(out, OUT_FILE, metadata={"format": "pt"})

    # ---- Post-write verification of the file on disk ----------------------------------
    with safe_open(OUT_FILE, "pt", device="cpu") as chk:
        chk_keys = list(chk.keys())
        if len(chk_keys) != EXPECTED_TOTAL or set(chk_keys) != set(names):
            fail(f"written file has {len(chk_keys)} tensors / wrong key set")
        for n in names:
            w = chk.get_tensor(n)
            b = base.get_tensor(n)
            if w.shape != b.shape or w.dtype != b.dtype:
                fail(f"written {n}: shape/dtype mismatch vs base")
            if n not in mlp_set and not torch.equal(w.view(torch.int16), b.view(torch.int16)):
                fail(f"written unchanged tensor {n} is not bit-exact with base")
    print(f"wrote {OUT_FILE}: {len(chk_keys)} tensors, {merged} merged with lambda={LAM}")


if __name__ == "__main__":
    main()
