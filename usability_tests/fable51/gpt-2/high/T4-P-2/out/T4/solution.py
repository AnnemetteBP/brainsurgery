"""T4: task-vector merge of two frozen-backbone fine-tunes of GPT-2 (124M).

out[X] = base[X] + lam*(ft1[X]-base[X]) + lam*(ft2[X]-base[X])  for the 48 MLP tensors,
everything else copied from the base unchanged.
"""
import re
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "inputs" / "base" / "model.safetensors"
FT1 = ROOT / "inputs" / "ft1" / "model.safetensors"
FT2 = ROOT / "inputs" / "ft2" / "model.safetensors"
OUT = ROOT / "out" / "T4" / "model.safetensors"

LAMBDA = 0.4
N_LAYERS = 12
EXPECTED_TOTAL = 160
EXPECTED_MERGED = 48

MLP_RE = re.compile(r"^h\.(\d+)\.mlp\.(c_fc|c_proj)\.(weight|bias)$")


def fail(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    base = load_file(str(BASE))
    ft1 = load_file(str(FT1))
    ft2 = load_file(str(FT2))

    # --- step 1: verify layout and shared (non-MLP) tensors before touching anything ---
    base_keys = set(base)
    if len(base_keys) != EXPECTED_TOTAL:
        fail(f"base has {len(base_keys)} tensors, expected {EXPECTED_TOTAL}")
    for name, sd in (("ft1", ft1), ("ft2", ft2)):
        if set(sd) != base_keys:
            missing = sorted(base_keys - set(sd))
            extra = sorted(set(sd) - base_keys)
            fail(f"{name} tensor names differ from base: missing={missing[:5]} extra={extra[:5]}")

    mlp_keys = {k for k in base_keys if MLP_RE.match(k)}
    expected_mlp = {
        f"h.{i}.mlp.{mod}.{p}"
        for i in range(N_LAYERS)
        for mod in ("c_fc", "c_proj")
        for p in ("weight", "bias")
    }
    if mlp_keys != expected_mlp:
        fail(f"MLP tensor set mismatch: got {len(mlp_keys)}, expected {len(expected_mlp)}")

    for k in sorted(base_keys):
        for name, sd in (("ft1", ft1), ("ft2", ft2)):
            if sd[k].shape != base[k].shape:
                fail(f"{name}[{k}] shape {tuple(sd[k].shape)} != base {tuple(base[k].shape)}")
            if sd[k].dtype != base[k].dtype:
                fail(f"{name}[{k}] dtype {sd[k].dtype} != base {base[k].dtype}")
        if k not in mlp_keys:
            for name, sd in (("ft1", ft1), ("ft2", ft2)):
                if not torch.equal(sd[k], base[k]):
                    fail(f"non-MLP tensor {k} differs between base and {name}")

    for k in sorted(mlp_keys):
        if base[k].dtype != torch.float32:
            fail(f"MLP tensor {k} is {base[k].dtype}, expected float32")

    # --- step 2/3: merge, each task vector taken against the untouched base ---
    out: dict[str, torch.Tensor] = {}
    merged = 0
    for k in base_keys:
        b = base[k]
        if k in mlp_keys:
            tv1 = ft1[k] - b
            tv2 = ft2[k] - b
            out[k] = (b + LAMBDA * tv1 + LAMBDA * tv2).to(torch.float32).contiguous()
            merged += 1
        else:
            out[k] = b.clone().contiguous()

    if merged != EXPECTED_MERGED:
        fail(f"merged {merged} tensors, expected {EXPECTED_MERGED}")
    if len(out) != EXPECTED_TOTAL:
        fail(f"output has {len(out)} tensors, expected {EXPECTED_TOTAL}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    save_file(out, str(OUT), metadata={"format": "pt"})

    # --- post-write verification ---
    check = load_file(str(OUT))
    if len(check) != EXPECTED_TOTAL or set(check) != base_keys:
        fail(f"written file has {len(check)} tensors / key mismatch")
    for k in base_keys:
        if check[k].shape != base[k].shape or check[k].dtype != base[k].dtype:
            fail(f"written {k} shape/dtype mismatch")
        if k not in mlp_keys and not torch.equal(check[k], base[k]):
            fail(f"written non-MLP tensor {k} differs from base")
    # merged tensors: recompute in float64 and compare relative Frobenius error
    worst = 0.0
    for k in mlp_keys:
        ref = base[k].double() + LAMBDA * (ft1[k].double() - base[k].double()) \
            + LAMBDA * (ft2[k].double() - base[k].double())
        err = (check[k].double() - ref).norm() / max(ref.norm().item(), 1e-30)
        worst = max(worst, err.item())
    if worst > 1e-5:
        fail(f"merged tensor relative Frobenius error {worst:.3e} exceeds 1e-5")

    print(f"OK: wrote {OUT} with {len(check)} tensors, {merged} merged, "
          f"max rel Frobenius err vs float64 ref {worst:.3e}")


if __name__ == "__main__":
    main()
