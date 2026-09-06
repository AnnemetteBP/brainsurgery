"""T4: task-vector merge of two fine-tunes of GPT-2 (124M).

out[X] = base[X] + lam*(ft1[X]-base[X]) + lam*(ft2[X]-base[X])  for the 48 MLP tensors,
everything else copied from base unchanged. Each task vector is taken against the
unmodified base. Fails loudly on any precondition violation.
"""
import re
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parents[2]
BASE, FT1, FT2 = (ROOT / "inputs" / d / "model.safetensors" for d in ("base", "ft1", "ft2"))
OUT = ROOT / "out" / "T4" / "model.safetensors"
LAMBDA = 0.4
N_LAYERS = 12
EXPECTED_TOTAL = 160
EXPECTED_MERGED = 48

MLP_RE = re.compile(r"^h\.(\d+)\.mlp\.(c_fc|c_proj)\.(weight|bias)$")


def fail(msg: str) -> None:
    raise SystemExit(f"ERROR: {msg}")


def main() -> None:
    base = load_file(str(BASE))
    ft1 = load_file(str(FT1))
    ft2 = load_file(str(FT2))

    # Step 1: same tensor names in all three checkpoints.
    kb, k1, k2 = set(base), set(ft1), set(ft2)
    if not (kb == k1 == k2):
        fail(
            "tensor name sets differ: "
            f"base-ft1={sorted(kb ^ k1)[:5]} base-ft2={sorted(kb ^ k2)[:5]}"
        )
    if len(kb) != EXPECTED_TOTAL:
        fail(f"expected {EXPECTED_TOTAL} tensors in base, found {len(kb)}")

    # Identify the MLP tensors explicitly and cross-check against the regex.
    mlp_names = {
        f"h.{i}.mlp.{mod}.{p}"
        for i in range(N_LAYERS)
        for mod in ("c_fc", "c_proj")
        for p in ("weight", "bias")
    }
    regex_names = {k for k in kb if MLP_RE.match(k)}
    if mlp_names != regex_names:
        fail(f"MLP name set mismatch: {sorted(mlp_names ^ regex_names)[:5]}")
    if len(mlp_names) != EXPECTED_MERGED:
        fail(f"expected {EXPECTED_MERGED} MLP tensors, got {len(mlp_names)}")
    missing = mlp_names - kb
    if missing:
        fail(f"MLP tensors missing from checkpoints: {sorted(missing)[:5]}")

    # Step 1 (cont.): every non-MLP tensor identical (shape, dtype, bit-exact values) in all three.
    for k in sorted(kb):
        b, a1, a2 = base[k], ft1[k], ft2[k]
        for tag, t in (("ft1", a1), ("ft2", a2)):
            if t.shape != b.shape or t.dtype != b.dtype:
                fail(f"{k}: {tag} shape/dtype {tuple(t.shape)}/{t.dtype} != base {tuple(b.shape)}/{b.dtype}")
        if b.dtype != torch.float32:
            fail(f"{k}: expected float32, got {b.dtype}")
        if k in mlp_names:
            continue
        if not torch.equal(b, a1):
            fail(f"non-MLP tensor {k} differs between base and ft1")
        if not torch.equal(b, a2):
            fail(f"non-MLP tensor {k} differs between base and ft2")
    print(f"verified: {len(kb)} shared names, {len(kb) - len(mlp_names)} non-MLP tensors identical")

    # Step 2/3: merge MLP tensors against the unmodified base; copy the rest.
    out: dict[str, torch.Tensor] = {}
    merged = 0
    for k in sorted(kb):
        b = base[k]
        if k in mlp_names:
            tv1 = ft1[k].float() - b.float()
            tv2 = ft2[k].float() - b.float()
            out[k] = (b.float() + LAMBDA * tv1 + LAMBDA * tv2).to(torch.float32).contiguous()
            merged += 1
        else:
            out[k] = b.contiguous()

    if merged != EXPECTED_MERGED:
        fail(f"merged {merged} tensors, expected {EXPECTED_MERGED}")
    if len(out) != EXPECTED_TOTAL:
        fail(f"output has {len(out)} tensors, expected {EXPECTED_TOTAL}")
    if set(out) != kb:
        fail("output key set differs from base key set")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    save_file(out, str(OUT), metadata={"format": "pt"})

    # Post-write verification of the file on disk.
    written = load_file(str(OUT))
    if len(written) != EXPECTED_TOTAL:
        fail(f"written file has {len(written)} tensors, expected {EXPECTED_TOTAL}")
    if set(written) != kb:
        fail("written key set differs from base key set")
    n_changed = 0
    for k in sorted(kb):
        w = written[k]
        if w.shape != base[k].shape or w.dtype != base[k].dtype:
            fail(f"written {k} shape/dtype mismatch")
        if k in mlp_names:
            expect = base[k] + LAMBDA * (ft1[k] - base[k]) + LAMBDA * (ft2[k] - base[k])
            rel = (w - expect).norm() / max(expect.norm().item(), 1e-30)
            if rel > 1e-6:
                fail(f"written {k} relative error {rel:.3e} too large")
            if not torch.equal(w, base[k]):
                n_changed += 1
        elif not torch.equal(w, base[k]):
            fail(f"written non-MLP tensor {k} differs from base")
    print(f"merged {merged} MLP tensors (lambda={LAMBDA}); {n_changed} differ from base on disk")
    print(f"wrote {OUT} with {len(written)} tensors")


if __name__ == "__main__":
    try:
        main()
    except SystemExit as e:
        if e.code not in (None, 0):
            print(e, file=sys.stderr)
        raise
