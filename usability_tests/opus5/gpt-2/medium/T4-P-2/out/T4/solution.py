"""T4: task-vector merge of two GPT-2 fine-tunes."""

from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
INPUTS = ROOT / "inputs"
OUT = HERE / "model.safetensors"

LAMBDA = 0.4
N_LAYERS = 12

MLP_KEYS = [
    f"h.{i}.mlp.{proj}.{kind}"
    for i in range(N_LAYERS)
    for proj in ("c_fc", "c_proj")
    for kind in ("weight", "bias")
]
assert len(MLP_KEYS) == 48, len(MLP_KEYS)


def main() -> None:
    base = load_file(INPUTS / "base" / "model.safetensors")
    ft1 = load_file(INPUTS / "ft1" / "model.safetensors")
    ft2 = load_file(INPUTS / "ft2" / "model.safetensors")

    # --- step 1: verification -------------------------------------------------
    if len(base) != 160:
        raise SystemExit(f"base has {len(base)} tensors, expected 160")
    for name, sd in (("ft1", ft1), ("ft2", ft2)):
        if set(sd) != set(base):
            missing = sorted(set(base) - set(sd))
            extra = sorted(set(sd) - set(base))
            raise SystemExit(
                f"{name} key set differs from base: missing={missing[:5]} extra={extra[:5]}"
            )

    mlp = set(MLP_KEYS)
    absent = sorted(mlp - set(base))
    if absent:
        raise SystemExit(f"expected MLP tensors are not in the checkpoint: {absent}")

    for key in MLP_KEYS:
        for name, sd in (("ft1", ft1), ("ft2", ft2)):
            if sd[key].shape != base[key].shape or sd[key].dtype != base[key].dtype:
                raise SystemExit(
                    f"{name}[{key}] shape/dtype differs: "
                    f"{tuple(sd[key].shape)}/{sd[key].dtype} vs "
                    f"{tuple(base[key].shape)}/{base[key].dtype}"
                )

    mismatched = []
    for key in sorted(set(base) - mlp):
        for name, sd in (("ft1", ft1), ("ft2", ft2)):
            t, b = sd[key], base[key]
            if t.shape != b.shape or t.dtype != b.dtype or not torch.equal(t, b):
                mismatched.append(f"{key} ({name})")
    if mismatched:
        raise SystemExit(
            f"{len(mismatched)} non-MLP tensors differ from the base, "
            f"the frozen-backbone assumption does not hold: {mismatched[:10]}"
        )

    # --- step 2/3: merge ------------------------------------------------------
    out: dict[str, torch.Tensor] = {}
    merged = 0
    for key, tensor in base.items():
        if key in mlp:
            b = tensor.to(torch.float32)
            merged_t = b + LAMBDA * (ft1[key].to(torch.float32) - b) + LAMBDA * (
                ft2[key].to(torch.float32) - b
            )
            out[key] = merged_t.to(tensor.dtype).contiguous()
            merged += 1
        else:
            out[key] = tensor.clone().contiguous()

    if merged != 48:
        raise SystemExit(f"merged {merged} tensors, expected 48")
    if len(out) != 160:
        raise SystemExit(f"output has {len(out)} tensors, expected 160")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    save_file(out, OUT)

    check = load_file(OUT)
    if len(check) != 160:
        raise SystemExit(f"written file has {len(check)} tensors, expected 160")
    for key in sorted(set(base) - mlp):
        if not torch.equal(check[key], base[key]):
            raise SystemExit(f"unchanged tensor {key} was not written bit-exactly")
    print(f"wrote {OUT} ({len(check)} tensors, {merged} merged, lambda={LAMBDA})")


if __name__ == "__main__":
    main()
