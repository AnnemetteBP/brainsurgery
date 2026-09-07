"""T4: task-vector merge of two GPT-2 fine-tunes (torch + safetensors)."""
import re
import sys

import torch
from safetensors.torch import load_file, save_file

LAMBDA = 0.4
MLP_RE = re.compile(r"^h\.(\d+)\.mlp\.(c_fc|c_proj)\.(weight|bias)$")


def main() -> None:
    base = load_file("inputs/base/model.safetensors")
    ft1 = load_file("inputs/ft1/model.safetensors")
    ft2 = load_file("inputs/ft2/model.safetensors")

    # Step 1: same names, and everything outside the MLP set identical in all three.
    if not (base.keys() == ft1.keys() == ft2.keys()):
        raise SystemExit("ERROR: tensor name sets differ between base/ft1/ft2")
    mlp = {k for k in base if MLP_RE.match(k)}
    if len(mlp) != 48:
        raise SystemExit(f"ERROR: expected 48 MLP tensors, found {len(mlp)}")
    for k in base:
        for name, ft in (("ft1", ft1), ("ft2", ft2)):
            if ft[k].shape != base[k].shape or ft[k].dtype != base[k].dtype:
                raise SystemExit(f"ERROR: shape/dtype mismatch for {k} in {name}")
            if k not in mlp and not torch.equal(base[k], ft[k]):
                raise SystemExit(f"ERROR: shared tensor {k} differs in {name}")

    # Step 2/3: each task vector taken against the unmodified base.
    out = {}
    merged = 0
    for k, b in base.items():
        if k in mlp:
            b32 = b.to(torch.float32)
            out[k] = (b32 + LAMBDA * (ft1[k].float() - b32) + LAMBDA * (ft2[k].float() - b32)).to(b.dtype).contiguous()
            merged += 1
        else:
            out[k] = b.contiguous()

    if merged != 48:
        raise SystemExit(f"ERROR: merged {merged} tensors, expected 48")
    if len(out) != 160:
        raise SystemExit(f"ERROR: output has {len(out)} tensors, expected 160")

    save_file(out, "out/T4/model.safetensors")
    check = load_file("out/T4/model.safetensors")
    if len(check) != 160:
        raise SystemExit(f"ERROR: saved file has {len(check)} tensors")
    print(f"OK: merged {merged} MLP tensors, wrote {len(check)} tensors")


if __name__ == "__main__":
    sys.exit(main())
