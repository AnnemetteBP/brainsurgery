"""T4: task-vector merge of two fine-tunes into a GPT-2 base."""
import os
import re
import sys

import torch
from safetensors.torch import load_file, save_file

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LAMBDA = 0.4
MLP_RE = re.compile(r"^h\.(\d+)\.mlp\.(c_fc|c_proj)\.(weight|bias)$")


def fail(msg):
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def main():
    base = load_file(os.path.join(ROOT, "inputs", "base", "model.safetensors"))
    ft1 = load_file(os.path.join(ROOT, "inputs", "ft1", "model.safetensors"))
    ft2 = load_file(os.path.join(ROOT, "inputs", "ft2", "model.safetensors"))

    # Step 1: verify names and shared tensors before touching anything.
    names = set(base)
    if names != set(ft1) or names != set(ft2):
        fail("tensor name sets differ between base, ft1 and ft2")
    if len(names) != 160:
        fail(f"expected 160 tensors, found {len(names)}")
    mlp = sorted(n for n in names if MLP_RE.match(n))
    if len(mlp) != 48:
        fail(f"expected 48 MLP tensors, found {len(mlp)}")
    for n in sorted(names - set(mlp)):
        for tag, ft in (("ft1", ft1), ("ft2", ft2)):
            if base[n].shape != ft[n].shape or base[n].dtype != ft[n].dtype:
                fail(f"{n}: shape/dtype differs in {tag}")
            if not torch.equal(base[n], ft[n]):
                fail(f"{n}: non-MLP tensor differs between base and {tag}")
    for n in mlp:
        for tag, ft in (("ft1", ft1), ("ft2", ft2)):
            if base[n].shape != ft[n].shape or base[n].dtype != ft[n].dtype:
                fail(f"{n}: shape/dtype differs in {tag}")

    # Step 2 + 3: merge against the unmodified base; copy everything else.
    out = {}
    merged = 0
    for n in sorted(names):
        b = base[n]
        if n in mlp:
            b32 = b.float()
            t = b32 + LAMBDA * (ft1[n].float() - b32) + LAMBDA * (ft2[n].float() - b32)
            out[n] = t.to(b.dtype).contiguous()
            merged += 1
        else:
            out[n] = b.clone().contiguous()

    if merged != 48:
        fail(f"merged {merged} tensors, expected 48")
    if len(out) != 160:
        fail(f"output has {len(out)} tensors, expected 160")

    dst = os.path.join(ROOT, "out", "T4", "model.safetensors")
    save_file(out, dst)
    check = load_file(dst)
    if len(check) != 160:
        fail(f"written file has {len(check)} tensors, expected 160")
    print(f"OK: merged {merged} MLP tensors, wrote {len(check)} tensors to {dst}")


if __name__ == "__main__":
    main()
