"""T1: depth-prune Pythia-1B from 16 to 12 blocks, renumbering contiguously."""
import os
import re
import sys

import torch
from safetensors.torch import load_file, save_file

IN_PATH = "inputs/base/model.safetensors"
OUT_DIR = "out/T1"
OUT_PATH = os.path.join(OUT_DIR, "model.safetensors")
REMOVE = {2, 6, 10, 14}
N_OLD = 16
N_NEW = N_OLD - len(REMOVE)
LAYER_RE = re.compile(r"^gpt_neox\.layers\.(\d+)\.(.+)$")


def fail(msg):
    print(f"FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


def main():
    src = load_file(IN_PATH)
    if len(src) != 244:
        fail(f"expected 244 input tensors, got {len(src)}")

    # Build the old->new index map once, from surviving indices in order.
    survivors = [i for i in range(N_OLD) if i not in REMOVE]
    remap = {old: new for new, old in enumerate(survivors)}

    # Construct a fresh dict, so renaming order cannot cause collisions.
    dst = {}
    for name, t in src.items():
        m = LAYER_RE.match(name)
        if m is None:
            dst[name] = t  # non-block tensor, unchanged
            continue
        old = int(m.group(1))
        if old >= N_OLD:
            fail(f"unexpected layer index {old} in {name}")
        if old in REMOVE:
            continue
        new_name = f"gpt_neox.layers.{remap[old]}.{m.group(2)}"
        if new_name in dst:
            fail(f"collision on {new_name}")
        dst[new_name] = t

    # Required checks.
    for name in dst:
        m = LAYER_RE.match(name)
        if m and int(m.group(1)) >= N_NEW:
            fail(f"block index >= {N_NEW} remains: {name}")
    qkv = sorted(
        int(LAYER_RE.match(n).group(1))
        for n in dst
        if n.endswith(".attention.query_key_value.weight") and LAYER_RE.match(n)
    )
    if qkv != list(range(N_NEW)):
        fail(f"expected exactly {N_NEW} contiguous blocks, got qkv indices {qkv}")
    if len(dst) != 184:
        fail(f"expected 184 output tensors, got {len(dst)}")

    # Extra sanity: every survivor block keeps all 15 tensors, values identical.
    for old, new in remap.items():
        old_keys = {k for k in src if k.startswith(f"gpt_neox.layers.{old}.")}
        new_keys = {k for k in dst if k.startswith(f"gpt_neox.layers.{new}.")}
        if len(old_keys) != 15 or len(new_keys) != 15:
            fail(f"block old={old}/new={new} has {len(old_keys)}/{len(new_keys)} tensors")
        for k in old_keys:
            rest = k[len(f"gpt_neox.layers.{old}.") :]
            nk = f"gpt_neox.layers.{new}.{rest}"
            a, b = src[k], dst[nk]
            if a.shape != b.shape or a.dtype != b.dtype or not torch.equal(a, b):
                fail(f"mismatch {k} -> {nk}")

    os.makedirs(OUT_DIR, exist_ok=True)
    save_file({k: v.contiguous() for k, v in dst.items()}, OUT_PATH)
    print(f"wrote {OUT_PATH} with {len(dst)} tensors, blocks 0..{N_NEW - 1}")


if __name__ == "__main__":
    main()
