#!/usr/bin/env python
"""T1: depth pruning with layer renumbering for Pythia-1B.

Drops blocks 2, 6, 10, 14 from the 16-layer checkpoint and renumbers the
surviving blocks to a contiguous 0..11 range.
"""

import re
import sys
from pathlib import Path

from safetensors.torch import load_file, save_file

HERE = Path(__file__).resolve().parent
SRC = HERE.parent.parent / "inputs" / "base" / "model.safetensors"
DST = HERE / "model.safetensors"

DROP = {2, 6, 10, 14}
N_OLD = 16
N_NEW = N_OLD - len(DROP)
N_EXPECTED_TENSORS = 184

LAYER_RE = re.compile(r"^gpt_neox\.layers\.(\d+)\.(.+)$")


def fail(msg):
    print(f"FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


def main():
    src = load_file(str(SRC))
    print(f"loaded {len(src)} tensors from {SRC}")

    # old index -> new index, surviving blocks in original order
    survivors = [i for i in range(N_OLD) if i not in DROP]
    remap = {old: new for new, old in enumerate(survivors)}
    print(f"remap: {remap}")

    out = {}
    for key, tensor in src.items():
        m = LAYER_RE.match(key)
        if m is None:
            out[key] = tensor  # non-block tensor, unchanged
            continue
        old = int(m.group(1))
        if old not in remap:
            continue  # dropped block
        new_key = f"gpt_neox.layers.{remap[old]}.{m.group(2)}"
        if new_key in out:
            fail(f"renumbering collision on {new_key}")
        out[new_key] = tensor

    # --- required checks -------------------------------------------------
    idxs = sorted({int(LAYER_RE.match(k).group(1)) for k in out if LAYER_RE.match(k)})

    stale = [k for k in out if LAYER_RE.match(k) and int(LAYER_RE.match(k).group(1)) >= N_NEW]
    if stale:
        fail(f"tensors of blocks >= {N_NEW} remain: {stale[:5]}")

    if idxs != list(range(N_NEW)):
        fail(f"block indices are not contiguous 0..{N_NEW - 1}: {idxs}")

    n_qkv = sum(1 for k in out if re.fullmatch(r"gpt_neox\.layers\.\d+\.attention\.query_key_value\.weight", k))
    if n_qkv != N_NEW:
        fail(f"expected {N_NEW} query_key_value.weight tensors, found {n_qkv}")

    if len(out) != N_EXPECTED_TENSORS:
        fail(f"expected {N_EXPECTED_TENSORS} tensors, got {len(out)}")

    # values/shapes/dtypes must be untouched
    inv = {new: old for old, new in remap.items()}
    for key, tensor in out.items():
        m = LAYER_RE.match(key)
        src_key = key if m is None else f"gpt_neox.layers.{inv[int(m.group(1))]}.{m.group(2)}"
        ref = src[src_key]
        if tensor.shape != ref.shape or tensor.dtype != ref.dtype:
            fail(f"shape/dtype changed for {key}")

    for key in ("gpt_neox.embed_in.weight", "embed_out.weight",
                "gpt_neox.final_layer_norm.weight", "gpt_neox.final_layer_norm.bias"):
        if key not in out:
            fail(f"missing non-block tensor {key}")

    save_file({k: v.contiguous() for k, v in out.items()}, str(DST))
    print(f"wrote {len(out)} tensors to {DST}")


if __name__ == "__main__":
    main()
