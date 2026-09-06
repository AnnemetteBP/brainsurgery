#!/usr/bin/env python
"""T1: depth-prune Pythia-1B from 16 to 12 blocks with contiguous renumbering.

Tools: safetensors (load/save), torch (tensors). A plain script was the
smallest correct route; see REPORT.md.
"""
import re
import sys
from pathlib import Path

from safetensors.torch import load_file, save_file

SRC = Path("inputs/base/model.safetensors")
DST = Path("out/T1/model.safetensors")

DROP = {2, 6, 10, 14}
N_OLD = 16
N_NEW = 12
N_TENSORS = 184

LAYER_RE = re.compile(r"^gpt_neox\.layers\.(\d+)\.(.+)$")


def fail(msg):
    print(f"CHECK FAILED: {msg}", file=sys.stderr)
    sys.exit(1)


def main():
    src = load_file(str(SRC))

    keep = [i for i in range(N_OLD) if i not in DROP]
    if len(keep) != N_NEW:
        fail(f"expected {N_NEW} surviving blocks, got {len(keep)}")
    remap = {old: new for new, old in enumerate(keep)}

    out = {}
    for key, value in src.items():
        m = LAYER_RE.match(key)
        if m is None:
            out[key] = value  # non-block tensor, unchanged
            continue
        old = int(m.group(1))
        if old in DROP:
            continue
        new_key = f"gpt_neox.layers.{remap[old]}.{m.group(2)}"
        if new_key in out:
            fail(f"renumbering collision on {new_key}")
        out[new_key] = value

    # --- required checks, before anything is written ---
    for i in range(N_NEW, N_OLD):
        stale = [k for k in out if k.startswith(f"gpt_neox.layers.{i}.")]
        if stale:
            fail(f"tensors of block {i} remain: {stale[:3]}")

    qkv = [k for k in out if re.fullmatch(
        r"gpt_neox\.layers\.\d+\.attention\.query_key_value\.weight", k)]
    if len(qkv) != N_NEW:
        fail(f"expected {N_NEW} query_key_value.weight tensors, got {len(qkv)}")

    idx = sorted(int(LAYER_RE.match(k).group(1)) for k in out if LAYER_RE.match(k))
    if set(idx) != set(range(N_NEW)):
        fail(f"block indices are not contiguous 0..{N_NEW - 1}: {sorted(set(idx))}")

    if len(out) != N_TENSORS:
        fail(f"expected {N_TENSORS} tensors, got {len(out)}")

    # values/shapes/dtypes must be untouched
    for new_key, value in out.items():
        m = LAYER_RE.match(new_key)
        old_key = new_key if m is None else \
            f"gpt_neox.layers.{keep[int(m.group(1))]}.{m.group(2)}"
        ref = src[old_key]
        if value.shape != ref.shape or value.dtype != ref.dtype:
            fail(f"shape/dtype changed for {new_key}")

    DST.parent.mkdir(parents=True, exist_ok=True)
    save_file({k: v.contiguous() for k, v in out.items()}, str(DST))
    print(f"wrote {DST} with {len(out)} tensors, {N_NEW} blocks")


if __name__ == "__main__":
    main()
