"""T1: depth-prune Pythia-1B from 16 to 12 blocks, renumbering contiguously.

Plain safetensors + torch script. Renaming happens into a fresh dict, so
there is no overwrite hazard regardless of order; collisions raise instead.
All checks run before anything is written.
"""
import re
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

SRC = Path("inputs/base/model.safetensors")
DST = Path("out/T1/model.safetensors")
REMOVE = {2, 6, 10, 14}
OLD_LAYERS = 16
NEW_LAYERS = 12
TENSORS_PER_BLOCK = 15
NON_BLOCK = 4
EXPECTED_TOTAL = NEW_LAYERS * TENSORS_PER_BLOCK + NON_BLOCK  # 184

LAYER_RE = re.compile(r"^gpt_neox\.layers\.(\d+)\.(.+)$")


def fail(msg: str) -> None:
    print(f"FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    survivors = [i for i in range(OLD_LAYERS) if i not in REMOVE]
    renumber = {old: new for new, old in enumerate(survivors)}
    assert len(renumber) == NEW_LAYERS

    with safe_open(str(SRC), framework="pt") as f:
        metadata = f.metadata()
        keys = list(f.keys())
        if len(keys) != OLD_LAYERS * TENSORS_PER_BLOCK + NON_BLOCK:
            fail(f"unexpected input tensor count {len(keys)}")
        out: dict[str, torch.Tensor] = {}
        removed = 0
        for k in keys:
            m = LAYER_RE.match(k)
            if m is None:
                new_key = k  # non-block tensor, unchanged
            else:
                old = int(m.group(1))
                if old >= OLD_LAYERS:
                    fail(f"layer index out of range in {k}")
                if old in REMOVE:
                    removed += 1
                    continue
                new_key = f"gpt_neox.layers.{renumber[old]}.{m.group(2)}"
            if new_key in out:
                fail(f"collision: {new_key} already present (from {k})")
            out[new_key] = f.get_tensor(k).contiguous()

    # Required checks (all before writing).
    if removed != len(REMOVE) * TENSORS_PER_BLOCK:
        fail(f"removed {removed} tensors, expected {len(REMOVE) * TENSORS_PER_BLOCK}")
    stale = [k for k in out if (m := LAYER_RE.match(k)) and int(m.group(1)) >= NEW_LAYERS]
    if stale:
        fail(f"tensors of blocks >= {NEW_LAYERS} remain: {stale[:5]}")
    qkv = sorted(
        int(m.group(1))
        for k in out
        if (m := LAYER_RE.match(k)) and m.group(2) == "attention.query_key_value.weight"
    )
    if qkv != list(range(NEW_LAYERS)):
        fail(f"expected qkv weights for blocks 0..{NEW_LAYERS - 1}, got {qkv}")
    per_block = {}
    for k in out:
        if m := LAYER_RE.match(k):
            per_block[int(m.group(1))] = per_block.get(int(m.group(1)), 0) + 1
    if any(n != TENSORS_PER_BLOCK for n in per_block.values()):
        fail(f"uneven tensor count per block: {per_block}")
    if len(out) != EXPECTED_TOTAL:
        fail(f"output has {len(out)} tensors, expected {EXPECTED_TOTAL}")

    DST.parent.mkdir(parents=True, exist_ok=True)
    save_file(out, str(DST), metadata=metadata)
    print(f"wrote {DST} with {len(out)} tensors; removed blocks {sorted(REMOVE)}")


if __name__ == "__main__":
    main()
