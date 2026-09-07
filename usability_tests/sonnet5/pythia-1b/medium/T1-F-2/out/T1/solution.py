"""
T1: Depth pruning with layer renumbering (Pythia-1B).

Remove transformer blocks 2, 6, 10, 14 from a 16-layer Pythia-1B checkpoint
and renumber the survivors to 0..11, preserving order. Non-block tensors are
copied unchanged. Fails loudly (raises / non-zero exit, no output written) if
any required invariant does not hold.

Uses `safetensors` directly for a straightforward key-rewrite; no adapter
merging or layer-slicing config is needed for a single flat rename+drop.
"""

import re
import sys
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file

DROP_BLOCKS = {2, 6, 10, 14}
NUM_ORIGINAL_BLOCKS = 16
LAYER_RE = re.compile(r"^gpt_neox\.layers\.(\d+)\.")

TENSORS_PER_BLOCK = 15
NUM_NON_BLOCK_TENSORS = 4
NUM_SURVIVING_BLOCKS = NUM_ORIGINAL_BLOCKS - len(DROP_BLOCKS)
EXPECTED_TOTAL = NUM_SURVIVING_BLOCKS * TENSORS_PER_BLOCK + NUM_NON_BLOCK_TENSORS


def main() -> None:
    in_path = Path(__file__).resolve().parents[2] / "inputs" / "base" / "model.safetensors"
    out_path = Path(__file__).resolve().parents[2] / "out" / "T1" / "model.safetensors"

    if not in_path.exists():
        raise FileNotFoundError(f"input checkpoint not found: {in_path}")

    surviving_old = [i for i in range(NUM_ORIGINAL_BLOCKS) if i not in DROP_BLOCKS]
    assert surviving_old == sorted(surviving_old)
    old_to_new = {old: new for new, old in enumerate(surviving_old)}

    tensors = {}
    non_block_count = 0
    block_count = 0

    with safe_open(str(in_path), framework="pt") as f:
        keys = list(f.keys())
        if len(keys) != 244:
            raise ValueError(f"expected 244 input tensors, got {len(keys)}")

        for key in keys:
            m = LAYER_RE.match(key)
            if m is None:
                # non-block tensor: unchanged
                tensors[key] = f.get_tensor(key)
                non_block_count += 1
                continue

            old_idx = int(m.group(1))
            if old_idx in DROP_BLOCKS:
                continue  # drop this tensor entirely

            new_idx = old_to_new[old_idx]
            new_key = f"gpt_neox.layers.{new_idx}.{key[m.end():]}"
            if new_key in tensors:
                raise ValueError(f"collision: {new_key} already produced (from {key})")
            tensors[new_key] = f.get_tensor(key)
            block_count += 1

    if non_block_count != NUM_NON_BLOCK_TENSORS:
        raise ValueError(f"expected {NUM_NON_BLOCK_TENSORS} non-block tensors, got {non_block_count}")
    if block_count != NUM_SURVIVING_BLOCKS * TENSORS_PER_BLOCK:
        raise ValueError(
            f"expected {NUM_SURVIVING_BLOCKS * TENSORS_PER_BLOCK} block tensors, got {block_count}"
        )

    # --- Required checks ---

    # no tensor of blocks 12, 13, 14, 15 remains (post-renumbering namespace)
    for banned in (12, 13, 14, 15):
        pattern = re.compile(rf"^gpt_neox\.layers\.{banned}\.")
        if any(pattern.match(k) for k in tensors):
            raise AssertionError(f"tensor of banned layer index {banned} present in output")

    # exactly 12 blocks remain
    qkv_pattern = re.compile(r"^gpt_neox\.layers\.(\d+)\.attention\.query_key_value\.weight$")
    qkv_matches = [k for k in tensors if qkv_pattern.match(k)]
    if len(qkv_matches) != NUM_SURVIVING_BLOCKS:
        raise AssertionError(
            f"expected exactly {NUM_SURVIVING_BLOCKS} query_key_value.weight tensors, "
            f"got {len(qkv_matches)}"
        )
    seen_indices = sorted(int(qkv_pattern.match(k).group(1)) for k in qkv_matches)
    if seen_indices != list(range(NUM_SURVIVING_BLOCKS)):
        raise AssertionError(f"surviving block indices are not contiguous 0..11: {seen_indices}")

    # exact tensor count
    if len(tensors) != EXPECTED_TOTAL:
        raise AssertionError(f"expected {EXPECTED_TOTAL} output tensors, got {len(tensors)}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(".safetensors.tmp")
    save_file(tensors, str(tmp_path))
    tmp_path.rename(out_path)

    print(f"wrote {out_path} with {len(tensors)} tensors "
          f"({NUM_SURVIVING_BLOCKS} blocks x {TENSORS_PER_BLOCK} + {NUM_NON_BLOCK_TENSORS} non-block)")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"FAILED: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)
