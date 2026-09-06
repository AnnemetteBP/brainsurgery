"""
T1: Depth pruning with layer renumbering (OLMo-1B-0724-hf)

Removes transformer blocks {2, 6, 10, 14} from a 16-layer checkpoint and
renumbers the surviving blocks to 0..11, preserving order. Non-block
tensors (embeddings, lm_head) pass through unchanged.
"""

import json
import os
import re
import sys

from safetensors import safe_open
from safetensors.torch import save_file

INPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "inputs", "base")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "T1", "model.safetensors")

DROP_BLOCKS = {2, 6, 10, 14}
NUM_ORIGINAL_BLOCKS = 16
NUM_SURVIVING_BLOCKS = NUM_ORIGINAL_BLOCKS - len(DROP_BLOCKS)

LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.")


def main():
    index_path = os.path.join(INPUT_DIR, "model.safetensors.index.json")
    with open(index_path) as f:
        index = json.load(f)
    weight_map = index["weight_map"]

    # Load every tensor into memory, grouped by shard file to avoid opening
    # each shard repeatedly.
    tensors_by_name = {}
    shard_files = sorted(set(weight_map.values()))
    for shard in shard_files:
        shard_path = os.path.join(INPUT_DIR, shard)
        with safe_open(shard_path, framework="pt") as sf:
            for key in sf.keys():
                tensors_by_name[key] = sf.get_tensor(key)

    if set(tensors_by_name.keys()) != set(weight_map.keys()):
        raise RuntimeError("Loaded tensor keys do not match index weight_map keys")

    # Build old->new block index mapping: surviving blocks keep relative order.
    surviving_old_indices = [i for i in range(NUM_ORIGINAL_BLOCKS) if i not in DROP_BLOCKS]
    if len(surviving_old_indices) != NUM_SURVIVING_BLOCKS:
        raise RuntimeError("Unexpected number of surviving blocks")
    old_to_new = {old: new for new, old in enumerate(surviving_old_indices)}

    output_tensors = {}
    for name, tensor in tensors_by_name.items():
        m = LAYER_RE.match(name)
        if m is None:
            # Non-block tensor: passthrough unchanged.
            output_tensors[name] = tensor
            continue
        old_idx = int(m.group(1))
        if old_idx in DROP_BLOCKS:
            continue
        new_idx = old_to_new[old_idx]
        new_name = f"model.layers.{new_idx}." + name[m.end():]
        output_tensors[new_name] = tensor

    # --- Required checks: fail loudly, do not write output on failure. ---

    # Exactly 12 blocks remain, with contiguous indices 0..11.
    q_proj_pattern = re.compile(r"^model\.layers\.(\d+)\.self_attn\.q_proj\.weight$")
    block_indices_present = sorted(
        int(mm.group(1)) for k in output_tensors if (mm := q_proj_pattern.match(k))
    )
    if block_indices_present != list(range(NUM_SURVIVING_BLOCKS)):
        raise RuntimeError(
            f"Expected contiguous block indices 0..{NUM_SURVIVING_BLOCKS - 1}, "
            f"got {block_indices_present}"
        )

    # No output tensor above the surviving block range (i.e. old blocks
    # 12-15 must not appear renumbered above index 11).
    max_index_present = max(block_indices_present)
    if max_index_present != NUM_SURVIVING_BLOCKS - 1:
        raise RuntimeError(f"Highest surviving block index is {max_index_present}, expected 11")

    expected_non_block = 2
    expected_total = NUM_SURVIVING_BLOCKS * 7 + expected_non_block
    if expected_total != 86:
        raise RuntimeError(f"Internal arithmetic error: expected 86, computed {expected_total}")
    if len(output_tensors) != 86:
        raise RuntimeError(f"Expected exactly 86 tensors in output, got {len(output_tensors)}")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    save_file(output_tensors, OUTPUT_PATH)
    print(f"Wrote {len(output_tensors)} tensors to {OUTPUT_PATH}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        sys.exit(1)
