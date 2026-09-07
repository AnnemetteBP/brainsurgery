"""
T1: Depth pruning with layer renumbering (OLMo-1B-0724-hf).

Remove transformer blocks 2, 6, 10, 14 from a 16-layer checkpoint and
renumber the surviving blocks to 0..11 (contiguous), keeping the two
non-block tensors (embeddings, lm_head) unchanged. Writes a single
merged safetensors file with exactly 86 tensors.
"""

import json
import os
import re
import sys

from safetensors import safe_open
from safetensors.torch import save_file

INPUT_DIR = "inputs/base"
OUTPUT_DIR = "out/T1"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "model.safetensors")

DROP_BLOCKS = {2, 6, 10, 14}
NUM_ORIGINAL_BLOCKS = 16
NUM_TENSORS_PER_BLOCK = 7
NUM_NON_BLOCK_TENSORS = 2

LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.")


def build_old_to_new_mapping():
    surviving_old = [i for i in range(NUM_ORIGINAL_BLOCKS) if i not in DROP_BLOCKS]
    return {old: new for new, old in enumerate(surviving_old)}


def main():
    index_path = os.path.join(INPUT_DIR, "model.safetensors.index.json")
    with open(index_path) as f:
        index = json.load(f)
    weight_map = index["weight_map"]

    old_to_new = build_old_to_new_mapping()
    if len(old_to_new) != NUM_ORIGINAL_BLOCKS - len(DROP_BLOCKS):
        print("FATAL: unexpected number of surviving blocks", file=sys.stderr)
        sys.exit(1)

    # Load every tensor from its shard.
    shard_cache = {}
    tensors = {}
    for name in weight_map:
        shard_file = weight_map[name]
        if shard_file not in shard_cache:
            shard_cache[shard_file] = safe_open(
                os.path.join(INPUT_DIR, shard_file), framework="pt"
            )
        tensors[name] = shard_cache[shard_file].get_tensor(name)

    output = {}
    for name, tensor in tensors.items():
        m = LAYER_RE.match(name)
        if m is None:
            # Non-block tensor: keep unchanged.
            output[name] = tensor
            continue
        old_idx = int(m.group(1))
        if old_idx in DROP_BLOCKS:
            continue
        new_idx = old_to_new[old_idx]
        new_name = LAYER_RE.sub(f"model.layers.{new_idx}.", name, count=1)
        if new_name in output:
            print(f"FATAL: name collision writing {new_name}", file=sys.stderr)
            sys.exit(1)
        output[new_name] = tensor

    # --- Required checks: fail loudly, no output written on failure. ---

    # No tensor of blocks 12, 13, 14, 15 (old indices dropped or renumbered
    # away) may remain in the output under those indices.
    for forbidden in (12, 13, 14, 15):
        pattern = re.compile(rf"^model\.layers\.{forbidden}\.")
        if any(pattern.match(name) for name in output):
            print(
                f"FATAL: tensor for forbidden block {forbidden} present in output",
                file=sys.stderr,
            )
            sys.exit(1)

    # Exactly 12 blocks remain.
    q_proj_pattern = re.compile(r"^model\.layers\.(\d+)\.self_attn\.q_proj\.weight$")
    block_indices = sorted(
        int(q_proj_pattern.match(name).group(1))
        for name in output
        if q_proj_pattern.match(name)
    )
    expected_block_indices = list(range(12))
    if block_indices != expected_block_indices:
        print(
            f"FATAL: expected blocks {expected_block_indices}, got {block_indices}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Exactly 86 tensors total (12 blocks * 7 + 2 non-block tensors).
    expected_total = len(expected_block_indices) * NUM_TENSORS_PER_BLOCK + NUM_NON_BLOCK_TENSORS
    if len(output) != expected_total:
        print(
            f"FATAL: expected {expected_total} tensors, got {len(output)}",
            file=sys.stderr,
        )
        sys.exit(1)
    if expected_total != 86:
        print("FATAL: expected_total sanity check failed (not 86)", file=sys.stderr)
        sys.exit(1)

    # Non-block tensors unchanged in count and identity.
    non_block_names = [n for n in output if LAYER_RE.match(n) is None]
    if len(non_block_names) != NUM_NON_BLOCK_TENSORS:
        print(
            f"FATAL: expected {NUM_NON_BLOCK_TENSORS} non-block tensors, "
            f"got {len(non_block_names)}",
            file=sys.stderr,
        )
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_file(output, OUTPUT_FILE, metadata={"format": "pt"})

    print(f"Wrote {len(output)} tensors to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
