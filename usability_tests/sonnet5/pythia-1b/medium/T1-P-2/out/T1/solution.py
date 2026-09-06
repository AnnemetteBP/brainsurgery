"""
Depth-prune Pythia-1B: remove transformer blocks 2, 6, 10, 14 and renumber
the remaining 12 blocks to 0..11, preserving order. Non-block tensors are
copied through unchanged.
"""

import re
import sys

from safetensors import safe_open
from safetensors.torch import save_file

INPUT_PATH = "inputs/base/model.safetensors"
OUTPUT_PATH = "out/T1/model.safetensors"

REMOVE_BLOCKS = {2, 6, 10, 14}
NUM_ORIGINAL_BLOCKS = 16
NUM_TARGET_BLOCKS = 12
TENSORS_PER_BLOCK = 15
NUM_NON_BLOCK_TENSORS = 4
EXPECTED_TOTAL_IN = NUM_ORIGINAL_BLOCKS * TENSORS_PER_BLOCK + NUM_NON_BLOCK_TENSORS
EXPECTED_TOTAL_OUT = NUM_TARGET_BLOCKS * TENSORS_PER_BLOCK + NUM_NON_BLOCK_TENSORS

LAYER_RE = re.compile(r"^gpt_neox\.layers\.(\d+)\.")

# Build old->new block index mapping: surviving blocks in original order,
# renumbered contiguously from 0.
surviving_old_indices = [i for i in range(NUM_ORIGINAL_BLOCKS) if i not in REMOVE_BLOCKS]
assert len(surviving_old_indices) == NUM_TARGET_BLOCKS, "unexpected number of surviving blocks"
old_to_new = {old: new for new, old in enumerate(surviving_old_indices)}


def fail(msg):
    print(f"FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


def main():
    tensors = {}
    with safe_open(INPUT_PATH, framework="pt") as f:
        keys = list(f.keys())
        if len(keys) != EXPECTED_TOTAL_IN:
            fail(f"expected {EXPECTED_TOTAL_IN} input tensors, found {len(keys)}")
        for key in keys:
            tensors[key] = f.get_tensor(key)

    output = {}
    seen_new_indices = set()

    for key, tensor in tensors.items():
        m = LAYER_RE.match(key)
        if m is None:
            # non-block tensor: pass through unchanged
            output[key] = tensor
            continue

        old_idx = int(m.group(1))
        if old_idx in REMOVE_BLOCKS:
            continue

        if old_idx not in old_to_new:
            fail(f"block index {old_idx} out of expected range 0..{NUM_ORIGINAL_BLOCKS - 1}")

        new_idx = old_to_new[old_idx]
        new_key = f"gpt_neox.layers.{new_idx}." + key[m.end() :]

        if new_key in output:
            fail(f"collision writing {new_key} (from {key})")

        output[new_key] = tensor
        seen_new_indices.add(new_idx)

    # Required checks: fail loudly if anything is off.
    for key in output:
        m = LAYER_RE.match(key)
        if m is not None and int(m.group(1)) >= NUM_TARGET_BLOCKS:
            fail(f"output contains out-of-range block index: {key}")

    qkv_weight_count = sum(
        1 for key in output if re.match(r"^gpt_neox\.layers\.\d+\.attention\.query_key_value\.weight$", key)
    )
    if qkv_weight_count != NUM_TARGET_BLOCKS:
        fail(f"expected {NUM_TARGET_BLOCKS} blocks, found {qkv_weight_count}")

    if seen_new_indices != set(range(NUM_TARGET_BLOCKS)):
        fail(f"new block indices not contiguous 0..{NUM_TARGET_BLOCKS - 1}: {sorted(seen_new_indices)}")

    if len(output) != EXPECTED_TOTAL_OUT:
        fail(f"expected {EXPECTED_TOTAL_OUT} output tensors, found {len(output)}")

    non_block_keys = {
        "gpt_neox.embed_in.weight",
        "embed_out.weight",
        "gpt_neox.final_layer_norm.weight",
        "gpt_neox.final_layer_norm.bias",
    }
    for key in non_block_keys:
        if key not in output:
            fail(f"missing non-block tensor {key}")
        if not output[key].equal(tensors[key]):
            fail(f"non-block tensor {key} was modified")

    # Ensure all tensors are contiguous for safetensors.
    for key in list(output.keys()):
        output[key] = output[key].contiguous()

    save_file(output, OUTPUT_PATH)
    print(f"Wrote {OUTPUT_PATH} with {len(output)} tensors.")


if __name__ == "__main__":
    main()
