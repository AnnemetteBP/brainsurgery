"""
T1: Depth pruning with layer renumbering (GPT-2 124M).

Remove transformer blocks 2, 5, 8 and renumber the remaining blocks so
that indices run 0..8 without gaps, preserving original order:

  old 0 -> 0, old 1 -> 1, old 3 -> 2, old 4 -> 3, old 6 -> 4,
  old 7 -> 5, old 9 -> 6, old 10 -> 7, old 11 -> 8

All other tensors are copied unchanged. Fails loudly if any required
check does not hold.
"""

import re
import sys

from safetensors.torch import load_file, save_file

INPUT_PATH = "inputs/base/model.safetensors"
OUTPUT_PATH = "out/T1/model.safetensors"

DROP_BLOCKS = {2, 5, 8}
NUM_ORIGINAL_BLOCKS = 12
EXPECTED_INPUT_TENSORS = 160
EXPECTED_OUTPUT_TENSORS = 121
TENSORS_PER_BLOCK = 13

BLOCK_KEY_RE = re.compile(r"^h\.(\d+)\.(.*)$")


def main() -> None:
    state_dict = load_file(INPUT_PATH)

    if len(state_dict) != EXPECTED_INPUT_TENSORS:
        sys.exit(
            f"expected {EXPECTED_INPUT_TENSORS} input tensors, got {len(state_dict)}"
        )

    # Build the old->new block index mapping for surviving blocks, in order.
    surviving_old_indices = [
        i for i in range(NUM_ORIGINAL_BLOCKS) if i not in DROP_BLOCKS
    ]
    old_to_new = {old: new for new, old in enumerate(surviving_old_indices)}

    output_state_dict = {}
    for key, tensor in state_dict.items():
        match = BLOCK_KEY_RE.match(key)
        if match is None:
            # Non-block tensor: copy unchanged.
            output_state_dict[key] = tensor
            continue

        old_index = int(match.group(1))
        rest = match.group(2)
        if old_index in DROP_BLOCKS:
            continue

        new_index = old_to_new[old_index]
        new_key = f"h.{new_index}.{rest}"
        if new_key in output_state_dict:
            sys.exit(f"collision: {new_key} already exists (from key {key})")
        output_state_dict[new_key] = tensor

    # --- Required checks ---

    # No tensor of dropped-and-shifted-away old blocks 9, 10, 11 remains
    # under their *original* indices (they should have been renumbered
    # away, since they survive but move to 6, 7, 8).
    for banned_old_index in (9, 10, 11):
        banned_prefix = f"h.{banned_old_index}."
        if any(k.startswith(banned_prefix) for k in output_state_dict):
            sys.exit(f"found leftover tensor with old block index {banned_old_index}")

    # Exactly 9 blocks remain.
    attn_weight_keys = [
        k for k in output_state_dict if re.match(r"^h\.\d+\.attn\.c_attn\.weight$", k)
    ]
    if len(attn_weight_keys) != 9:
        sys.exit(f"expected 9 surviving blocks, found {len(attn_weight_keys)}")

    block_indices_present = sorted(
        int(re.match(r"^h\.(\d+)\.", k).group(1)) for k in attn_weight_keys
    )
    if block_indices_present != list(range(9)):
        sys.exit(f"block indices are not contiguous 0..8: {block_indices_present}")

    # Total tensor count.
    if len(output_state_dict) != EXPECTED_OUTPUT_TENSORS:
        sys.exit(
            f"expected {EXPECTED_OUTPUT_TENSORS} output tensors, "
            f"got {len(output_state_dict)}"
        )

    # Sanity: every surviving block has exactly TENSORS_PER_BLOCK tensors.
    for new_index in range(9):
        prefix = f"h.{new_index}."
        count = sum(1 for k in output_state_dict if k.startswith(prefix))
        if count != TENSORS_PER_BLOCK:
            sys.exit(
                f"block {new_index} has {count} tensors, expected {TENSORS_PER_BLOCK}"
            )

    save_file(output_state_dict, OUTPUT_PATH)
    print(f"wrote {OUTPUT_PATH} with {len(output_state_dict)} tensors")


if __name__ == "__main__":
    main()
