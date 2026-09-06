"""
T1: Depth pruning with layer renumbering (Pythia-1B).

Remove transformer blocks 2, 6, 10, 14 from a 16-layer Pythia-1B checkpoint
and renumber the surviving blocks so indices are contiguous (0..11).

Old -> new block index mapping (order preserved, removed blocks dropped):
  0->0, 1->1, 3->2, 4->3, 5->4, 7->5, 8->6, 9->7, 11->8, 12->9, 13->10, 15->11
"""

import os
import re
import sys

from safetensors.torch import load_file, save_file

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH = os.path.join(REPO_ROOT, "..", "..", "inputs", "base", "model.safetensors")
OUTPUT_DIR = os.path.join(REPO_ROOT)
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "model.safetensors")

REMOVED_BLOCKS = {2, 6, 10, 14}
NUM_ORIGINAL_BLOCKS = 16
TENSORS_PER_BLOCK = 15
NUM_NON_BLOCK_TENSORS = 4
EXPECTED_TOTAL_INPUT = NUM_ORIGINAL_BLOCKS * TENSORS_PER_BLOCK + NUM_NON_BLOCK_TENSORS
EXPECTED_SURVIVING_BLOCKS = NUM_ORIGINAL_BLOCKS - len(REMOVED_BLOCKS)
EXPECTED_TOTAL_OUTPUT = EXPECTED_SURVIVING_BLOCKS * TENSORS_PER_BLOCK + NUM_NON_BLOCK_TENSORS

LAYER_RE = re.compile(r"^gpt_neox\.layers\.(\d+)\.")


def fail(msg: str) -> None:
    print(f"FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    if not os.path.isfile(INPUT_PATH):
        fail(f"input checkpoint not found at {INPUT_PATH}")

    state_dict = load_file(INPUT_PATH)

    if len(state_dict) != EXPECTED_TOTAL_INPUT:
        fail(
            f"expected {EXPECTED_TOTAL_INPUT} input tensors, found {len(state_dict)}"
        )

    # Build the old -> new block index mapping: keep original order, drop
    # removed blocks, renumber the rest contiguously from 0.
    surviving_old_indices = [
        i for i in range(NUM_ORIGINAL_BLOCKS) if i not in REMOVED_BLOCKS
    ]
    if len(surviving_old_indices) != EXPECTED_SURVIVING_BLOCKS:
        fail(
            f"expected {EXPECTED_SURVIVING_BLOCKS} surviving blocks, "
            f"computed {len(surviving_old_indices)}"
        )
    old_to_new = {old: new for new, old in enumerate(surviving_old_indices)}

    new_state_dict = {}
    per_block_count = {}

    for name, tensor in state_dict.items():
        match = LAYER_RE.match(name)
        if match is None:
            # Non-block tensor: passes through unchanged.
            if name in new_state_dict:
                fail(f"duplicate non-block tensor name: {name}")
            new_state_dict[name] = tensor
            continue

        old_idx = int(match.group(1))
        if old_idx in REMOVED_BLOCKS:
            continue  # dropped

        if old_idx not in old_to_new:
            fail(f"unexpected block index {old_idx} not in 0..{NUM_ORIGINAL_BLOCKS - 1}")

        new_idx = old_to_new[old_idx]
        rest = name[match.end():]
        new_name = f"gpt_neox.layers.{new_idx}.{rest}"

        if new_name in new_state_dict:
            fail(f"collision: {new_name} already exists (from old block {old_idx})")

        new_state_dict[new_name] = tensor
        per_block_count[new_idx] = per_block_count.get(new_idx, 0) + 1

    # --- Required checks: fail loudly, write nothing, if any do not hold ---

    # No tensor of the removed blocks' old indices (12, 13, 14, 15 per the
    # spec's explicit check, and generally any removed index) remains.
    for name in new_state_dict:
        match = LAYER_RE.match(name)
        if match is None:
            continue
        idx = int(match.group(1))
        if idx not in range(EXPECTED_SURVIVING_BLOCKS):
            fail(f"tensor {name} has out-of-range block index {idx}")

    qkv_weight_count = sum(
        1
        for name in new_state_dict
        if re.match(r"^gpt_neox\.layers\.\d+\.attention\.query_key_value\.weight$", name)
    )
    if qkv_weight_count != EXPECTED_SURVIVING_BLOCKS:
        fail(
            f"expected exactly {EXPECTED_SURVIVING_BLOCKS} blocks "
            f"(query_key_value.weight count), found {qkv_weight_count}"
        )

    for new_idx, count in per_block_count.items():
        if count != TENSORS_PER_BLOCK:
            fail(
                f"block {new_idx} has {count} tensors, expected {TENSORS_PER_BLOCK}"
            )

    if len(new_state_dict) != EXPECTED_TOTAL_OUTPUT:
        fail(
            f"expected {EXPECTED_TOTAL_OUTPUT} output tensors, "
            f"got {len(new_state_dict)}"
        )

    # Verify the 4 non-block tensors are present and unchanged (identity, we
    # never touched them, but check they made it through untouched by name).
    expected_non_block = {
        "gpt_neox.embed_in.weight",
        "embed_out.weight",
        "gpt_neox.final_layer_norm.weight",
        "gpt_neox.final_layer_norm.bias",
    }
    for name in expected_non_block:
        if name not in new_state_dict:
            fail(f"missing non-block tensor: {name}")
        if not new_state_dict[name].equal(state_dict[name]):
            fail(f"non-block tensor {name} was modified")

    # Spot-check value/shape/dtype preservation for a renumbered tensor.
    for old_idx, new_idx in old_to_new.items():
        old_name = f"gpt_neox.layers.{old_idx}.attention.query_key_value.weight"
        new_name = f"gpt_neox.layers.{new_idx}.attention.query_key_value.weight"
        old_tensor = state_dict[old_name]
        new_tensor = new_state_dict[new_name]
        if old_tensor.shape != new_tensor.shape:
            fail(f"shape mismatch for {old_name} -> {new_name}")
        if old_tensor.dtype != new_tensor.dtype:
            fail(f"dtype mismatch for {old_name} -> {new_name}")
        if not old_tensor.equal(new_tensor):
            fail(f"value mismatch for {old_name} -> {new_name}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # Tensors from load_file are already contiguous, individually-owned
    # tensors (safetensors does not create shared storage on load), so a
    # plain save_file is safe here.
    save_file(new_state_dict, OUTPUT_PATH, metadata={"format": "pt"})

    # Final sanity check: reload and verify tensor count on disk.
    reloaded = load_file(OUTPUT_PATH)
    if len(reloaded) != EXPECTED_TOTAL_OUTPUT:
        fail(
            f"output file has {len(reloaded)} tensors on disk, "
            f"expected {EXPECTED_TOTAL_OUTPUT}"
        )

    print(f"OK: wrote {len(new_state_dict)} tensors to {OUTPUT_PATH}")
    print(f"Block mapping (old -> new): {old_to_new}")


if __name__ == "__main__":
    main()
