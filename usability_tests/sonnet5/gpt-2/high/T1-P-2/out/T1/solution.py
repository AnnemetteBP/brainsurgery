"""
T1: Depth pruning with layer renumbering (GPT-2, 124M).

Remove transformer blocks 2, 5, 8 from a 12-layer GPT-2 checkpoint and
renumber the surviving blocks (0,1,3,4,6,7,9,10,11) to (0,1,2,3,4,5,6,7,8)
so indices are contiguous again. Non-block tensors are copied unchanged.

Fails loudly (raises / non-zero exit, no output written) if any required
check does not hold.
"""

import re
import sys
from pathlib import Path

from safetensors.torch import load_file, save_file

INPUT_PATH = Path("inputs/base/model.safetensors")
OUTPUT_DIR = Path("out/T1")
OUTPUT_PATH = OUTPUT_DIR / "model.safetensors"

DROP_BLOCKS = {2, 5, 8}
TOTAL_BLOCKS_IN = 12
EXPECTED_SURVIVING_BLOCKS = 9
EXPECTED_TOTAL_TENSORS_OUT = 121

# Matches tensor names like "h.<i>.<rest...>", capturing the block index.
BLOCK_RE = re.compile(r"^h\.(\d+)\.(.+)$")


def main() -> None:
    if not INPUT_PATH.is_file():
        raise FileNotFoundError(f"missing input checkpoint: {INPUT_PATH}")

    state_dict = load_file(str(INPUT_PATH))

    # Build old->new block index mapping: surviving blocks keep their
    # relative order, renumbered contiguously starting at 0.
    surviving_old_indices = sorted(
        i for i in range(TOTAL_BLOCKS_IN) if i not in DROP_BLOCKS
    )
    old_to_new = {old: new for new, old in enumerate(surviving_old_indices)}

    new_state_dict = {}
    non_block_count = 0
    for name, tensor in state_dict.items():
        m = BLOCK_RE.match(name)
        if m is None:
            # Non-block tensor (wte, wpe, ln_f.*): copy unchanged.
            if name in new_state_dict:
                raise RuntimeError(f"duplicate non-block tensor name: {name}")
            new_state_dict[name] = tensor
            non_block_count += 1
            continue

        old_idx = int(m.group(1))
        rest = m.group(2)
        if old_idx in DROP_BLOCKS:
            continue  # drop this tensor entirely

        new_idx = old_to_new[old_idx]
        new_name = f"h.{new_idx}.{rest}"
        if new_name in new_state_dict:
            raise RuntimeError(
                f"collision while renumbering: {name} -> {new_name} "
                f"already exists (renumbering order bug)"
            )
        new_state_dict[new_name] = tensor

    # --- Required checks: fail loudly, write nothing on failure. ---

    # No tensor of blocks 9, 10, 11 remains (the *dropped highest* old
    # indices among the ones that must not survive: the spec explicitly
    # calls out blocks 9, 10, 11 must not remain *under those indices*).
    for forbidden in (9, 10, 11):
        prefix = f"h.{forbidden}."
        leaked = [n for n in new_state_dict if n.startswith(prefix)]
        if leaked:
            raise RuntimeError(
                f"found {len(leaked)} tensor(s) still named under block "
                f"{forbidden} after renumbering: {leaked[:3]}"
            )

    # Exactly 9 blocks remain.
    c_attn_weights = [
        n for n in new_state_dict if re.match(r"^h\.\d+\.attn\.c_attn\.weight$", n)
    ]
    if len(c_attn_weights) != EXPECTED_SURVIVING_BLOCKS:
        raise RuntimeError(
            f"expected {EXPECTED_SURVIVING_BLOCKS} surviving blocks, found "
            f"{len(c_attn_weights)} (matched: {sorted(c_attn_weights)})"
        )
    got_indices = sorted(
        int(re.match(r"^h\.(\d+)\.", n).group(1)) for n in c_attn_weights
    )
    if got_indices != list(range(EXPECTED_SURVIVING_BLOCKS)):
        raise RuntimeError(
            f"surviving block indices are not contiguous 0..8: {got_indices}"
        )

    # Non-block tensor count unchanged (4 expected for GPT-2).
    if non_block_count != 4:
        raise RuntimeError(
            f"expected 4 non-block tensors, found {non_block_count}"
        )

    # Output has exactly 121 tensors.
    if len(new_state_dict) != EXPECTED_TOTAL_TENSORS_OUT:
        raise RuntimeError(
            f"expected {EXPECTED_TOTAL_TENSORS_OUT} output tensors, got "
            f"{len(new_state_dict)}"
        )

    # Sanity: every surviving tensor must be contiguous for safetensors.
    new_state_dict = {k: v.contiguous() for k, v in new_state_dict.items()}

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_file(new_state_dict, str(OUTPUT_PATH))

    print(f"wrote {len(new_state_dict)} tensors to {OUTPUT_PATH}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001 - top-level failure boundary
        print(f"FAILED: {exc}", file=sys.stderr)
        sys.exit(1)
