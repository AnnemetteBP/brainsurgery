"""
T1: Depth pruning with layer renumbering (GPT-2, 124M).

Plain script on top of `safetensors` (no need for mergekit/torch-state-bridge
machinery for a task this small and exact-value-sensitive). Removes blocks
2, 5, 8 and renumbers survivors 0..8 preserving order, leaving the 4
non-block tensors untouched. Fails loudly (non-zero exit, no output written)
if any required check does not hold.
"""

import re
import sys
from pathlib import Path

from safetensors.torch import load_file, save_file

DROP_BLOCKS = {2, 5, 8}
NUM_ORIG_BLOCKS = 12
BLOCK_RE = re.compile(r"^h\.(\d+)\.")

IN_PATH = Path("inputs/base/model.safetensors")
OUT_PATH = Path("out/T1/model.safetensors")


def main() -> None:
    state = load_file(str(IN_PATH))

    surviving_old_indices = [i for i in range(NUM_ORIG_BLOCKS) if i not in DROP_BLOCKS]
    old_to_new = {old: new for new, old in enumerate(surviving_old_indices)}

    new_state = {}
    for name, tensor in state.items():
        m = BLOCK_RE.match(name)
        if m is None:
            # non-block tensor: unchanged
            new_state[name] = tensor
            continue
        old_idx = int(m.group(1))
        if old_idx in DROP_BLOCKS:
            continue
        new_idx = old_to_new[old_idx]
        new_name = f"h.{new_idx}.{name[m.end():]}"
        if new_name in new_state:
            raise RuntimeError(f"collision: {new_name} already assigned (from {name})")
        new_state[new_name] = tensor

    # --- Required checks: fail loudly, write nothing on failure ---

    for bad in (9, 10, 11):
        for name in new_state:
            m = BLOCK_RE.match(name)
            if m and int(m.group(1)) == bad:
                raise RuntimeError(f"tensor of dropped-index block {bad} remains: {name}")

    c_attn_weights = [n for n in new_state if re.match(r"^h\.\d+\.attn\.c_attn\.weight$", n)]
    if len(c_attn_weights) != 9:
        raise RuntimeError(
            f"expected exactly 9 surviving blocks, found {len(c_attn_weights)}: {c_attn_weights}"
        )
    block_indices = sorted(int(BLOCK_RE.match(n).group(1)) for n in c_attn_weights)
    if block_indices != list(range(9)):
        raise RuntimeError(f"surviving block indices not contiguous 0..8: {block_indices}")

    if len(new_state) != 121:
        raise RuntimeError(f"expected exactly 121 tensors, got {len(new_state)}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_file(new_state, str(OUT_PATH))
    print(f"wrote {OUT_PATH} with {len(new_state)} tensors")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"FAILED: {exc}", file=sys.stderr)
        sys.exit(1)
