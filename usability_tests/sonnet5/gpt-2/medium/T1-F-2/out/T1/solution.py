"""T1: depth-prune GPT-2 by dropping blocks 2, 5, 8 and renumbering the rest
so indices run 0..8 contiguously.

Approach: plain script on top of `safetensors` (load_file / save_file) plus
`re` for key parsing. mergekit's layer-slicing passthrough config only
supports contiguous layer ranges, not "drop 2, 5, 8 and keep the rest"; a
few lines of direct key rewriting is simpler and easier to verify than
fighting a YAML slicing DSL into doing a non-contiguous selection. Building a
brand-new dict (rather than renaming keys in place) sidesteps the
old-index -> new-index collision hazard entirely, since every new key is
written into a fresh dict exactly once.
"""

import re
import sys
from pathlib import Path

from safetensors.torch import load_file, save_file

DROP = {2, 5, 8}
N_LAYERS_IN = 12

IN_PATH = Path("inputs/base/model.safetensors")
OUT_PATH = Path("out/T1/model.safetensors")

BLOCK_RE = re.compile(r"^h\.(\d+)\.")


def main() -> None:
    state = load_file(str(IN_PATH))

    # Old surviving indices in original order -> contiguous new indices.
    surviving_old = [i for i in range(N_LAYERS_IN) if i not in DROP]
    remap = {old: new for new, old in enumerate(surviving_old)}

    new_state = {}
    n_block_tensors = 0
    n_other_tensors = 0
    for key, tensor in state.items():
        m = BLOCK_RE.match(key)
        if m is None:
            new_state[key] = tensor
            n_other_tensors += 1
            continue
        old_idx = int(m.group(1))
        if old_idx in DROP:
            continue
        new_idx = remap[old_idx]
        new_key = f"h.{new_idx}." + key[m.end() :]
        if new_key in new_state:
            raise RuntimeError(f"collision: {new_key} already written (from {key})")
        new_state[new_key] = tensor
        n_block_tensors += 1

    # --- Required checks: fail loudly, no output written, on any violation ---
    for dropped in (9, 10, 11):
        if any(re.match(rf"^h\.{dropped}\.", k) for k in new_state):
            raise AssertionError(f"tensor of dropped block {dropped} present in output")

    n_surviving_blocks = sum(
        1 for k in new_state if re.match(r"^h\.\d+\.attn\.c_attn\.weight$", k)
    )
    if n_surviving_blocks != 9:
        raise AssertionError(f"expected exactly 9 surviving blocks, got {n_surviving_blocks}")

    if len(new_state) != 121:
        raise AssertionError(f"expected exactly 121 tensors in output, got {len(new_state)}")

    if n_other_tensors != 4:
        raise AssertionError(f"expected exactly 4 non-block tensors, got {n_other_tensors}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_file(new_state, str(OUT_PATH))
    print(f"wrote {OUT_PATH} with {len(new_state)} tensors")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # fail loudly, non-zero exit, no partial output
        print(f"FAILED: {exc}", file=sys.stderr)
        if OUT_PATH.exists():
            OUT_PATH.unlink()
        sys.exit(1)
