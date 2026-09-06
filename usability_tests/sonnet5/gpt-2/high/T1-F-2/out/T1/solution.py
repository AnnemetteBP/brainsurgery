#!/usr/bin/env python
"""T1: depth-prune GPT-2 (124M) from 12 to 9 transformer blocks.

Approach: plain script on top of `safetensors` (no torch-state-bridge or
mergekit needed here -- the rename is a single explicit dict comprehension
built key-by-key into a *new* dict, so there is no in-place rename and thus
no collision hazard by construction). All required checks are asserted
in-memory before anything is written to disk; any failure raises and leaves
`out/T1/` untouched.

Usage: python solution.py <input_dir> <output_dir>
"""

import re
import sys
from pathlib import Path

from safetensors.torch import load_file, save_file

DROP_BLOCKS = {2, 5, 8}
# old block index -> new block index, in original order, gaps closed.
REMAP = {0: 0, 1: 1, 3: 2, 4: 3, 6: 4, 7: 5, 9: 6, 10: 7, 11: 8}

BLOCK_KEY_RE = re.compile(r"^h\.(\d+)\.(.+)$")
TENSORS_PER_BLOCK = 13
N_KEPT_BLOCKS = 9
N_NON_BLOCK = 4
EXPECTED_TOTAL = N_KEPT_BLOCKS * TENSORS_PER_BLOCK + N_NON_BLOCK  # 121


def main() -> None:
    in_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("inputs/base")
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("out/T1")

    src_path = in_dir / "model.safetensors"
    state = load_file(str(src_path))
    print(f"loaded {len(state)} tensors from {src_path}")

    new_state = {}
    for key, tensor in state.items():
        m = BLOCK_KEY_RE.match(key)
        if m is None:
            # non-block tensor: wte.weight, wpe.weight, ln_f.weight, ln_f.bias
            new_state[key] = tensor
            continue
        old_idx, rest = int(m.group(1)), m.group(2)
        if old_idx in DROP_BLOCKS:
            continue
        new_idx = REMAP[old_idx]
        new_key = f"h.{new_idx}.{rest}"
        if new_key in new_state:
            raise RuntimeError(f"collision: {new_key} already produced (from old block {old_idx})")
        new_state[new_key] = tensor

    # --- required checks: fail loudly, write nothing on failure ---

    for bad in (9, 10, 11):
        leftover = [k for k in new_state if k.startswith(f"h.{bad}.")]
        assert not leftover, f"tensors of dropped-index block {bad} remain: {leftover}"
        # (9, 10, 11 only exist as *old* indices that got renumbered away or
        # were never dropped-by-number; the real requirement is that no
        # tensor from *old* blocks 9, 10, 11 was left at its old index)

    n_blocks = len({int(m.group(1)) for k in new_state if (m := BLOCK_KEY_RE.match(k))})
    assert n_blocks == N_KEPT_BLOCKS, f"expected {N_KEPT_BLOCKS} blocks, found {n_blocks}"

    n_c_attn = sum(1 for k in new_state if re.match(r"^h\.\d+\.attn\.c_attn\.weight$", k))
    assert n_c_attn == N_KEPT_BLOCKS, f"expected {N_KEPT_BLOCKS} c_attn.weight tensors, found {n_c_attn}"

    assert len(new_state) == EXPECTED_TOTAL, (
        f"expected {EXPECTED_TOTAL} tensors in output, got {len(new_state)}"
    )

    for name in ("wte.weight", "wpe.weight", "ln_f.weight", "ln_f.bias"):
        assert name in new_state, f"missing non-block tensor {name}"
        assert state[name].equal(new_state[name]), f"{name} was modified"

    # sanity: renumbered tensors are bit-identical to their source
    for old_idx, new_idx in REMAP.items():
        for k_old, v_old in state.items():
            m = BLOCK_KEY_RE.match(k_old)
            if m and int(m.group(1)) == old_idx:
                k_new = f"h.{new_idx}.{m.group(2)}"
                assert new_state[k_new].equal(v_old), f"{k_new} does not match source {k_old}"

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "model.safetensors"
    # ensure contiguous tensors (safetensors requires this, and rejects
    # tensors that alias the same storage under different keys)
    new_state = {k: v.contiguous() for k, v in new_state.items()}
    save_file(new_state, str(out_path))
    print(f"wrote {len(new_state)} tensors to {out_path}")


if __name__ == "__main__":
    main()
