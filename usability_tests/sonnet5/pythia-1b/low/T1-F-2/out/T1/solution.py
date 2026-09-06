"""T1: depth-prune Pythia-1B by dropping blocks 2, 6, 10, 14 and renumbering
the survivors to 0..11, contiguous, in original order.

Plain script on top of `safetensors` (load_file / save_file). No merge tool
is a better fit here: this is a pure key-rewrite + drop on a single
checkpoint, not a multi-model merge (mergekit) or a LoRA merge (peft), and
torch-state-bridge's regex-capture rules add no safety over an explicit
dict comprehension for a fixed 16->12 remap where collisions are trivial to
reason about directly. Fails loudly (raises / non-zero exit, no output
written) if any required invariant does not hold.
"""

import re
import sys
from pathlib import Path

from safetensors.torch import load_file, save_file

HERE = Path(__file__).resolve().parent
IN_PATH = HERE.parent.parent / "inputs" / "base" / "model.safetensors"
OUT_DIR = HERE
OUT_PATH = OUT_DIR / "model.safetensors"

DROP_BLOCKS = {2, 6, 10, 14}
N_ORIG = 16
N_TENSORS_PER_BLOCK = 15
N_NON_BLOCK = 4
N_EXPECTED_OUT = 184

LAYER_RE = re.compile(r"^gpt_neox\.layers\.(\d+)\.")


def build_remap(n_orig: int, drop: set[int]) -> dict[int, int]:
    survivors = [i for i in range(n_orig) if i not in drop]
    return {old: new for new, old in enumerate(survivors)}


def main() -> None:
    if not IN_PATH.is_file():
        sys.exit(f"input checkpoint not found: {IN_PATH}")

    tensors = load_file(str(IN_PATH))
    if len(tensors) != N_ORIG * N_TENSORS_PER_BLOCK + N_NON_BLOCK:
        sys.exit(
            f"unexpected input tensor count {len(tensors)}, "
            f"expected {N_ORIG * N_TENSORS_PER_BLOCK + N_NON_BLOCK}"
        )

    remap = build_remap(N_ORIG, DROP_BLOCKS)

    out: dict[str, object] = {}
    seen_new_names: set[str] = set()
    block_counts: dict[int, int] = {}

    for name, tensor in tensors.items():
        m = LAYER_RE.match(name)
        if m is None:
            # Non-block tensor: unchanged.
            out[name] = tensor
            continue
        old_idx = int(m.group(1))
        if old_idx in DROP_BLOCKS:
            continue
        new_idx = remap[old_idx]
        new_name = f"gpt_neox.layers.{new_idx}." + name[m.end() :]
        if new_name in seen_new_names:
            sys.exit(f"collision producing duplicate key: {new_name}")
        seen_new_names.add(new_name)
        block_counts[new_idx] = block_counts.get(new_idx, 0) + 1
        out[new_name] = tensor

    # --- Required checks: fail loudly, write nothing on failure. ---

    for bad_idx in (12, 13, 14, 15):
        if any(f"gpt_neox.layers.{bad_idx}." in k for k in out):
            sys.exit(f"tensor of dropped/out-of-range block {bad_idx} still present")

    qkv_weight_keys = [
        k for k in out if re.fullmatch(r"gpt_neox\.layers\.\d+\.attention\.query_key_value\.weight", k)
    ]
    if len(qkv_weight_keys) != 12:
        sys.exit(f"expected exactly 12 surviving blocks, found {len(qkv_weight_keys)}")

    if set(block_counts) != set(range(12)):
        sys.exit(f"surviving block indices are not exactly 0..11: {sorted(block_counts)}")
    for idx, count in block_counts.items():
        if count != N_TENSORS_PER_BLOCK:
            sys.exit(f"block {idx} has {count} tensors, expected {N_TENSORS_PER_BLOCK}")

    if len(out) != N_EXPECTED_OUT:
        sys.exit(f"output has {len(out)} tensors, expected {N_EXPECTED_OUT}")

    # Sanity: values/shapes/dtypes for survivors are the exact same tensor
    # objects as loaded from input, and non-block tensors are untouched.
    non_block_in = {k: v for k, v in tensors.items() if LAYER_RE.match(k) is None}
    non_block_out = {k: v for k, v in out.items() if LAYER_RE.match(k) is None}
    if set(non_block_in) != set(non_block_out):
        sys.exit("non-block tensor set changed")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    save_file(out, str(OUT_PATH))
    print(f"wrote {OUT_PATH} with {len(out)} tensors")


if __name__ == "__main__":
    main()
