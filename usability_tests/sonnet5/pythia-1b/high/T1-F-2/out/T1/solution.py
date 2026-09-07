"""T1: depth pruning with layer renumbering (Pythia-1B).

Removes transformer blocks {2, 6, 10, 14} from a 16-layer Pythia-1B
safetensors checkpoint and renumbers the surviving blocks to 0..11,
preserving order. Non-block tensors are copied unchanged.

Uses only `safetensors` + `re` — a plain rule-based key rewrite is simpler
and more auditable here than pulling in mergekit's layer-slicing YAML or
torch-state-bridge's regex-capture engine for a single, one-shot rename.

Fails loudly (raises / non-zero exit, no output written) if any required
check does not hold.
"""

import re
import sys
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file

DROP_BLOCKS = {2, 6, 10, 14}
NUM_ORIGINAL_BLOCKS = 16
TENSORS_PER_BLOCK = 15
NUM_NON_BLOCK_TENSORS = 4

LAYER_RE = re.compile(r"^gpt_neox\.layers\.(\d+)\.(.+)$")


def build_renumbering(num_original: int, drop: set[int]) -> dict[int, int]:
    """Map surviving old block indices -> new contiguous indices, in order."""
    survivors = [i for i in range(num_original) if i not in drop]
    return {old: new for new, old in enumerate(survivors)}


def main() -> None:
    in_path = Path("inputs/base/model.safetensors")
    out_path = Path("out/T1/model.safetensors")

    renumber = build_renumbering(NUM_ORIGINAL_BLOCKS, DROP_BLOCKS)
    expected_surviving = NUM_ORIGINAL_BLOCKS - len(DROP_BLOCKS)
    assert expected_surviving == 12, expected_surviving

    tensors: dict[str, "torch.Tensor"] = {}
    seen_old_blocks: set[int] = set()

    with safe_open(str(in_path), framework="pt") as f:
        keys = list(f.keys())
        for key in keys:
            m = LAYER_RE.match(key)
            if m is None:
                # Non-block tensor: copy unchanged.
                tensors[key] = f.get_tensor(key)
                continue

            old_idx = int(m.group(1))
            rest = m.group(2)
            seen_old_blocks.add(old_idx)

            if old_idx in DROP_BLOCKS:
                continue  # drop every tensor of this block

            new_idx = renumber[old_idx]
            new_key = f"gpt_neox.layers.{new_idx}.{rest}"
            if new_key in tensors:
                raise RuntimeError(
                    f"collision: {new_key} already produced "
                    f"(renumbering old block {old_idx} -> {new_idx})"
                )
            tensors[new_key] = f.get_tensor(key)

    # --- Required checks: fail loudly, write nothing on failure. ---

    if seen_old_blocks != set(range(NUM_ORIGINAL_BLOCKS)):
        raise RuntimeError(
            f"unexpected block indices in input: {sorted(seen_old_blocks)}"
        )

    for bad in (12, 13, 14, 15):
        prefix = f"gpt_neox.layers.{bad}."
        if any(k.startswith(prefix) for k in tensors):
            raise RuntimeError(f"tensor of dropped/out-of-range block {bad} remains")

    qkv_weight_keys = [
        k
        for k in tensors
        if re.match(r"^gpt_neox\.layers\.\d+\.attention\.query_key_value\.weight$", k)
    ]
    if len(qkv_weight_keys) != 12:
        raise RuntimeError(
            f"expected exactly 12 blocks, found {len(qkv_weight_keys)} "
            f"query_key_value.weight tensors"
        )

    expected_total = expected_surviving * TENSORS_PER_BLOCK + NUM_NON_BLOCK_TENSORS
    assert expected_total == 184, expected_total
    if len(tensors) != 184:
        raise RuntimeError(f"expected 184 tensors in output, got {len(tensors)}")

    # Non-block tensors present and untouched (identity copy, checked by key set).
    non_block_keys = {
        "gpt_neox.embed_in.weight",
        "embed_out.weight",
        "gpt_neox.final_layer_norm.weight",
        "gpt_neox.final_layer_norm.bias",
    }
    if not non_block_keys.issubset(tensors.keys()):
        missing = non_block_keys - tensors.keys()
        raise RuntimeError(f"missing non-block tensors: {missing}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(out_path))

    print(f"wrote {out_path}: {len(tensors)} tensors")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001 - top-level: report and fail loudly
        print(f"FAILED: {exc}", file=sys.stderr)
        sys.exit(1)
