"""
T1: Depth pruning with layer renumbering (OLMo-1B-0724-hf).

Removes transformer blocks 2, 6, 10, 14 from a 16-layer sharded safetensors
checkpoint and renumbers the surviving blocks to 0..11 (order-preserving),
writing a single merged model.safetensors. Uses only `safetensors` + `json`
(stdlib) directly on the state dict: the simplest, most auditable route for a
pure rename/drop/repack operation, avoiding the extra indirection of
mergekit's YAML slicing config or torch-state-bridge's rule engine for a
transform this small.

Fails loudly (raises / non-zero exit, no output written) if any required
check does not hold.
"""

import json
import re
import sys
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file

INPUT_DIR = Path(__file__).resolve().parents[2] / "inputs" / "base"
OUTPUT_PATH = Path(__file__).resolve().parents[0] / "model.safetensors"

DROP_BLOCKS = {2, 6, 10, 14}
# Old index -> new index, for surviving blocks, order-preserving.
SURVIVORS = [i for i in range(16) if i not in DROP_BLOCKS]
assert len(SURVIVORS) == 12
OLD_TO_NEW = {old: new for new, old in enumerate(SURVIVORS)}

LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.(.+)$")


def load_state_dict(input_dir: Path) -> dict:
    index_path = input_dir / "model.safetensors.index.json"
    with open(index_path) as f:
        index = json.load(f)
    weight_map = index["weight_map"]

    shard_names = sorted(set(weight_map.values()))
    tensors = {}
    for shard_name in shard_names:
        shard_path = input_dir / shard_name
        with safe_open(shard_path, framework="pt") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)

    if set(tensors.keys()) != set(weight_map.keys()):
        raise RuntimeError(
            "Tensor keys loaded from shards do not match the index's weight_map; "
            f"missing={set(weight_map) - set(tensors)}, "
            f"extra={set(tensors) - set(weight_map)}"
        )
    return tensors


def rename_key(key: str) -> str | None:
    """Return the renamed key, or None if this tensor belongs to a dropped block."""
    m = LAYER_RE.match(key)
    if m is None:
        return key  # non-block tensor, unchanged
    old_idx = int(m.group(1))
    rest = m.group(2)
    if old_idx in DROP_BLOCKS:
        return None
    new_idx = OLD_TO_NEW[old_idx]
    return f"model.layers.{new_idx}.{rest}"


def main() -> None:
    state_dict = load_state_dict(INPUT_DIR)

    out = {}
    collisions = {}
    for key, tensor in state_dict.items():
        new_key = rename_key(key)
        if new_key is None:
            continue
        if new_key in out:
            collisions.setdefault(new_key, []).append(key)
        out[new_key] = tensor

    if collisions:
        raise RuntimeError(f"Renumbering produced key collisions: {collisions}")

    # --- Required checks: fail loudly, write nothing on failure. ---

    # No tensor of blocks 12, 13, 14, 15 remains.
    for i in (12, 13, 14, 15):
        bad = [k for k in out if k.startswith(f"model.layers.{i}.")]
        if bad:
            raise RuntimeError(f"Found leftover tensors for dropped/out-of-range block {i}: {bad}")

    # Exactly 12 blocks remain.
    block_indices = set()
    for k in out:
        m = LAYER_RE.match(k)
        if m:
            block_indices.add(int(m.group(1)))
    if block_indices != set(range(12)):
        raise RuntimeError(f"Expected block indices 0..11, got {sorted(block_indices)}")

    q_proj_count = sum(1 for k in out if re.fullmatch(r"model\.layers\.\d+\.self_attn\.q_proj\.weight", k))
    if q_proj_count != 12:
        raise RuntimeError(f"Expected exactly 12 q_proj tensors, found {q_proj_count}")

    # Non-block tensors unchanged in count/name.
    non_block_keys = {k for k in out if LAYER_RE.match(k) is None}
    expected_non_block = {"model.embed_tokens.weight", "lm_head.weight"}
    if non_block_keys != expected_non_block:
        raise RuntimeError(f"Non-block tensors mismatch: {non_block_keys}")

    # Exactly 86 tensors total.
    if len(out) != 86:
        raise RuntimeError(f"Expected exactly 86 tensors in output, got {len(out)}")

    out = {k: v.contiguous() for k, v in out.items()}

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_file(out, str(OUTPUT_PATH), metadata={"format": "pt"})

    print(f"Wrote {len(out)} tensors to {OUTPUT_PATH}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"FAILED: {e}", file=sys.stderr)
        if OUTPUT_PATH.exists():
            OUTPUT_PATH.unlink()
        sys.exit(1)
