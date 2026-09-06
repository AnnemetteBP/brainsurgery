"""
Depth-prune OLMo-1B-0724-hf: drop transformer blocks 2, 6, 10, 14 and
renumber the survivors to 0..11, keeping everything else unchanged.
"""

import json
import re
import sys
from pathlib import Path

from safetensors.torch import load_file, save_file

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]  # out/T1 -> out -> sandbox root
INPUT_DIR = REPO_ROOT / "inputs" / "base"
OUTPUT_DIR = REPO_ROOT / "out" / "T1"
OUTPUT_FILE = OUTPUT_DIR / "model.safetensors"

DROP_BLOCKS = {2, 6, 10, 14}
NUM_ORIG_BLOCKS = 16
LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.")


def load_all_tensors(input_dir: Path) -> dict:
    index_path = input_dir / "model.safetensors.index.json"
    with open(index_path) as f:
        index = json.load(f)
    weight_map = index["weight_map"]

    shard_names = sorted(set(weight_map.values()))
    shard_tensors = {}
    for shard_name in shard_names:
        shard_tensors[shard_name] = load_file(str(input_dir / shard_name))

    tensors = {}
    for key, shard_name in weight_map.items():
        tensors[key] = shard_tensors[shard_name][key]
    return tensors


def build_renumber_map(drop_blocks: set, num_orig_blocks: int) -> dict:
    survivors = [i for i in range(num_orig_blocks) if i not in drop_blocks]
    return {old: new for new, old in enumerate(survivors)}


def main() -> None:
    tensors = load_all_tensors(INPUT_DIR)
    if len(tensors) != 114:
        print(f"FATAL: expected 114 input tensors, found {len(tensors)}", file=sys.stderr)
        sys.exit(1)

    renumber = build_renumber_map(DROP_BLOCKS, NUM_ORIG_BLOCKS)

    output_tensors = {}
    for key, tensor in tensors.items():
        m = LAYER_RE.match(key)
        if m is None:
            # Non-block tensor (embed_tokens, lm_head): unchanged.
            output_tensors[key] = tensor
            continue
        old_idx = int(m.group(1))
        if old_idx in DROP_BLOCKS:
            continue
        new_idx = renumber[old_idx]
        new_key = f"model.layers.{new_idx}." + key[m.end() :]
        if new_key in output_tensors:
            print(f"FATAL: key collision on {new_key}", file=sys.stderr)
            sys.exit(1)
        output_tensors[new_key] = tensor

    # --- Required checks: fail loudly, no output written on failure. ---

    for bad in (12, 13, 14, 15):
        pattern = re.compile(rf"^model\.layers\.{bad}\.")
        if any(pattern.match(k) for k in output_tensors):
            print(f"FATAL: tensor of block {bad} remains in output", file=sys.stderr)
            sys.exit(1)

    q_proj_pattern = re.compile(r"^model\.layers\.(\d+)\.self_attn\.q_proj\.weight$")
    q_proj_indices = sorted(
        int(m.group(1)) for k in output_tensors if (m := q_proj_pattern.match(k))
    )
    if q_proj_indices != list(range(12)):
        print(
            f"FATAL: expected q_proj blocks 0..11 contiguous, got {q_proj_indices}",
            file=sys.stderr,
        )
        sys.exit(1)

    if len(output_tensors) != 86:
        print(f"FATAL: expected 86 output tensors, got {len(output_tensors)}", file=sys.stderr)
        sys.exit(1)

    for key in ("model.embed_tokens.weight", "lm_head.weight"):
        if not torch_equal_shape_dtype(output_tensors[key], tensors[key]):
            print(f"FATAL: non-block tensor {key} was mutated", file=sys.stderr)
            sys.exit(1)
        if not (output_tensors[key] == tensors[key]).all():
            print(f"FATAL: non-block tensor {key} values changed", file=sys.stderr)
            sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_file(output_tensors, str(OUTPUT_FILE))
    print(f"Wrote {len(output_tensors)} tensors to {OUTPUT_FILE}")


def torch_equal_shape_dtype(a, b) -> bool:
    return a.shape == b.shape and a.dtype == b.dtype


if __name__ == "__main__":
    main()
