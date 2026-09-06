#!/usr/bin/env python3
"""
T1: Depth pruning with layer renumbering (OLMo-1B-0724-hf).

Remove transformer blocks {2, 6, 10, 14} from the 16-layer checkpoint and
renumber the surviving 12 blocks to 0..11 in original order, leaving the two
non-block tensors untouched. Writes out/T1/model.safetensors (single file).

Approach: load the full sharded safetensors checkpoint into memory (plain
tensors, via safetensors.torch), build a brand-new dict keyed by the target
names (so there is no in-place rename and therefore no collision hazard: old
and new key spaces are never mixed), then run required checks before saving.
Fails loudly (raises, no output written) if any check does not hold.
"""

import json
import re
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

REMOVE = {2, 6, 10, 14}
INPUT_DIR = Path(__file__).resolve().parents[2] / "inputs" / "base"
OUTPUT_PATH = Path(__file__).resolve().parent / "model.safetensors"

LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.(.+)$")


def load_sharded_state_dict(input_dir: Path) -> dict[str, torch.Tensor]:
    index_path = input_dir / "model.safetensors.index.json"
    with open(index_path) as f:
        index = json.load(f)
    shard_names = sorted(set(index["weight_map"].values()))
    state_dict: dict[str, torch.Tensor] = {}
    for shard_name in shard_names:
        shard = load_file(input_dir / shard_name)
        overlap = set(shard) & set(state_dict)
        if overlap:
            raise RuntimeError(f"duplicate keys across shards: {overlap}")
        state_dict.update(shard)
    expected_keys = set(index["weight_map"].keys())
    if set(state_dict.keys()) != expected_keys:
        raise RuntimeError("loaded keys do not match index's weight_map")
    return state_dict


def build_layer_mapping(old_indices: list[int]) -> dict[int, int]:
    surviving = sorted(i for i in old_indices if i not in REMOVE)
    return {old: new for new, old in enumerate(surviving)}


def main() -> None:
    state_dict = load_sharded_state_dict(INPUT_DIR)

    old_layer_indices = set()
    non_layer_keys = []
    for key in state_dict:
        m = LAYER_RE.match(key)
        if m:
            old_layer_indices.add(int(m.group(1)))
        else:
            non_layer_keys.append(key)

    mapping = build_layer_mapping(sorted(old_layer_indices))

    new_state_dict: dict[str, torch.Tensor] = {}

    # Non-block tensors pass through unchanged.
    for key in non_layer_keys:
        new_state_dict[key] = state_dict[key]

    # Renumber surviving blocks into a fresh key space -- no in-place
    # renames, so old/new indices never collide.
    for key, tensor in state_dict.items():
        m = LAYER_RE.match(key)
        if not m:
            continue
        old_idx = int(m.group(1))
        if old_idx not in mapping:
            continue  # removed block
        new_idx = mapping[old_idx]
        new_key = f"model.layers.{new_idx}.{m.group(2)}"
        if new_key in new_state_dict:
            raise RuntimeError(f"collision writing {new_key}")
        new_state_dict[new_key] = tensor

    # --- Required checks: fail loudly, write nothing on failure. ---

    for removed in (12, 13, 14, 15):
        pattern = re.compile(rf"^model\.layers\.{removed}\.")
        if any(pattern.match(k) for k in new_state_dict):
            raise RuntimeError(f"tensor of removed block {removed} still present")

    q_proj_pattern = re.compile(r"^model\.layers\.(\d+)\.self_attn\.q_proj\.weight$")
    q_proj_indices = sorted(
        int(m.group(1)) for k in new_state_dict if (m := q_proj_pattern.match(k))
    )
    if q_proj_indices != list(range(12)):
        raise RuntimeError(
            f"expected exactly 12 contiguous blocks 0..11, got q_proj indices {q_proj_indices}"
        )

    if len(new_state_dict) != 86:
        raise RuntimeError(f"expected exactly 86 tensors, got {len(new_state_dict)}")

    # Non-block tensor sanity: unchanged identity/shape/dtype.
    for key in non_layer_keys:
        if new_state_dict[key].shape != state_dict[key].shape:
            raise RuntimeError(f"non-block tensor {key} shape changed")
        if new_state_dict[key].dtype != state_dict[key].dtype:
            raise RuntimeError(f"non-block tensor {key} dtype changed")

    # safetensors requires contiguous, non-shared storage per tensor.
    new_state_dict = {k: v.contiguous().clone() for k, v in new_state_dict.items()}

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_file(new_state_dict, str(OUTPUT_PATH))

    print(f"wrote {OUTPUT_PATH} with {len(new_state_dict)} tensors")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # fail loudly, no partial output
        if OUTPUT_PATH.exists():
            OUTPUT_PATH.unlink()
        print(f"FAILED: {exc}", file=sys.stderr)
        sys.exit(1)
