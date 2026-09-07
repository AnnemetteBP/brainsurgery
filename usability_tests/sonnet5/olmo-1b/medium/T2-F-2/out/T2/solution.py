"""
T2: structured attention-head pruning for OLMo-1B-0724-hf.

Removes head 5 (0-indexed) from every layer's q_proj, k_proj, v_proj (row
blocks) and o_proj (column blocks). All other tensors pass through unchanged.

Uses plain torch + safetensors (no HF `prune_heads`, since OLMo does not
implement the generic PreTrainedModel head-pruning hooks that method relies
on -- it is only wired up for a handful of encoder models). The head layout
(rows for q/k/v, columns for o) and block boundaries are exactly as given in
TASK.md, so a direct slice-and-checkpoint script is both simpler and more
auditable than forcing a generic tool through an unsupported path.
"""

import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

HEAD_DIM = 128
NUM_HEADS = 16
HIDDEN = 2048
PRUNE_HEAD = 5
NUM_LAYERS = 16

IN_DIR = Path(__file__).resolve().parents[2] / "inputs" / "base"
OUT_PATH = Path(__file__).resolve().parent / "model.safetensors"


def load_all_tensors(in_dir: Path) -> dict[str, torch.Tensor]:
    index_path = in_dir / "model.safetensors.index.json"
    with open(index_path) as f:
        index = json.load(f)
    weight_map: dict[str, str] = index["weight_map"]

    shard_files = sorted(set(weight_map.values()))
    shard_tensors: dict[str, dict[str, torch.Tensor]] = {}
    for shard in shard_files:
        tensors = {}
        with safe_open(in_dir / shard, framework="pt") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)
        shard_tensors[shard] = tensors

    result = {}
    for key, shard in weight_map.items():
        result[key] = shard_tensors[shard][key]
    assert len(result) == len(weight_map), "duplicate/missing keys while loading shards"
    return result


def row_keep_indices(prune_head: int) -> torch.Tensor:
    lo = prune_head * HEAD_DIM
    hi = lo + HEAD_DIM
    return torch.cat([torch.arange(0, lo), torch.arange(hi, NUM_HEADS * HEAD_DIM)])


def prune_layer(tensors: dict[str, torch.Tensor], layer_idx: int, keep_idx: torch.Tensor) -> None:
    prefix = f"model.layers.{layer_idx}.self_attn"
    for proj in ("q_proj", "k_proj", "v_proj"):
        key = f"{prefix}.{proj}.weight"
        w = tensors[key]
        assert w.shape == (HIDDEN, HIDDEN), f"unexpected shape for {key}: {tuple(w.shape)}"
        tensors[key] = w.index_select(0, keep_idx).contiguous()

    key = f"{prefix}.o_proj.weight"
    w = tensors[key]
    assert w.shape == (HIDDEN, HIDDEN), f"unexpected shape for {key}: {tuple(w.shape)}"
    tensors[key] = w.index_select(1, keep_idx).contiguous()


def main() -> None:
    tensors = load_all_tensors(IN_DIR)
    expected_total = len(tensors)
    assert expected_total == 114, f"expected 114 input tensors, found {expected_total}"

    keep_idx = row_keep_indices(PRUNE_HEAD)
    assert keep_idx.numel() == NUM_HEADS * HEAD_DIM - HEAD_DIM  # 1920

    for layer_idx in range(NUM_LAYERS):
        prune_layer(tensors, layer_idx, keep_idx)

    # Required checks: fail loudly before writing.
    checks = [
        ("model.layers.0.self_attn.q_proj.weight", (1920, 2048)),
        ("model.layers.0.self_attn.k_proj.weight", (1920, 2048)),
        ("model.layers.0.self_attn.v_proj.weight", (1920, 2048)),
        ("model.layers.0.self_attn.o_proj.weight", (2048, 1920)),
    ]
    for key, expected_shape in checks:
        actual = tuple(tensors[key].shape)
        if actual != expected_shape:
            print(f"FAIL: {key} has shape {actual}, expected {expected_shape}", file=sys.stderr)
            sys.exit(1)

    if len(tensors) != 114:
        print(f"FAIL: output has {len(tensors)} tensors, expected 114", file=sys.stderr)
        sys.exit(1)

    # Sanity check for a layer that isn't layer 0, and that untouched tensors
    # (MLP, norms, embeddings) kept their original shape.
    for layer_idx in range(NUM_LAYERS):
        prefix = f"model.layers.{layer_idx}.self_attn"
        assert tensors[f"{prefix}.q_proj.weight"].shape == (1920, 2048)
        assert tensors[f"{prefix}.k_proj.weight"].shape == (1920, 2048)
        assert tensors[f"{prefix}.v_proj.weight"].shape == (1920, 2048)
        assert tensors[f"{prefix}.o_proj.weight"].shape == (2048, 1920)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(OUT_PATH))
    print(f"Wrote {len(tensors)} tensors to {OUT_PATH}")


if __name__ == "__main__":
    main()
