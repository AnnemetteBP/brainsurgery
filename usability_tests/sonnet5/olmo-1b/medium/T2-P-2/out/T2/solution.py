"""Remove attention head 5 from every layer of OLMo-1B-0724-hf.

Reads the sharded safetensors checkpoint at inputs/base, slices out head 5
from the q/k/v/o attention projections of every layer, and writes a single
model.safetensors file with all 114 tensors (unchanged tensors copied as-is).
"""

import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

HERE = Path(__file__).resolve().parent
IN_DIR = HERE.parent.parent / "inputs" / "base"
OUT_PATH = HERE / "model.safetensors"

NUM_LAYERS = 16
NUM_HEADS = 16
HEAD_DIM = 128
HIDDEN = 2048
HEAD_TO_REMOVE = 5

QKV_NAMES = ["q_proj", "k_proj", "v_proj"]


def load_all_tensors(in_dir: Path) -> dict[str, torch.Tensor]:
    index_path = in_dir / "model.safetensors.index.json"
    with open(index_path) as f:
        index = json.load(f)
    weight_map = index["weight_map"]

    shard_files = sorted(set(weight_map.values()))
    tensors: dict[str, torch.Tensor] = {}
    for shard in shard_files:
        with safe_open(in_dir / shard, framework="pt") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)

    assert set(tensors.keys()) == set(weight_map.keys()), (
        "tensor keys loaded from shards do not match the index's weight_map"
    )
    return tensors


def row_keep_indices(head_to_remove: int) -> torch.Tensor:
    """Rows to keep for a row-block-per-head tensor (q/k/v): drop head's row block."""
    start = head_to_remove * HEAD_DIM
    end = start + HEAD_DIM
    keep = list(range(0, start)) + list(range(end, NUM_HEADS * HEAD_DIM))
    return torch.tensor(keep, dtype=torch.long)


def col_keep_indices(head_to_remove: int) -> torch.Tensor:
    """Columns to keep for a column-block-per-head tensor (o_proj): drop head's column block."""
    return row_keep_indices(head_to_remove)


def main() -> None:
    tensors = load_all_tensors(IN_DIR)

    row_keep = row_keep_indices(HEAD_TO_REMOVE)
    col_keep = col_keep_indices(HEAD_TO_REMOVE)

    for i in range(NUM_LAYERS):
        for name in QKV_NAMES:
            key = f"model.layers.{i}.self_attn.{name}.weight"
            orig = tensors[key]
            assert orig.shape == (HIDDEN, HIDDEN), f"{key} has unexpected shape {tuple(orig.shape)}"
            tensors[key] = orig[row_keep, :].contiguous()

        o_key = f"model.layers.{i}.self_attn.o_proj.weight"
        orig_o = tensors[o_key]
        assert orig_o.shape == (HIDDEN, HIDDEN), f"{o_key} has unexpected shape {tuple(orig_o.shape)}"
        tensors[o_key] = orig_o[:, col_keep].contiguous()

    # Required checks: fail loudly before writing if anything is off.
    expected_qkv_shape = (HIDDEN - HEAD_DIM, HIDDEN)
    expected_o_shape = (HIDDEN, HIDDEN - HEAD_DIM)

    for name in QKV_NAMES:
        key = f"model.layers.0.self_attn.{name}.weight"
        assert tensors[key].shape == expected_qkv_shape, (
            f"{key} has shape {tuple(tensors[key].shape)}, expected {expected_qkv_shape}"
        )
    o_key0 = "model.layers.0.self_attn.o_proj.weight"
    assert tensors[o_key0].shape == expected_o_shape, (
        f"{o_key0} has shape {tuple(tensors[o_key0].shape)}, expected {expected_o_shape}"
    )
    assert len(tensors) == 114, f"expected 114 tensors, got {len(tensors)}"

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(OUT_PATH))
    print(f"Wrote {len(tensors)} tensors to {OUT_PATH}")


if __name__ == "__main__":
    main()
