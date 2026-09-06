"""Structured attention-head pruning for OLMo-1B-0724-hf.

Removes head 5 (0-indexed) from every layer's q_proj, k_proj, v_proj
(row blocks) and o_proj (column blocks), leaving 15 heads of 128 dims
each (1920 = 15 * 128) per layer. All other tensors are copied through
unchanged.
"""

import json
import os

import torch
from safetensors.torch import save_file

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
INPUT_DIR = os.path.join(REPO_ROOT, "inputs", "base")
OUTPUT_DIR = os.path.join(REPO_ROOT, "out", "T2")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "model.safetensors")

NUM_LAYERS = 16
NUM_HEADS = 16
HEAD_DIM = 128
HIDDEN_SIZE = NUM_HEADS * HEAD_DIM  # 2048
PRUNE_HEAD = 5
EXPECTED_TENSOR_COUNT = 114

# Head 5 occupies rows/cols [5*128, 6*128) = [640, 768).
PRUNE_START = PRUNE_HEAD * HEAD_DIM  # 640
PRUNE_END = PRUNE_START + HEAD_DIM  # 768


def load_state_dict(input_dir: str) -> dict[str, torch.Tensor]:
    index_path = os.path.join(input_dir, "model.safetensors.index.json")
    with open(index_path) as f:
        index = json.load(f)
    weight_map = index["weight_map"]

    shard_cache: dict[str, dict[str, torch.Tensor]] = {}
    state_dict = {}
    for name, shard_file in weight_map.items():
        if shard_file not in shard_cache:
            from safetensors import safe_open

            shard_path = os.path.join(input_dir, shard_file)
            tensors = {}
            with safe_open(shard_path, framework="pt") as f:
                for key in f.keys():
                    tensors[key] = f.get_tensor(key)
            shard_cache[shard_file] = tensors
        state_dict[name] = shard_cache[shard_file][name]
    return state_dict


def prune_rows(tensor: torch.Tensor) -> torch.Tensor:
    assert tensor.shape[0] == HIDDEN_SIZE, f"expected {HIDDEN_SIZE} rows, got {tensor.shape[0]}"
    kept = torch.cat([tensor[:PRUNE_START], tensor[PRUNE_END:]], dim=0)
    return kept.contiguous()


def prune_cols(tensor: torch.Tensor) -> torch.Tensor:
    assert tensor.shape[1] == HIDDEN_SIZE, f"expected {HIDDEN_SIZE} cols, got {tensor.shape[1]}"
    kept = torch.cat([tensor[:, :PRUNE_START], tensor[:, PRUNE_END:]], dim=1)
    return kept.contiguous()


def main() -> None:
    state_dict = load_state_dict(INPUT_DIR)
    assert len(state_dict) == EXPECTED_TENSOR_COUNT, (
        f"expected {EXPECTED_TENSOR_COUNT} input tensors, got {len(state_dict)}"
    )

    output: dict[str, torch.Tensor] = {}
    for name, tensor in state_dict.items():
        is_row_pruned = any(
            name == f"model.layers.{i}.self_attn.{proj}.weight"
            for i in range(NUM_LAYERS)
            for proj in ("q_proj", "k_proj", "v_proj")
        )
        is_col_pruned = any(
            name == f"model.layers.{i}.self_attn.o_proj.weight" for i in range(NUM_LAYERS)
        )

        if is_row_pruned:
            output[name] = prune_rows(tensor)
        elif is_col_pruned:
            output[name] = prune_cols(tensor)
        else:
            output[name] = tensor.contiguous()

    expected_reduced = NUM_HEADS * HEAD_DIM - HEAD_DIM  # 1920

    for i in range(NUM_LAYERS):
        for proj in ("q_proj", "k_proj", "v_proj"):
            key = f"model.layers.{i}.self_attn.{proj}.weight"
            shape = output[key].shape
            assert shape == (expected_reduced, HIDDEN_SIZE), (
                f"{key} has shape {tuple(shape)}, expected ({expected_reduced}, {HIDDEN_SIZE})"
            )
        key = f"model.layers.{i}.self_attn.o_proj.weight"
        shape = output[key].shape
        assert shape == (HIDDEN_SIZE, expected_reduced), (
            f"{key} has shape {tuple(shape)}, expected ({HIDDEN_SIZE}, {expected_reduced})"
        )

    assert output["model.layers.0.self_attn.q_proj.weight"].shape == (1920, 2048)
    assert output["model.layers.0.self_attn.k_proj.weight"].shape == (1920, 2048)
    assert output["model.layers.0.self_attn.v_proj.weight"].shape == (1920, 2048)
    assert output["model.layers.0.self_attn.o_proj.weight"].shape == (2048, 1920)
    assert len(output) == EXPECTED_TENSOR_COUNT, (
        f"expected {EXPECTED_TENSOR_COUNT} output tensors, got {len(output)}"
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_file(output, OUTPUT_PATH)
    print(f"Wrote {len(output)} tensors to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
