"""Prune attention head 5 from every layer of OLMo-1B-0724-hf.

Loads the sharded safetensors checkpoint, slices the head-5 row/column block
out of q_proj/k_proj/v_proj (row blocks) and o_proj (column blocks) for every
layer, leaves everything else untouched, and writes a single-file
safetensors checkpoint.
"""

import json
import os

import torch
from safetensors.torch import load_file, save_file

IN_DIR = "inputs/base"
OUT_DIR = "out/T2"
OUT_PATH = os.path.join(OUT_DIR, "model.safetensors")

NUM_LAYERS = 16
HEAD_DIM = 128
HEAD_TO_PRUNE = 5
HIDDEN = 2048

# Row/col range removed by dropping head 5: rows/cols 640..767 (0-indexed).
HEAD_START = HEAD_TO_PRUNE * HEAD_DIM
HEAD_END = HEAD_START + HEAD_DIM


def keep_indices(size: int) -> torch.Tensor:
    idx = torch.cat(
        [torch.arange(0, HEAD_START), torch.arange(HEAD_END, size)]
    )
    return idx


def load_all_tensors(in_dir: str) -> dict[str, torch.Tensor]:
    index_path = os.path.join(in_dir, "model.safetensors.index.json")
    with open(index_path) as f:
        index = json.load(f)
    shard_files = sorted(set(index["weight_map"].values()))
    tensors: dict[str, torch.Tensor] = {}
    for shard in shard_files:
        shard_tensors = load_file(os.path.join(in_dir, shard))
        tensors.update(shard_tensors)
    expected_keys = set(index["weight_map"].keys())
    assert set(tensors.keys()) == expected_keys, (
        f"Loaded tensor keys do not match index: "
        f"missing={expected_keys - set(tensors.keys())} "
        f"extra={set(tensors.keys()) - expected_keys}"
    )
    return tensors


def main() -> None:
    tensors = load_all_tensors(IN_DIR)
    assert len(tensors) == 114, f"expected 114 input tensors, got {len(tensors)}"

    row_keep = keep_indices(HIDDEN)  # for q_proj/k_proj/v_proj (row blocks)
    col_keep = keep_indices(HIDDEN)  # for o_proj (column blocks)

    out = dict(tensors)  # start as a copy; per-layer entries overwritten below

    for i in range(NUM_LAYERS):
        prefix = f"model.layers.{i}.self_attn"
        for proj in ("q_proj", "k_proj", "v_proj"):
            key = f"{prefix}.{proj}.weight"
            w = tensors[key]
            assert w.shape == (HIDDEN, HIDDEN), f"{key} has unexpected shape {tuple(w.shape)}"
            out[key] = w.index_select(0, row_keep).contiguous()

        key = f"{prefix}.o_proj.weight"
        w = tensors[key]
        assert w.shape == (HIDDEN, HIDDEN), f"{key} has unexpected shape {tuple(w.shape)}"
        out[key] = w.index_select(1, col_keep).contiguous()

    # Required checks before writing.
    assert out["model.layers.0.self_attn.q_proj.weight"].shape == (1920, 2048)
    assert out["model.layers.0.self_attn.k_proj.weight"].shape == (1920, 2048)
    assert out["model.layers.0.self_attn.v_proj.weight"].shape == (1920, 2048)
    assert out["model.layers.0.self_attn.o_proj.weight"].shape == (2048, 1920)
    assert len(out) == 114, f"expected 114 output tensors, got {len(out)}"

    os.makedirs(OUT_DIR, exist_ok=True)
    save_file(out, OUT_PATH)
    print(f"Wrote {OUT_PATH} with {len(out)} tensors.")


if __name__ == "__main__":
    main()
