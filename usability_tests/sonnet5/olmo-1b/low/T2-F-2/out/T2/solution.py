"""
T2: Structured attention-head pruning (OLMo-1B-0724-hf), remove head 5 of 16
from every layer.

Approach: plain script on top of `safetensors` + `torch`. Load the sharded
checkpoint, and for every layer's q/k/v/o projection weights, slice out the
128-row (or 128-column, for o_proj) block belonging to head 5, keeping the
surrounding blocks in original order. All other tensors are copied through
unchanged. Shapes are asserted before writing.
"""

import json
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

IN_DIR = Path("inputs/base")
OUT_DIR = Path("out/T2")
OUT_FILE = OUT_DIR / "model.safetensors"

NUM_LAYERS = 16
HEAD_DIM = 128
HIDDEN = 2048
PRUNE_HEAD = 5

ROW_START = PRUNE_HEAD * HEAD_DIM  # 640
ROW_END = ROW_START + HEAD_DIM  # 768


def load_all_tensors(in_dir: Path) -> dict[str, torch.Tensor]:
    index_path = in_dir / "model.safetensors.index.json"
    with open(index_path) as f:
        index = json.load(f)
    shard_names = sorted(set(index["weight_map"].values()))
    tensors: dict[str, torch.Tensor] = {}
    for shard_name in shard_names:
        shard = load_file(in_dir / shard_name)
        tensors.update(shard)
    # Sanity: every key in the index must have been loaded, and no extras.
    assert set(tensors.keys()) == set(index["weight_map"].keys()), (
        "loaded tensor keys do not match index weight_map keys"
    )
    return tensors


def prune_row_blocks(t: torch.Tensor) -> torch.Tensor:
    # heads are row blocks: keep rows 0..639, 768..2047
    assert t.shape == (HIDDEN, HIDDEN), f"unexpected shape {tuple(t.shape)}"
    return torch.cat([t[:ROW_START, :], t[ROW_END:, :]], dim=0)


def prune_col_blocks(t: torch.Tensor) -> torch.Tensor:
    # heads are column blocks (o_proj): keep columns 0..639, 768..2047
    assert t.shape == (HIDDEN, HIDDEN), f"unexpected shape {tuple(t.shape)}"
    return torch.cat([t[:, :ROW_START], t[:, ROW_END:]], dim=1)


def main() -> None:
    tensors = load_all_tensors(IN_DIR)
    n_input = len(tensors)

    out_tensors: dict[str, torch.Tensor] = {}
    for name, tensor in tensors.items():
        if name.startswith("model.layers.") and ".self_attn." in name:
            suffix = name.rsplit(".self_attn.", 1)[1]
            if suffix == "q_proj.weight":
                out_tensors[name] = prune_row_blocks(tensor)
                continue
            if suffix == "k_proj.weight":
                out_tensors[name] = prune_row_blocks(tensor)
                continue
            if suffix == "v_proj.weight":
                out_tensors[name] = prune_row_blocks(tensor)
                continue
            if suffix == "o_proj.weight":
                out_tensors[name] = prune_col_blocks(tensor)
                continue
        out_tensors[name] = tensor

    # --- Required checks: fail loudly before writing ---
    assert len(out_tensors) == n_input == 114, (
        f"expected 114 tensors, got in={n_input} out={len(out_tensors)}"
    )

    def check_shape(name: str, expected: tuple[int, ...]) -> None:
        got = tuple(out_tensors[name].shape)
        assert got == expected, f"{name}: expected {expected}, got {got}"

    check_shape("model.layers.0.self_attn.q_proj.weight", (1920, 2048))
    check_shape("model.layers.0.self_attn.k_proj.weight", (1920, 2048))
    check_shape("model.layers.0.self_attn.v_proj.weight", (1920, 2048))
    check_shape("model.layers.0.self_attn.o_proj.weight", (2048, 1920))

    for i in range(NUM_LAYERS):
        check_shape(f"model.layers.{i}.self_attn.q_proj.weight", (1920, 2048))
        check_shape(f"model.layers.{i}.self_attn.k_proj.weight", (1920, 2048))
        check_shape(f"model.layers.{i}.self_attn.v_proj.weight", (1920, 2048))
        check_shape(f"model.layers.{i}.self_attn.o_proj.weight", (2048, 1920))

    for name, tensor in out_tensors.items():
        assert tensor.is_contiguous(), f"{name} is not contiguous"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    save_file(out_tensors, str(OUT_FILE))
    print(f"Wrote {len(out_tensors)} tensors to {OUT_FILE}")


if __name__ == "__main__":
    main()
    sys.exit(0)
