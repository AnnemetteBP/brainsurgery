"""T2: structured attention-head pruning for OLMo-1B-0724-hf.

Removes head 5 (0-indexed) from every layer's q/k/v/o attention projections
by slicing the checkpoint directly with `safetensors`. Heads are contiguous
128-wide blocks; for q/k/v the head axis is rows (dim 0, nn.Linear [out, in]
layout), for o_proj the head axis is columns (dim 1). Pruned head 5 occupies
rows/cols 640..767, so the kept ranges are 0..639 and 768..2047, concatenated
in that order. All other tensors are copied through unchanged, bit for bit.

Plain safetensors + torch slicing was chosen over transformers'
`PreTrainedModel.prune_heads` because that API prunes heads via an internal
mask/index_select path intended for live models, not checkpoints, and does
not guarantee the exact row/column ordering the spec requires or a bit-exact
copy of the untouched tensors. Direct slicing on the raw tensors is simpler,
has no floating-point risk (a slice is a memory view, not a computation), and
maps 1:1 onto the spec's row/column ranges.
"""

import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

HERE = Path(__file__).resolve().parent
INPUT_DIR = HERE.parent.parent / "inputs" / "base"
OUTPUT_PATH = HERE / "model.safetensors"

HIDDEN_SIZE = 2048
NUM_HEADS = 16
HEAD_DIM = 128
PRUNE_HEAD = 5
NUM_LAYERS = 16
EXPECTED_TOTAL_TENSORS = 114

# Row/column ranges to keep on the head axis, in order.
_prune_start = PRUNE_HEAD * HEAD_DIM
_prune_end = _prune_start + HEAD_DIM
KEEP_RANGES = [(0, _prune_start), (_prune_end, HIDDEN_SIZE)]


def load_index(input_dir: Path) -> dict[str, str]:
    with open(input_dir / "model.safetensors.index.json") as f:
        index = json.load(f)
    return index["weight_map"]


def slice_rows(tensor: torch.Tensor) -> torch.Tensor:
    """Keep row blocks in KEEP_RANGES, concatenated in order (dim 0)."""
    pieces = [tensor[start:end, :] for start, end in KEEP_RANGES]
    return torch.cat(pieces, dim=0).contiguous()


def slice_cols(tensor: torch.Tensor) -> torch.Tensor:
    """Keep column blocks in KEEP_RANGES, concatenated in order (dim 1)."""
    pieces = [tensor[:, start:end] for start, end in KEEP_RANGES]
    return torch.cat(pieces, dim=1).contiguous()


def main() -> None:
    weight_map = load_index(INPUT_DIR)
    shard_names = sorted(set(weight_map.values()))
    open_files = {name: safe_open(INPUT_DIR / name, framework="pt") for name in shard_names}

    row_pruned = {f"model.layers.{i}.self_attn.{proj}_proj.weight" for i in range(NUM_LAYERS) for proj in ("q", "k", "v")}
    col_pruned = {f"model.layers.{i}.self_attn.o_proj.weight" for i in range(NUM_LAYERS)}

    output: dict[str, torch.Tensor] = {}
    for name, shard in weight_map.items():
        tensor = open_files[shard].get_tensor(name)
        if name in row_pruned:
            assert tensor.shape == (HIDDEN_SIZE, HIDDEN_SIZE), f"{name}: unexpected input shape {tuple(tensor.shape)}"
            tensor = slice_rows(tensor)
        elif name in col_pruned:
            assert tensor.shape == (HIDDEN_SIZE, HIDDEN_SIZE), f"{name}: unexpected input shape {tuple(tensor.shape)}"
            tensor = slice_cols(tensor)
        output[name] = tensor

    # Required checks: fail loudly before writing anything.
    expected_qkv_shape = (HIDDEN_SIZE - HEAD_DIM, HIDDEN_SIZE)
    expected_o_shape = (HIDDEN_SIZE, HIDDEN_SIZE - HEAD_DIM)
    for proj, expected in (("q", expected_qkv_shape), ("k", expected_qkv_shape), ("v", expected_qkv_shape)):
        name = f"model.layers.0.self_attn.{proj}_proj.weight"
        actual = tuple(output[name].shape)
        assert actual == expected, f"{name}: expected {expected}, got {actual}"
    name = "model.layers.0.self_attn.o_proj.weight"
    actual = tuple(output[name].shape)
    assert actual == expected_o_shape, f"{name}: expected {expected_o_shape}, got {actual}"

    assert len(output) == EXPECTED_TOTAL_TENSORS, f"expected {EXPECTED_TOTAL_TENSORS} tensors, got {len(output)}"
    assert set(output.keys()) == set(weight_map.keys()), "tensor name set changed"

    # Also verify every layer, not just layer 0, and that non-head tensors
    # are byte-identical to the input.
    for i in range(NUM_LAYERS):
        for proj in ("q", "k", "v"):
            name = f"model.layers.{i}.self_attn.{proj}_proj.weight"
            assert tuple(output[name].shape) == expected_qkv_shape, name
        name = f"model.layers.{i}.self_attn.o_proj.weight"
        assert tuple(output[name].shape) == expected_o_shape, name

    for name, shard in weight_map.items():
        if name in row_pruned or name in col_pruned:
            continue
        original = open_files[shard].get_tensor(name)
        assert torch.equal(output[name], original), f"{name}: unexpected modification to untouched tensor"

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_file(output, str(OUTPUT_PATH))
    print(f"Wrote {len(output)} tensors to {OUTPUT_PATH}")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"CHECK FAILED: {e}", file=sys.stderr)
        sys.exit(1)
