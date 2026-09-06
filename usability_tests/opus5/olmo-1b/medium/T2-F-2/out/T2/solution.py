"""T2: structured attention-head pruning for OLMo-1B-0724-hf.

Removes head 5 from every layer at the checkpoint level, by slicing the
head-bearing projection tensors directly:

  q/k/v_proj.weight  [out=n_heads*head_dim, in]  -> heads are row blocks
  o_proj.weight      [out, in=n_heads*head_dim]  -> heads are column blocks

Everything else (embeddings, lm_head, MLP matrices) is copied unchanged and
tensor names are preserved.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

IN_DIR = Path("inputs/base")
OUT_DIR = Path("out/T2")
OUT_FILE = OUT_DIR / "model.safetensors"

NUM_LAYERS = 16
NUM_HEADS = 16
HEAD_DIM = 128
HIDDEN = NUM_HEADS * HEAD_DIM  # 2048
PRUNE_HEAD = 5

ROW_TENSORS = ("q_proj", "k_proj", "v_proj")  # heads on dim 0
COL_TENSORS = ("o_proj",)  # heads on dim 1

EXPECTED_TENSORS = 114
KEPT = HIDDEN - HEAD_DIM  # 1920


def keep_index() -> torch.Tensor:
    """Indices of the surviving head slots, in original order."""
    lo = PRUNE_HEAD * HEAD_DIM
    hi = lo + HEAD_DIM
    idx = torch.cat([torch.arange(0, lo), torch.arange(hi, HIDDEN)])
    assert idx.numel() == KEPT, f"keep index has {idx.numel()} entries, expected {KEPT}"
    return idx


def load_state_dict() -> dict[str, torch.Tensor]:
    """Load every shard listed in the index into one flat state dict."""
    index = json.loads((IN_DIR / "model.safetensors.index.json").read_text())
    state: dict[str, torch.Tensor] = {}
    for shard in sorted(set(index["weight_map"].values())):
        for name, tensor in load_file(IN_DIR / shard).items():
            assert name not in state, f"duplicate tensor across shards: {name}"
            state[name] = tensor
    missing = set(index["weight_map"]) - set(state)
    assert not missing, f"tensors listed in the index but not loaded: {sorted(missing)}"
    return state


def prune(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    idx = keep_index()
    out: dict[str, torch.Tensor] = {}
    touched = 0
    for name, tensor in state.items():
        parts = name.split(".")
        is_attn = len(parts) == 6 and parts[0:2] == ["model", "layers"] and parts[3] == "self_attn"
        proj = parts[4] if is_attn else None
        if proj in ROW_TENSORS:
            assert tensor.shape == (HIDDEN, HIDDEN), f"{name}: unexpected shape {tuple(tensor.shape)}"
            new = tensor.index_select(0, idx).contiguous()
            touched += 1
        elif proj in COL_TENSORS:
            assert tensor.shape == (HIDDEN, HIDDEN), f"{name}: unexpected shape {tuple(tensor.shape)}"
            new = tensor.index_select(1, idx).contiguous()
            touched += 1
        else:
            new = tensor
        assert new.dtype == tensor.dtype, f"{name}: dtype changed {tensor.dtype} -> {new.dtype}"
        out[name] = new
    expected_touched = NUM_LAYERS * (len(ROW_TENSORS) + len(COL_TENSORS))
    assert touched == expected_touched, f"sliced {touched} tensors, expected {expected_touched}"
    return out


def check(original: dict[str, torch.Tensor], pruned: dict[str, torch.Tensor]) -> None:
    """Required checks. Anything failing here aborts before we write."""
    required = {
        "model.layers.0.self_attn.q_proj.weight": (KEPT, HIDDEN),
        "model.layers.0.self_attn.k_proj.weight": (KEPT, HIDDEN),
        "model.layers.0.self_attn.v_proj.weight": (KEPT, HIDDEN),
        "model.layers.0.self_attn.o_proj.weight": (HIDDEN, KEPT),
    }
    for name, shape in required.items():
        assert name in pruned, f"missing required tensor {name}"
        got = tuple(pruned[name].shape)
        assert got == shape, f"{name}: shape {got}, expected {shape}"

    assert len(pruned) == EXPECTED_TENSORS, f"output has {len(pruned)} tensors, expected {EXPECTED_TENSORS}"
    assert set(pruned) == set(original), "tensor names changed"

    # Same checks for every remaining layer, plus untouched tensors verified
    # element-wise against the input.
    for i in range(NUM_LAYERS):
        for proj in ROW_TENSORS:
            t = pruned[f"model.layers.{i}.self_attn.{proj}.weight"]
            assert tuple(t.shape) == (KEPT, HIDDEN), f"layer {i} {proj}: shape {tuple(t.shape)}"
        t = pruned[f"model.layers.{i}.self_attn.o_proj.weight"]
        assert tuple(t.shape) == (HIDDEN, KEPT), f"layer {i} o_proj: shape {tuple(t.shape)}"

    lo, hi = PRUNE_HEAD * HEAD_DIM, (PRUNE_HEAD + 1) * HEAD_DIM
    for name, before in original.items():
        after = pruned[name]
        if tuple(after.shape) == tuple(before.shape):
            assert torch.equal(after, before), f"{name}: changed but should be untouched"
        elif after.shape[0] != before.shape[0]:
            assert torch.equal(after[:lo], before[:lo]), f"{name}: rows before the cut moved"
            assert torch.equal(after[lo:], before[hi:]), f"{name}: rows after the cut are misaligned"
        else:
            assert torch.equal(after[:, :lo], before[:, :lo]), f"{name}: cols before the cut moved"
            assert torch.equal(after[:, lo:], before[:, hi:]), f"{name}: cols after the cut are misaligned"


def main() -> None:
    state = load_state_dict()
    assert len(state) == EXPECTED_TENSORS, f"input has {len(state)} tensors, expected {EXPECTED_TENSORS}"
    pruned = prune(state)
    check(state, pruned)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    save_file(pruned, str(OUT_FILE))

    written = load_file(OUT_FILE)
    assert len(written) == EXPECTED_TENSORS, f"wrote {len(written)} tensors, expected {EXPECTED_TENSORS}"
    for name, tensor in pruned.items():
        assert torch.equal(written[name], tensor), f"{name}: readback mismatch"
    print(f"wrote {OUT_FILE} with {len(written)} tensors")


if __name__ == "__main__":
    main()
