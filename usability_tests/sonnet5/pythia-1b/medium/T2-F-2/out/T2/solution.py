"""
T2: Structured attention-head pruning (Pythia-1B), condition F.

Removes head 5 (0-indexed) from every layer of the fused GPT-NeoX attention
projections. Plain script on top of `safetensors` + `torch` (both allowed
under F-allowed.md) rather than `transformers.prune_heads`: GPT-NeoX's fused,
per-head-interleaved `query_key_value` layout means the pruning has to
operate on the raw row/column blocks described in TASK.md, and a direct
slice-and-checkpoint script makes that block arithmetic explicit and
independently checkable, rather than relying on a generic pruning helper to
have the right GPT-NeoX-specific slicing built in.

Layout recap (see TASK.md):
  - query_key_value.weight: [6144, 2048], 8 heads x 768 rows, each 768-row
    block interleaved as [q(256) | k(256) | v(256)] for that head.
  - query_key_value.bias:   [6144], same per-head row layout.
  - dense.weight:           [2048, 2048], 8 heads x 256 output columns... no,
    heads are 256-wide *input* column blocks (dense consumes concatenated
    head outputs on its `in` axis, matching nn.Linear [out, in]).

Removing head H (0-indexed) means dropping:
  - rows [768*H : 768*H+768) from query_key_value.weight and .bias
  - columns [256*H : 256*H+256) from dense.weight
"""

import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

HEAD_TO_PRUNE = 5
NUM_LAYERS = 16
HEAD_DIM = 256
QKV_BLOCK = 3 * HEAD_DIM  # 768: one head's interleaved q,k,v rows

IN_PATH = Path("inputs/base/model.safetensors")
OUT_DIR = Path("out/T2")
OUT_PATH = OUT_DIR / "model.safetensors"


def drop_block(tensor: torch.Tensor, dim: int, start: int, block: int) -> torch.Tensor:
    """Remove `block` entries along `dim` starting at `start`, keep the rest in order."""
    idx = list(range(tensor.shape[dim]))
    del idx[start : start + block]
    return tensor.index_select(dim, torch.tensor(idx, dtype=torch.long))


def main() -> None:
    state = load_file(str(IN_PATH))
    orig_keys = set(state.keys())
    if len(orig_keys) != 244:
        sys.exit(f"expected 244 input tensors, got {len(orig_keys)}")

    out = {}
    for key, tensor in state.items():
        if key.endswith("attention.query_key_value.weight"):
            i = int(key.split(".")[2])
            assert tensor.shape == (6144, 2048), f"{key}: unexpected shape {tuple(tensor.shape)}"
            new = drop_block(tensor, dim=0, start=QKV_BLOCK * HEAD_TO_PRUNE, block=QKV_BLOCK)
            assert new.shape == (5376, 2048), f"{key}: got {tuple(new.shape)}"
            out[key] = new.contiguous()
        elif key.endswith("attention.query_key_value.bias"):
            i = int(key.split(".")[2])
            assert tensor.shape == (6144,), f"{key}: unexpected shape {tuple(tensor.shape)}"
            new = drop_block(tensor, dim=0, start=QKV_BLOCK * HEAD_TO_PRUNE, block=QKV_BLOCK)
            assert new.shape == (5376,), f"{key}: got {tuple(new.shape)}"
            out[key] = new.contiguous()
        elif key.endswith("attention.dense.weight"):
            i = int(key.split(".")[2])
            assert tensor.shape == (2048, 2048), f"{key}: unexpected shape {tuple(tensor.shape)}"
            new = drop_block(tensor, dim=1, start=HEAD_DIM * HEAD_TO_PRUNE, block=HEAD_DIM)
            assert new.shape == (2048, 1792), f"{key}: got {tuple(new.shape)}"
            out[key] = new.contiguous()
        else:
            out[key] = tensor

    # Required checks (fail loudly before writing).
    w0 = out["gpt_neox.layers.0.attention.query_key_value.weight"]
    b0 = out["gpt_neox.layers.0.attention.query_key_value.bias"]
    d0 = out["gpt_neox.layers.0.attention.dense.weight"]
    assert tuple(w0.shape) == (5376, 2048), w0.shape
    assert tuple(b0.shape) == (5376,), b0.shape
    assert tuple(d0.shape) == (2048, 1792), d0.shape
    assert len(out) == 244, len(out)
    assert set(out.keys()) == orig_keys, "tensor names changed"

    # Sanity: every layer touched, all dtypes preserved.
    touched = set()
    for i in range(NUM_LAYERS):
        for suffix in ("query_key_value.weight", "query_key_value.bias", "dense.weight"):
            k = f"gpt_neox.layers.{i}.attention.{suffix}"
            assert k in out, f"missing {k}"
            touched.add(k)
            assert out[k].dtype == state[k].dtype, f"{k}: dtype changed"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    save_file(out, str(OUT_PATH))
    print(f"wrote {OUT_PATH} with {len(out)} tensors; pruned head {HEAD_TO_PRUNE} "
          f"from {len(touched)} tensors across {NUM_LAYERS} layers")


if __name__ == "__main__":
    main()
