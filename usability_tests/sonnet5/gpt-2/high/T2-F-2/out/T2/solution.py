"""
T2: structured attention-head pruning for GPT-2 (124M).

Removes head 5 (0-indexed) from every layer's attention block, at the
checkpoint level, by slicing the fused c_attn (qkv) weight/bias and the
c_proj weight. Implemented as a plain script over `safetensors` + `torch`
(condition F allowlist) because the transform is a fixed set of column/row
slices given directly by the task spec; no model-loading round trip through
`transformers.prune_heads` is needed and would risk not matching the exact
per-block ordering required here.

GPT-2 uses Conv1D layout ([in, out]), the transpose of nn.Linear:
  - c_attn.weight: [768, 2304]  -> heads are column blocks (dim=1)
  - c_attn.bias:   [2304]       -> heads are blocks along the only dim
  - c_proj.weight: [768, 768]   -> heads are row blocks (dim=0)
"""

import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

HIDDEN = 768
NUM_HEADS = 12
HEAD_DIM = HIDDEN // NUM_HEADS  # 64
NUM_LAYERS = 12
PRUNE_HEAD = 5

SRC = Path("inputs/base/model.safetensors")
DST = Path("out/T2/model.safetensors")


def keep_indices(n_segments: int, segment_width: int, prune_head: int) -> list[int]:
    """Indices to keep along a dimension made of `n_segments` concatenated
    blocks of width `segment_width`, each itself split into NUM_HEADS heads
    of HEAD_DIM, dropping `prune_head` from every segment."""
    idx: list[int] = []
    for seg in range(n_segments):
        base = seg * segment_width
        for h in range(NUM_HEADS):
            if h == prune_head:
                continue
            start = base + h * HEAD_DIM
            idx.extend(range(start, start + HEAD_DIM))
    return idx


def main() -> None:
    if not SRC.exists():
        sys.exit(f"missing input: {SRC}")

    qkv_keep = keep_indices(n_segments=3, segment_width=HIDDEN, prune_head=PRUNE_HEAD)
    proj_keep = keep_indices(n_segments=1, segment_width=HIDDEN, prune_head=PRUNE_HEAD)

    assert qkv_keep[:6] == [0, 1, 2, 3, 4, 5]  # sanity: block ordering starts at 0
    assert len(qkv_keep) == 2112
    assert len(proj_keep) == 704

    qkv_idx = torch.tensor(qkv_keep, dtype=torch.long)
    proj_idx = torch.tensor(proj_keep, dtype=torch.long)

    out: dict[str, torch.Tensor] = {}
    with safe_open(str(SRC), framework="pt") as f:
        keys = list(f.keys())
        assert len(keys) == 160, f"expected 160 input tensors, found {len(keys)}"
        for key in keys:
            t = f.get_tensor(key)
            layer = None
            if key.startswith("h."):
                layer = int(key.split(".", 2)[1])
            if layer is not None and key == f"h.{layer}.attn.c_attn.weight":
                t = t.index_select(1, qkv_idx).contiguous()
            elif layer is not None and key == f"h.{layer}.attn.c_attn.bias":
                t = t.index_select(0, qkv_idx).contiguous()
            elif layer is not None and key == f"h.{layer}.attn.c_proj.weight":
                t = t.index_select(0, proj_idx).contiguous()
            out[key] = t

    # Required checks: fail loudly before writing anything.
    assert out["h.0.attn.c_attn.weight"].shape == (768, 2112), out["h.0.attn.c_attn.weight"].shape
    assert out["h.0.attn.c_attn.bias"].shape == (2112,), out["h.0.attn.c_attn.bias"].shape
    assert out["h.0.attn.c_proj.weight"].shape == (704, 768), out["h.0.attn.c_proj.weight"].shape
    assert len(out) == 160, f"expected 160 output tensors, got {len(out)}"

    for i in range(NUM_LAYERS):
        assert out[f"h.{i}.attn.c_attn.weight"].shape == (768, 2112)
        assert out[f"h.{i}.attn.c_attn.bias"].shape == (2112,)
        assert out[f"h.{i}.attn.c_proj.weight"].shape == (704, 768)
        assert out[f"h.{i}.attn.c_proj.bias"].shape == (768,)
        assert out[f"h.{i}.attn.bias"].shape == (1, 1, 1024, 1024)

    DST.parent.mkdir(parents=True, exist_ok=True)
    save_file(out, str(DST))
    print(f"wrote {DST} with {len(out)} tensors")


if __name__ == "__main__":
    main()
