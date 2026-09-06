"""
T2: Structured attention-head pruning (GPT-2 124M).

Remove head 5 (0-indexed) from every layer's attention. GPT-2 stores
projection weights in Conv1D layout ([in, out]), the transpose of nn.Linear.

Per layer:
  - attn.c_attn.weight [768, 2304]: fused [q|k|v], heads are 64-wide column
    blocks inside each 768-wide segment. Drop columns 320:384 (head 5) from
    the q segment (cols 0:768), the k segment (cols 768:1536), and the v
    segment (cols 1536:2304).
  - attn.c_attn.bias [2304]: same layout along its single axis.
  - attn.c_proj.weight [768, 768]: heads are 64-wide row blocks (input side
    of the output projection). Drop rows 320:384.
  - attn.c_proj.bias, attn.bias: untouched (not per-head).
  - everything else: untouched, copied through unmodified.
"""

from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

IN_PATH = Path("inputs/base/model.safetensors")
OUT_PATH = Path("out/T2/model.safetensors")

N_LAYERS = 12
N_HEADS = 12
HEAD_DIM = 64
HIDDEN = N_HEADS * HEAD_DIM  # 768
HEAD_TO_DROP = 5


def keep_indices_excluding_head(n_heads, head_dim, head_to_drop):
    """Indices along a HIDDEN-wide axis that keep all heads except one."""
    idx = []
    for h in range(n_heads):
        if h == head_to_drop:
            continue
        start = h * head_dim
        idx.extend(range(start, start + head_dim))
    return torch.tensor(idx, dtype=torch.long)


def main():
    state_dict = load_file(str(IN_PATH))
    assert len(state_dict) == 160, f"expected 160 input tensors, got {len(state_dict)}"

    keep = keep_indices_excluding_head(N_HEADS, HEAD_DIM, HEAD_TO_DROP)
    assert keep.numel() == (N_HEADS - 1) * HEAD_DIM  # 704

    out = {}
    for name, tensor in state_dict.items():
        is_per_layer_attn = False
        for i in range(N_LAYERS):
            prefix = f"h.{i}.attn."
            if not name.startswith(prefix):
                continue
            suffix = name[len(prefix):]

            if suffix == "c_attn.weight":
                # [768, 2304] = [in, q|k|v out]; keep-per-segment on axis 1.
                assert tensor.shape == (HIDDEN, 3 * HIDDEN), (name, tensor.shape)
                segments = []
                for s in range(3):
                    seg = tensor[:, s * HIDDEN:(s + 1) * HIDDEN]
                    segments.append(seg[:, keep])
                out[name] = torch.cat(segments, dim=1).contiguous()
                assert out[name].shape == (HIDDEN, 3 * (HIDDEN - HEAD_DIM))
                is_per_layer_attn = True

            elif suffix == "c_attn.bias":
                # [2304] = [q|k|v]; keep-per-segment on the single axis.
                assert tensor.shape == (3 * HIDDEN,), (name, tensor.shape)
                segments = []
                for s in range(3):
                    seg = tensor[s * HIDDEN:(s + 1) * HIDDEN]
                    segments.append(seg[keep])
                out[name] = torch.cat(segments, dim=0).contiguous()
                assert out[name].shape == (3 * (HIDDEN - HEAD_DIM),)
                is_per_layer_attn = True

            elif suffix == "c_proj.weight":
                # [768, 768] = [in, out]; heads are row blocks (axis 0).
                assert tensor.shape == (HIDDEN, HIDDEN), (name, tensor.shape)
                out[name] = tensor[keep, :].contiguous()
                assert out[name].shape == (HIDDEN - HEAD_DIM, HIDDEN)
                is_per_layer_attn = True

            # c_proj.bias and attn.bias fall through untouched below.
            break

        if not is_per_layer_attn and name not in out:
            out[name] = tensor.clone().contiguous()

    # Required checks.
    assert out["h.0.attn.c_attn.weight"].shape == (768, 2112), out["h.0.attn.c_attn.weight"].shape
    assert out["h.0.attn.c_attn.bias"].shape == (2112,), out["h.0.attn.c_attn.bias"].shape
    assert out["h.0.attn.c_proj.weight"].shape == (704, 768), out["h.0.attn.c_proj.weight"].shape
    assert len(out) == 160, f"expected 160 output tensors, got {len(out)}"

    for i in range(N_LAYERS):
        assert out[f"h.{i}.attn.c_attn.weight"].shape == (768, 2112)
        assert out[f"h.{i}.attn.c_attn.bias"].shape == (2112,)
        assert out[f"h.{i}.attn.c_proj.weight"].shape == (704, 768)
        # Untouched tensors keep their original shape.
        assert out[f"h.{i}.attn.c_proj.bias"].shape == state_dict[f"h.{i}.attn.c_proj.bias"].shape
        assert torch.equal(
            out[f"h.{i}.attn.c_proj.bias"], state_dict[f"h.{i}.attn.c_proj.bias"]
        )
        assert torch.equal(
            out[f"h.{i}.attn.bias"], state_dict[f"h.{i}.attn.bias"]
        )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_file(out, str(OUT_PATH))
    print(f"Wrote {len(out)} tensors to {OUT_PATH}")


if __name__ == "__main__":
    main()
