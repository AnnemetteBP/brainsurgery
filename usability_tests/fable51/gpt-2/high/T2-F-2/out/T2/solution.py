"""T2: remove attention head 5 from every layer of GPT-2 (124M) at the checkpoint level.

Plain safetensors + torch script. Per layer i:
  h.<i>.attn.c_attn.weight [768, 2304] -> [768, 2112]  (drop head-5 column block in q, k and v)
  h.<i>.attn.c_attn.bias   [2304]      -> [2112]
  h.<i>.attn.c_proj.weight [768, 768]  -> [704, 768]   (drop head-5 row block)
Every other tensor is copied unchanged. Output: out/T2/model.safetensors, 160 tensors.
"""

import os
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

SRC = "inputs/base/model.safetensors"
DST_DIR = "out/T2"
DST = os.path.join(DST_DIR, "model.safetensors")

N_LAYERS = 12
N_HEADS = 12
HEAD_DIM = 64
HIDDEN = N_HEADS * HEAD_DIM  # 768
PRUNE_HEAD = 5
EXPECTED_TENSORS = 160


def keep_index(n_segments: int) -> torch.LongTensor:
    """Indices to keep along a head-bearing axis made of `n_segments` 768-wide segments."""
    idx = []
    for seg in range(n_segments):
        base = seg * HIDDEN
        for head in range(N_HEADS):
            if head == PRUNE_HEAD:
                continue
            start = base + head * HEAD_DIM
            idx.extend(range(start, start + HEAD_DIM))
    return torch.tensor(idx, dtype=torch.long)


QKV_KEEP = keep_index(3)  # 2112 entries for the fused [q | k | v] axis
OUT_KEEP = keep_index(1)  # 704 entries for the c_proj input axis

# Sanity: the kept ranges must match the task statement exactly.
assert QKV_KEEP.tolist() == (
    list(range(0, 320)) + list(range(384, 768)) + list(range(768, 1088))
    + list(range(1152, 1536)) + list(range(1536, 1856)) + list(range(1920, 2304))
)
assert OUT_KEEP.tolist() == list(range(0, 320)) + list(range(384, 768))


def check(cond: bool, msg: str) -> None:
    if not cond:
        print(f"CHECK FAILED: {msg}", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    out: dict[str, torch.Tensor] = {}
    with safe_open(SRC, framework="pt") as f:
        keys = list(f.keys())
        check(len(keys) == EXPECTED_TENSORS, f"input has {len(keys)} tensors, expected {EXPECTED_TENSORS}")
        for k in keys:
            t = f.get_tensor(k)
            parts = k.split(".")
            is_layer = len(parts) >= 4 and parts[0] == "h" and parts[2] == "attn"
            if is_layer and parts[3] == "c_attn" and parts[4] == "weight":
                check(tuple(t.shape) == (HIDDEN, 3 * HIDDEN), f"{k} has shape {tuple(t.shape)}")
                t = t.index_select(1, QKV_KEEP)
            elif is_layer and parts[3] == "c_attn" and parts[4] == "bias":
                check(tuple(t.shape) == (3 * HIDDEN,), f"{k} has shape {tuple(t.shape)}")
                t = t.index_select(0, QKV_KEEP)
            elif is_layer and parts[3] == "c_proj" and parts[4] == "weight":
                check(tuple(t.shape) == (HIDDEN, HIDDEN), f"{k} has shape {tuple(t.shape)}")
                t = t.index_select(0, OUT_KEEP)
            out[k] = t.contiguous()

    # Required checks (fail loudly before writing).
    check(tuple(out["h.0.attn.c_attn.weight"].shape) == (768, 2112),
          f"h.0.attn.c_attn.weight shape {tuple(out['h.0.attn.c_attn.weight'].shape)}")
    check(tuple(out["h.0.attn.c_attn.bias"].shape) == (2112,),
          f"h.0.attn.c_attn.bias shape {tuple(out['h.0.attn.c_attn.bias'].shape)}")
    check(tuple(out["h.0.attn.c_proj.weight"].shape) == (704, 768),
          f"h.0.attn.c_proj.weight shape {tuple(out['h.0.attn.c_proj.weight'].shape)}")
    check(len(out) == EXPECTED_TENSORS, f"output has {len(out)} tensors, expected {EXPECTED_TENSORS}")

    # Extra: every layer was touched, and every layer has the pruned shapes.
    for i in range(N_LAYERS):
        check(tuple(out[f"h.{i}.attn.c_attn.weight"].shape) == (768, 2112), f"layer {i} c_attn.weight")
        check(tuple(out[f"h.{i}.attn.c_attn.bias"].shape) == (2112,), f"layer {i} c_attn.bias")
        check(tuple(out[f"h.{i}.attn.c_proj.weight"].shape) == (704, 768), f"layer {i} c_proj.weight")
        check(tuple(out[f"h.{i}.attn.c_proj.bias"].shape) == (768,), f"layer {i} c_proj.bias untouched")
        check(tuple(out[f"h.{i}.attn.bias"].shape) == (1, 1, 1024, 1024), f"layer {i} attn.bias untouched")

    os.makedirs(DST_DIR, exist_ok=True)
    save_file(out, DST, metadata={"format": "pt"})
    print(f"wrote {DST} with {len(out)} tensors")


if __name__ == "__main__":
    main()
