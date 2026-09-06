"""
T2: Structured attention-head pruning (GPT-2 124M), condition F.

Note on tooling: `transformers` 5.12.1 (the pinned version for this
condition) no longer exposes `PreTrainedModel.prune_heads` /
`find_pruneable_heads_and_indices` (removed upstream), so the route the task
brief suggests isn't actually available. `mergekit` and `torch-state-bridge`
operate on whole tensors / renamed keys, not sub-tensor column/row slices, so
they don't fit this either. This is a plain slicing script on top of
`safetensors` + `torch`, driven directly off the block boundaries given in
TASK.md, with the required shape/count checks enforced before writing.
"""

import torch
from safetensors.torch import load_file, save_file

INPUT_FILE = "inputs/base/model.safetensors"
OUTPUT_FILE = "out/T2/model.safetensors"

N_LAYERS = 12
HEAD_DIM = 64
N_HEADS = 12
HIDDEN = 768
HEAD_TO_PRUNE = 5

# Column/row blocks to KEEP, in order, for a 768-wide segment with head 5 removed.
KEEP_HEADS = [h for h in range(N_HEADS) if h != HEAD_TO_PRUNE]


def keep_slices(width_per_head: int, offset: int = 0) -> list[slice]:
    return [
        slice(offset + h * width_per_head, offset + (h + 1) * width_per_head)
        for h in KEEP_HEADS
    ]


def prune_c_attn_weight(w: torch.Tensor) -> torch.Tensor:
    # [768, 2304] = [in, q(768) | k(768) | v(768)]; drop head 5's 64 columns
    # from each of the three 768-wide segments.
    parts = []
    for seg in range(3):  # q, k, v
        offset = seg * HIDDEN
        for sl in keep_slices(HEAD_DIM, offset):
            parts.append(w[:, sl])
    return torch.cat(parts, dim=1)


def prune_c_attn_bias(b: torch.Tensor) -> torch.Tensor:
    parts = []
    for seg in range(3):
        offset = seg * HIDDEN
        for sl in keep_slices(HEAD_DIM, offset):
            parts.append(b[sl])
    return torch.cat(parts, dim=0)


def prune_c_proj_weight(w: torch.Tensor) -> torch.Tensor:
    # [768, 768] = [in (heads, row blocks) | out]; drop head 5's 64 rows.
    parts = [w[sl, :] for sl in keep_slices(HEAD_DIM)]
    return torch.cat(parts, dim=0)


def main() -> None:
    state_dict = load_file(INPUT_FILE)
    out = {}

    for key, tensor in state_dict.items():
        if key.endswith("attn.c_attn.weight"):
            out[key] = prune_c_attn_weight(tensor)
        elif key.endswith("attn.c_attn.bias"):
            out[key] = prune_c_attn_bias(tensor)
        elif key.endswith("attn.c_proj.weight"):
            out[key] = prune_c_proj_weight(tensor)
        else:
            out[key] = tensor

    # --- Required checks: fail loudly before writing ---
    def check_shape(name: str, expected: tuple[int, ...]) -> None:
        actual = tuple(out[name].shape)
        if actual != expected:
            raise AssertionError(f"{name}: expected shape {expected}, got {actual}")

    for i in range(N_LAYERS):
        check_shape(f"h.{i}.attn.c_attn.weight", (768, 2112))
        check_shape(f"h.{i}.attn.c_attn.bias", (2112,))
        check_shape(f"h.{i}.attn.c_proj.weight", (704, 768))
        # untouched, per the spec
        check_shape(f"h.{i}.attn.c_proj.bias", (768,))

    if len(out) != 160:
        raise AssertionError(f"expected 160 tensors, got {len(out)}")
    if len(state_dict) != 160:
        raise AssertionError(f"input had {len(state_dict)} tensors, expected 160")

    # --- Write ---
    out = {k: v.contiguous() for k, v in out.items()}
    save_file(out, OUTPUT_FILE)
    print(f"wrote {OUTPUT_FILE} with {len(out)} tensors")


if __name__ == "__main__":
    main()
