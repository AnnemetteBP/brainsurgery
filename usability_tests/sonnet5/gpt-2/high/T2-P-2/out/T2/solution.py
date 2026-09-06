"""Structured attention-head pruning for GPT-2 (124M).

Removes head 5 (0-indexed) from every layer's attention block by slicing
the fused c_attn.{weight,bias} (column blocks) and c_proj.weight (row
blocks). Every other tensor is passed through unchanged.
"""

from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

HERE = Path(__file__).resolve().parent
INPUT_PATH = HERE.parent.parent / "inputs" / "base" / "model.safetensors"
OUTPUT_DIR = HERE
OUTPUT_PATH = OUTPUT_DIR / "model.safetensors"

NUM_LAYERS = 12
NUM_HEADS = 12
HEAD_DIM = 64
HIDDEN = NUM_HEADS * HEAD_DIM  # 768
HEAD_TO_PRUNE = 5

# Column/row ranges to KEEP within a single 768-wide head-block segment,
# i.e. everything except [head*64, (head+1)*64).
KEEP_LO = HEAD_TO_PRUNE * HEAD_DIM  # 320
KEEP_HI = (HEAD_TO_PRUNE + 1) * HEAD_DIM  # 384


def keep_indices_for_segment(offset: int) -> torch.Tensor:
    """Indices (absolute) to keep for one 768-wide segment starting at `offset`."""
    lo = torch.arange(offset, offset + KEEP_LO)
    hi = torch.arange(offset + KEEP_HI, offset + HIDDEN)
    return torch.cat([lo, hi])


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"input checkpoint not found: {INPUT_PATH}")

    state_dict = load_file(str(INPUT_PATH))

    if len(state_dict) != 160:
        raise AssertionError(f"expected 160 input tensors, found {len(state_dict)}")

    # Indices to keep for the fused c_attn tensors: q-segment, k-segment,
    # v-segment, each 768 wide, concatenated in order.
    qkv_keep = torch.cat(
        [keep_indices_for_segment(seg * HIDDEN) for seg in range(3)]
    )
    # Sanity-check against the exact ranges given in the task spec.
    expected_qkv_keep = torch.cat(
        [
            torch.arange(0, 320),
            torch.arange(384, 768),
            torch.arange(768, 1088),
            torch.arange(1152, 1536),
            torch.arange(1536, 1856),
            torch.arange(1920, 2304),
        ]
    )
    if not torch.equal(qkv_keep, expected_qkv_keep):
        raise AssertionError("computed c_attn keep-indices do not match task spec")

    # Indices to keep for c_proj.weight rows (single 768-wide head-block dim).
    proj_keep = keep_indices_for_segment(0)
    expected_proj_keep = torch.cat([torch.arange(0, 320), torch.arange(384, 768)])
    if not torch.equal(proj_keep, expected_proj_keep):
        raise AssertionError("computed c_proj keep-indices do not match task spec")

    output: dict[str, torch.Tensor] = {}
    for name, tensor in state_dict.items():
        is_pruned = False
        for i in range(NUM_LAYERS):
            if name == f"h.{i}.attn.c_attn.weight":
                if tensor.shape != (HIDDEN, 3 * HIDDEN):
                    raise AssertionError(f"{name}: unexpected shape {tuple(tensor.shape)}")
                output[name] = tensor[:, qkv_keep].contiguous()
                is_pruned = True
                break
            if name == f"h.{i}.attn.c_attn.bias":
                if tensor.shape != (3 * HIDDEN,):
                    raise AssertionError(f"{name}: unexpected shape {tuple(tensor.shape)}")
                output[name] = tensor[qkv_keep].contiguous()
                is_pruned = True
                break
            if name == f"h.{i}.attn.c_proj.weight":
                if tensor.shape != (HIDDEN, HIDDEN):
                    raise AssertionError(f"{name}: unexpected shape {tuple(tensor.shape)}")
                output[name] = tensor[proj_keep, :].contiguous()
                is_pruned = True
                break
        if not is_pruned:
            output[name] = tensor.clone()

    # Required checks: fail loudly before writing if anything is off.
    checks = {
        "h.0.attn.c_attn.weight": (HIDDEN, 2112),
        "h.0.attn.c_attn.bias": (2112,),
        "h.0.attn.c_proj.weight": (704, HIDDEN),
    }
    for name, expected_shape in checks.items():
        actual_shape = tuple(output[name].shape)
        if actual_shape != expected_shape:
            raise AssertionError(
                f"required check failed: {name} has shape {actual_shape}, "
                f"expected {expected_shape}"
            )

    if len(output) != 160:
        raise AssertionError(f"required check failed: output has {len(output)} tensors, expected 160")

    # Full per-layer shape/name checks (not just layer 0).
    for i in range(NUM_LAYERS):
        w = output[f"h.{i}.attn.c_attn.weight"]
        b = output[f"h.{i}.attn.c_attn.bias"]
        p = output[f"h.{i}.attn.c_proj.weight"]
        if tuple(w.shape) != (HIDDEN, 2112):
            raise AssertionError(f"h.{i}.attn.c_attn.weight has shape {tuple(w.shape)}")
        if tuple(b.shape) != (2112,):
            raise AssertionError(f"h.{i}.attn.c_attn.bias has shape {tuple(b.shape)}")
        if tuple(p.shape) != (704, HIDDEN):
            raise AssertionError(f"h.{i}.attn.c_proj.weight has shape {tuple(p.shape)}")
        # untouched tensors keep their original shape
        pb_name = f"h.{i}.attn.c_proj.bias"
        mask_name = f"h.{i}.attn.bias"
        if tuple(output[pb_name].shape) != (HIDDEN,):
            raise AssertionError(f"{pb_name} has shape {tuple(output[pb_name].shape)}")
        if tuple(output[mask_name].shape) != (1, 1, 1024, 1024):
            raise AssertionError(f"{mask_name} has shape {tuple(output[mask_name].shape)}")

    if set(output.keys()) != set(state_dict.keys()):
        raise AssertionError("output tensor names differ from input tensor names")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_file(output, str(OUTPUT_PATH))
    print(f"wrote {len(output)} tensors to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
