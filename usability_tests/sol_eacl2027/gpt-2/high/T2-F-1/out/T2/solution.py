from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


INPUT = Path("inputs/base/model.safetensors")
OUTPUT = Path("out/T2/model.safetensors")
LAYERS = 12
HIDDEN_SIZE = 768
HEAD_SIZE = 64
HEAD_TO_REMOVE = 5
EXPECTED_TENSORS = 160


def without_head_indices(segment_width: int = HIDDEN_SIZE) -> torch.Tensor:
    """Indices for one Q/K/V segment after removing the requested head."""
    start = HEAD_TO_REMOVE * HEAD_SIZE
    return torch.cat(
        (
            torch.arange(0, start, dtype=torch.long),
            torch.arange(start + HEAD_SIZE, segment_width, dtype=torch.long),
        )
    )


def main() -> None:
    state = load_file(INPUT, device="cpu")
    if len(state) != EXPECTED_TENSORS:
        raise ValueError(
            f"expected {EXPECTED_TENSORS} input tensors, found {len(state)}"
        )

    per_segment = without_head_indices()
    # c_attn is fused [q | k | v], so apply the same head removal separately
    # within all three 768-wide segments and retain q, k, v ordering.
    fused_indices = torch.cat(
        tuple(per_segment + segment * HIDDEN_SIZE for segment in range(3))
    )

    output = dict(state)
    for layer in range(LAYERS):
        prefix = f"h.{layer}.attn"
        c_attn_weight = f"{prefix}.c_attn.weight"
        c_attn_bias = f"{prefix}.c_attn.bias"
        c_proj_weight = f"{prefix}.c_proj.weight"

        expected_inputs = {
            c_attn_weight: (HIDDEN_SIZE, 3 * HIDDEN_SIZE),
            c_attn_bias: (3 * HIDDEN_SIZE,),
            c_proj_weight: (HIDDEN_SIZE, HIDDEN_SIZE),
        }
        for name, expected_shape in expected_inputs.items():
            if name not in state:
                raise KeyError(f"missing required tensor: {name}")
            if tuple(state[name].shape) != expected_shape:
                raise ValueError(
                    f"{name}: expected input shape {expected_shape}, "
                    f"found {tuple(state[name].shape)}"
                )

        output[c_attn_weight] = state[c_attn_weight].index_select(1, fused_indices)
        output[c_attn_bias] = state[c_attn_bias].index_select(0, fused_indices)
        output[c_proj_weight] = state[c_proj_weight].index_select(0, per_segment)

    # Required pre-write checks.
    required_shapes = {
        "h.0.attn.c_attn.weight": (768, 2112),
        "h.0.attn.c_attn.bias": (2112,),
        "h.0.attn.c_proj.weight": (704, 768),
    }
    for name, expected_shape in required_shapes.items():
        actual_shape = tuple(output[name].shape)
        if actual_shape != expected_shape:
            raise AssertionError(
                f"{name}: expected output shape {expected_shape}, found {actual_shape}"
            )
    if len(output) != EXPECTED_TENSORS:
        raise AssertionError(
            f"expected {EXPECTED_TENSORS} output tensors, found {len(output)}"
        )

    # Also enforce the same output shape contract in every layer.
    for layer in range(LAYERS):
        prefix = f"h.{layer}.attn"
        expected = {
            f"{prefix}.c_attn.weight": (768, 2112),
            f"{prefix}.c_attn.bias": (2112,),
            f"{prefix}.c_proj.weight": (704, 768),
        }
        for name, expected_shape in expected.items():
            if tuple(output[name].shape) != expected_shape:
                raise AssertionError(
                    f"{name}: expected output shape {expected_shape}, "
                    f"found {tuple(output[name].shape)}"
                )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    save_file(output, OUTPUT)

    # Reopen the serialized result so a truncated or malformed save fails loudly.
    saved = load_file(OUTPUT, device="cpu")
    if len(saved) != EXPECTED_TENSORS:
        raise AssertionError(
            f"saved file has {len(saved)} tensors, expected {EXPECTED_TENSORS}"
        )
    for name, expected_shape in required_shapes.items():
        if tuple(saved[name].shape) != expected_shape:
            raise AssertionError(
                f"saved {name}: expected shape {expected_shape}, "
                f"found {tuple(saved[name].shape)}"
            )

    print(f"wrote {OUTPUT} with {len(saved)} tensors")


if __name__ == "__main__":
    main()
