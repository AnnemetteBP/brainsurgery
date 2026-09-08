from pathlib import Path

from safetensors.torch import load_file, save_file


INPUT = Path("inputs/base/model.safetensors")
OUTPUT = Path("out/T2/model.safetensors")
LAYERS = 12
HIDDEN_SIZE = 768
HEAD_SIZE = 64
HEAD_TO_REMOVE = 5


def keep_indices(segment_width: int, segments: int):
    """Indices retaining every head except HEAD_TO_REMOVE in each segment."""
    start = HEAD_TO_REMOVE * HEAD_SIZE
    stop = start + HEAD_SIZE
    indices = []
    for segment in range(segments):
        base = segment * segment_width
        indices.extend(range(base, base + start))
        indices.extend(range(base + stop, base + segment_width))
    return indices


def main() -> None:
    tensors = load_file(INPUT, device="cpu")
    if len(tensors) != 160:
        raise ValueError(f"expected 160 input tensors, found {len(tensors)}")

    qkv_keep = keep_indices(HIDDEN_SIZE, segments=3)
    output_keep = keep_indices(HIDDEN_SIZE, segments=1)

    for layer in range(LAYERS):
        prefix = f"h.{layer}.attn"
        weight_key = f"{prefix}.c_attn.weight"
        bias_key = f"{prefix}.c_attn.bias"
        projection_key = f"{prefix}.c_proj.weight"

        if tuple(tensors[weight_key].shape) != (768, 2304):
            raise ValueError(f"unexpected shape for {weight_key}: {tensors[weight_key].shape}")
        if tuple(tensors[bias_key].shape) != (2304,):
            raise ValueError(f"unexpected shape for {bias_key}: {tensors[bias_key].shape}")
        if tuple(tensors[projection_key].shape) != (768, 768):
            raise ValueError(
                f"unexpected shape for {projection_key}: {tensors[projection_key].shape}"
            )

        tensors[weight_key] = tensors[weight_key][:, qkv_keep].contiguous()
        tensors[bias_key] = tensors[bias_key][qkv_keep].contiguous()
        tensors[projection_key] = tensors[projection_key][output_keep, :].contiguous()

    # Required pre-write checks: abort instead of serializing an invalid artifact.
    assert tuple(tensors["h.0.attn.c_attn.weight"].shape) == (768, 2112)
    assert tuple(tensors["h.0.attn.c_attn.bias"].shape) == (2112,)
    assert tuple(tensors["h.0.attn.c_proj.weight"].shape) == (704, 768)
    assert len(tensors) == 160

    save_file(tensors, OUTPUT)


if __name__ == "__main__":
    main()
