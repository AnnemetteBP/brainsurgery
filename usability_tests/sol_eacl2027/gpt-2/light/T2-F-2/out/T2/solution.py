from pathlib import Path

from safetensors.torch import load_file, save_file


INPUT = Path("inputs/base/model.safetensors")
OUTPUT = Path("out/T2/model.safetensors")
LAYERS = 12
HIDDEN = 768
HEAD_SIZE = 64
HEAD_TO_REMOVE = 5


def keep_one_segment(tensor, axis):
    """Remove the selected 64-wide head slice from one 768-wide segment."""
    start = HEAD_TO_REMOVE * HEAD_SIZE
    stop = start + HEAD_SIZE
    left = tensor.narrow(axis, 0, start)
    right = tensor.narrow(axis, stop, HIDDEN - stop)
    return __import__("torch").cat((left, right), dim=axis)


def main():
    tensors = load_file(INPUT)
    assert len(tensors) == 160, f"expected 160 input tensors, got {len(tensors)}"

    for layer in range(LAYERS):
        prefix = f"h.{layer}.attn"
        weight_key = f"{prefix}.c_attn.weight"
        bias_key = f"{prefix}.c_attn.bias"
        proj_key = f"{prefix}.c_proj.weight"

        weight = tensors[weight_key]
        bias = tensors[bias_key]
        proj = tensors[proj_key]
        assert tuple(weight.shape) == (768, 2304), (weight_key, weight.shape)
        assert tuple(bias.shape) == (2304,), (bias_key, bias.shape)
        assert tuple(proj.shape) == (768, 768), (proj_key, proj.shape)

        # c_attn is fused as q | k | v. Prune independently within each
        # 768-wide segment, then restore the original q, k, v ordering.
        tensors[weight_key] = __import__("torch").cat(
            [keep_one_segment(part, 1) for part in weight.split(HIDDEN, dim=1)],
            dim=1,
        )
        tensors[bias_key] = __import__("torch").cat(
            [keep_one_segment(part, 0) for part in bias.split(HIDDEN, dim=0)],
            dim=0,
        )
        tensors[proj_key] = keep_one_segment(proj, 0)

        assert tuple(tensors[weight_key].shape) == (768, 2112), weight_key
        assert tuple(tensors[bias_key].shape) == (2112,), bias_key
        assert tuple(tensors[proj_key].shape) == (704, 768), proj_key

    # Explicit required pre-write checks.
    assert tuple(tensors["h.0.attn.c_attn.weight"].shape) == (768, 2112)
    assert tuple(tensors["h.0.attn.c_attn.bias"].shape) == (2112,)
    assert tuple(tensors["h.0.attn.c_proj.weight"].shape) == (704, 768)
    assert len(tensors) == 160, f"expected 160 output tensors, got {len(tensors)}"

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, OUTPUT)


if __name__ == "__main__":
    main()
