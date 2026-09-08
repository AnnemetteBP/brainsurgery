"""Remove attention head 5 from every Pythia-1B transformer layer."""

from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file


INPUT = Path("inputs/base/model.safetensors")
OUTPUT = Path("out/T2/model.safetensors")
LAYERS = 16
EXPECTED_TENSORS = 244

QKV_INPUT_SHAPE = (6144, 2048)
QKV_OUTPUT_SHAPE = (5376, 2048)
QKV_BIAS_INPUT_SHAPE = (6144,)
QKV_BIAS_OUTPUT_SHAPE = (5376,)
DENSE_INPUT_SHAPE = (2048, 2048)
DENSE_OUTPUT_SHAPE = (2048, 1792)

# Head 5 is rows 5*768:6*768 in fused QKV, and columns
# 5*256:6*256 in the output projection.
QKV_CUT = (3840, 4608)
DENSE_CUT = (1280, 1536)


def checked_cat(tensor: torch.Tensor, axis: int, cut: tuple[int, int]) -> torch.Tensor:
    """Remove one interval on an axis and return contiguous storage."""
    before = tensor.narrow(axis, 0, cut[0])
    after = tensor.narrow(axis, cut[1], tensor.shape[axis] - cut[1])
    return torch.cat((before, after), dim=axis).contiguous()


def main() -> None:
    if not INPUT.is_file():
        raise FileNotFoundError(f"missing input checkpoint: {INPUT}")

    tensors = load_file(INPUT, device="cpu")
    if len(tensors) != EXPECTED_TENSORS:
        raise AssertionError(
            f"input contains {len(tensors)} tensors, expected {EXPECTED_TENSORS}"
        )

    original_keys = set(tensors)
    changed_keys: set[str] = set()

    for layer in range(LAYERS):
        prefix = f"gpt_neox.layers.{layer}.attention"
        qkv_weight_key = f"{prefix}.query_key_value.weight"
        qkv_bias_key = f"{prefix}.query_key_value.bias"
        dense_weight_key = f"{prefix}.dense.weight"

        required = {qkv_weight_key, qkv_bias_key, dense_weight_key}
        missing = required - tensors.keys()
        if missing:
            raise KeyError(f"layer {layer} is missing tensors: {sorted(missing)}")

        qkv_weight = tensors[qkv_weight_key]
        qkv_bias = tensors[qkv_bias_key]
        dense_weight = tensors[dense_weight_key]
        actual_shapes = (
            tuple(qkv_weight.shape),
            tuple(qkv_bias.shape),
            tuple(dense_weight.shape),
        )
        expected_shapes = (
            QKV_INPUT_SHAPE,
            QKV_BIAS_INPUT_SHAPE,
            DENSE_INPUT_SHAPE,
        )
        if actual_shapes != expected_shapes:
            raise AssertionError(
                f"layer {layer} input shapes {actual_shapes}, expected {expected_shapes}"
            )

        new_qkv_weight = checked_cat(qkv_weight, 0, QKV_CUT)
        new_qkv_bias = checked_cat(qkv_bias, 0, QKV_CUT)
        new_dense_weight = checked_cat(dense_weight, 1, DENSE_CUT)

        # Check both resulting shapes and exact retained-slice order.
        if tuple(new_qkv_weight.shape) != QKV_OUTPUT_SHAPE:
            raise AssertionError(f"layer {layer} bad QKV weight output shape")
        if tuple(new_qkv_bias.shape) != QKV_BIAS_OUTPUT_SHAPE:
            raise AssertionError(f"layer {layer} bad QKV bias output shape")
        if tuple(new_dense_weight.shape) != DENSE_OUTPUT_SHAPE:
            raise AssertionError(f"layer {layer} bad dense weight output shape")
        if not torch.equal(new_qkv_weight[:3840], qkv_weight[:3840]):
            raise AssertionError(f"layer {layer} QKV weight prefix changed")
        if not torch.equal(new_qkv_weight[3840:], qkv_weight[4608:]):
            raise AssertionError(f"layer {layer} QKV weight suffix changed")
        if not torch.equal(new_qkv_bias[:3840], qkv_bias[:3840]):
            raise AssertionError(f"layer {layer} QKV bias prefix changed")
        if not torch.equal(new_qkv_bias[3840:], qkv_bias[4608:]):
            raise AssertionError(f"layer {layer} QKV bias suffix changed")
        if not torch.equal(new_dense_weight[:, :1280], dense_weight[:, :1280]):
            raise AssertionError(f"layer {layer} dense weight prefix changed")
        if not torch.equal(new_dense_weight[:, 1280:], dense_weight[:, 1536:]):
            raise AssertionError(f"layer {layer} dense weight suffix changed")

        tensors[qkv_weight_key] = new_qkv_weight
        tensors[qkv_bias_key] = new_qkv_bias
        tensors[dense_weight_key] = new_dense_weight
        changed_keys.update(required)

    # Required pre-write checks from TASK.md, plus key and edit-count checks.
    layer0 = "gpt_neox.layers.0.attention"
    assert tuple(tensors[f"{layer0}.query_key_value.weight"].shape) == (5376, 2048)
    assert tuple(tensors[f"{layer0}.query_key_value.bias"].shape) == (5376,)
    assert tuple(tensors[f"{layer0}.dense.weight"].shape) == (2048, 1792)
    assert len(tensors) == EXPECTED_TENSORS
    assert set(tensors) == original_keys
    assert len(changed_keys) == LAYERS * 3

    with safe_open(INPUT, framework="pt", device="cpu") as source:
        metadata = source.metadata()
    save_file(tensors, OUTPUT, metadata=metadata)

    # Re-open the serialized artifact so file-level failures cannot go unnoticed.
    written = load_file(OUTPUT, device="cpu")
    assert len(written) == EXPECTED_TENSORS
    assert set(written) == original_keys
    assert tuple(written[f"{layer0}.query_key_value.weight"].shape) == (5376, 2048)
    assert tuple(written[f"{layer0}.query_key_value.bias"].shape) == (5376,)
    assert tuple(written[f"{layer0}.dense.weight"].shape) == (2048, 1792)

    print(f"wrote {OUTPUT} with {len(written)} tensors")


if __name__ == "__main__":
    main()
