from __future__ import annotations

import torch

from .binary import BinaryMappingSpec, BinaryMappingTransform, DestinationPolicy
from ..tensor_checks import require_same_shape_dtype_device
from ..transform import (
    ResolvedMapping,
    StateDictProvider,
    TensorRef,
    TransformError,
    parse_slice,
    register_transform,
    select_tensor,
)


class SubtractInPlaceTransformError(TransformError):
    pass


class SubtractInPlaceTransform(BinaryMappingTransform[BinaryMappingSpec]):
    name = "subtract_"
    error_type = SubtractInPlaceTransformError
    spec_type = BinaryMappingSpec
    destination_policy = DestinationPolicy.MUST_EXIST
    progress_desc = "Applying subtract_ transforms"
    help_text = (
        "Subtracts source tensors from destination tensors in-place.\n"
        "\n"
        "Computes: to <- to - from. The destination tensor must already exist.\n"
        "Both references may include slicing.\n"
        "\n"
        "Examples:\n"
        "  subtract_: { from: delta.weight, to: model.weight }\n"
        "  subtract_: { from: 'a::[:, :10]', to: 'b::[:, :10]' }"
    )

    def validate_refs(self, from_ref: TensorRef, to_ref: TensorRef) -> None:
        if from_ref.slice_spec is not None:
            parse_slice(from_ref.slice_spec)
        if to_ref.slice_spec is not None:
            parse_slice(to_ref.slice_spec)

    def apply_mapping(self, item: ResolvedMapping, provider: StateDictProvider) -> None:
        src_sd = provider.get_state_dict(item.src_model)
        dst_sd = provider.get_state_dict(item.dst_model)

        src_view = select_tensor(src_sd[item.src_name], item.src_slice)
        dst_view = select_tensor(dst_sd[item.dst_name], item.dst_slice)
        require_same_shape_dtype_device(
            src_view,
            dst_view,
            error_type=SubtractInPlaceTransformError,
            op_name="subtracting",
            left_name=item.src_name,
            right_name=item.dst_name,
        )

        dst_view.sub_(src_view)


def _unit_test_subtract_in_place_apply_success() -> None:
    class _Provider:
        def __init__(self) -> None:
            self._state_dict = {
                "src": torch.tensor([1.0, 2.0], dtype=torch.float32),
                "dst": torch.tensor([5.0, 7.0], dtype=torch.float32),
            }

        def get_state_dict(self, model: str):
            assert model == "model"
            return self._state_dict

    provider = _Provider()
    item = ResolvedMapping(
        src_model="model",
        src_name="src",
        src_slice=None,
        dst_model="model",
        dst_name="dst",
        dst_slice=None,
    )
    SubtractInPlaceTransform().apply_mapping(item, provider)
    assert provider._state_dict["dst"].tolist() == [4.0, 5.0]


def _unit_test_subtract_in_place_dtype_mismatch() -> None:
    class _Provider:
        def __init__(self) -> None:
            self._state_dict = {
                "src": torch.tensor([1.0, 2.0], dtype=torch.float16),
                "dst": torch.tensor([5.0, 7.0], dtype=torch.float32),
            }

        def get_state_dict(self, model: str):
            assert model == "model"
            return self._state_dict

    item = ResolvedMapping(
        src_model="model",
        src_name="src",
        src_slice=None,
        dst_model="model",
        dst_name="dst",
        dst_slice=None,
    )
    try:
        SubtractInPlaceTransform().apply_mapping(item, _Provider())
    except SubtractInPlaceTransformError as exc:
        assert "dtype mismatch" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected dtype mismatch")


def _unit_test_subtract_in_place_compile_accepts_slices() -> None:
    spec = SubtractInPlaceTransform().compile(
        {"from": "a::[:2]", "to": "b::[:2]"},
        default_model="model",
    )
    assert spec.from_ref.slice_spec == "[:2]"
    assert spec.to_ref.slice_spec == "[:2]"


__unit_tests__ = [
    _unit_test_subtract_in_place_apply_success,
    _unit_test_subtract_in_place_dtype_mismatch,
    _unit_test_subtract_in_place_compile_accepts_slices,
]


register_transform(SubtractInPlaceTransform())
