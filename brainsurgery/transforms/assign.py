from __future__ import annotations

import torch

from .binary import BinaryMappingSpec, BinaryMappingTransform, DestinationPolicy
from ..mappings import ResolvedMapping
from ..refs import TensorRef, parse_slice, select_tensor
from ..tensor_checks import require_same_shape_dtype_device
from ..transform import (
    StateDictProvider,
    TransformError,
    register_transform,
)


class AssignTransformError(TransformError):
    pass


class AssignTransform(BinaryMappingTransform[BinaryMappingSpec]):
    name = "assign"
    error_type = AssignTransformError
    spec_type = BinaryMappingSpec
    destination_policy = DestinationPolicy.MUST_EXIST
    progress_desc = "Applying assign transforms"
    help_text = (
        "Copies tensor values from 'from' into 'to'. The destination tensor must "
        "already exist. Source and destination (after slicing) must have identical shapes.\n"
        "\n"
        "Both references support slicing.\n"
        "\n"
        "Examples:\n"
        "  assign: { from: a.weight, to: b.weight }\n"
        "  assign: { from: 'a.weight::[:, :10]', to: 'b.weight::[:, :10]' }"
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
            error_type=AssignTransformError,
            op_name="assigning",
            left_name=item.src_name,
            right_name=item.dst_name,
        )

        dst_view.copy_(src_view)


def _unit_test_assign_dtype_compatibility() -> None:
    class _Provider:
        def __init__(self) -> None:
            self._state_dict = {
                "src": torch.ones((2, 2), dtype=torch.float32),
                "dst": torch.ones((2, 2), dtype=torch.float16),
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
        AssignTransform().apply_mapping(item, _Provider())
    except AssignTransformError as exc:
        assert "dtype mismatch" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected dtype mismatch error")


def _unit_test_assign_shape_compatibility() -> None:
    class _Provider:
        def __init__(self) -> None:
            self._state_dict = {
                "src": torch.ones((2, 2), dtype=torch.float32),
                "dst": torch.ones((3, 2), dtype=torch.float32),
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
        AssignTransform().apply_mapping(item, _Provider())
    except AssignTransformError as exc:
        assert "shape mismatch" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected shape mismatch error")


def _unit_test_assign_successful_copy() -> None:
    class _Provider:
        def __init__(self) -> None:
            self._state_dict = {
                "src": torch.tensor([1.0, 2.0], dtype=torch.float32),
                "dst": torch.tensor([0.0, 0.0], dtype=torch.float32),
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
    AssignTransform().apply_mapping(item, provider)
    assert torch.equal(provider._state_dict["dst"], provider._state_dict["src"])


__unit_tests__ = [
    _unit_test_assign_dtype_compatibility,
    _unit_test_assign_shape_compatibility,
    _unit_test_assign_successful_copy,
]


register_transform(AssignTransform())
