from __future__ import annotations

import torch

from ..ternary import (
    DestinationPolicy,
    ResolvedTernaryMapping,
    TernaryMappingSpec,
    TernaryMappingTransform,
)
from ..refs import TensorRef, parse_slice, select_tensor
from ..tensor_checks import require_same_shape_dtype_device3
from ..transform import (
    StateDictProvider,
    TransformError,
    register_transform,
)


class MultiplyTransformError(TransformError):
    pass


class MultiplyTransform(TernaryMappingTransform[TernaryMappingSpec]):
    name = "multiply"
    error_type = MultiplyTransformError
    spec_type = TernaryMappingSpec
    destination_policy = DestinationPolicy.MUST_EXIST
    progress_desc = "Applying multiply transforms"
    help_text = (
        "Computes elementwise multiplication from 'from_a' and 'from_b' into 'to'. "
        "Destination tensors must already exist.\n"
        "\n"
        "All references may include slicing and may be regex/structured mappings.\n"
        "\n"
        "Example:\n"
        "  multiply: { from_a: a.weight, from_b: b.weight, to: out.weight }"
    )

    def validate_refs(self, from_a_ref: TensorRef, from_b_ref: TensorRef, to_ref: TensorRef) -> None:
        if from_a_ref.slice_spec is not None:
            parse_slice(from_a_ref.slice_spec)
        if from_b_ref.slice_spec is not None:
            parse_slice(from_b_ref.slice_spec)
        if to_ref.slice_spec is not None:
            parse_slice(to_ref.slice_spec)

    def apply_mapping(self, item: ResolvedTernaryMapping, provider: StateDictProvider) -> None:
        src_a_sd = provider.get_state_dict(item.src_a_model)
        src_b_sd = provider.get_state_dict(item.src_b_model)
        dst_sd = provider.get_state_dict(item.dst_model)

        src_a_view = select_tensor(src_a_sd[item.src_a_name], item.src_a_slice)
        src_b_view = select_tensor(src_b_sd[item.src_b_name], item.src_b_slice)
        dst_view = select_tensor(dst_sd[item.dst_name], item.dst_slice)

        require_same_shape_dtype_device3(
            src_a_view,
            src_b_view,
            dst_view,
            error_type=MultiplyTransformError,
            op_name="multiplying",
            first_name=item.src_a_name,
            second_name=item.src_b_name,
            dest_name=item.dst_name,
            symbol="*",
        )
        dst_view.copy_(src_a_view * src_b_view)


def _unit_test_multiply_apply_success() -> None:
    class _Provider:
        def __init__(self) -> None:
            self._state_dict = {
                "a": torch.tensor([2.0, 3.0]),
                "b": torch.tensor([4.0, 5.0]),
                "dst": torch.tensor([0.0, 0.0]),
            }

        def get_state_dict(self, model: str):
            assert model == "m"
            return self._state_dict

    provider = _Provider()
    item = ResolvedTernaryMapping(
        src_a_model="m",
        src_a_name="a",
        src_a_slice=None,
        src_b_model="m",
        src_b_name="b",
        src_b_slice=None,
        dst_model="m",
        dst_name="dst",
        dst_slice=None,
    )
    MultiplyTransform().apply_mapping(item, provider)
    assert provider._state_dict["dst"].tolist() == [8.0, 15.0]


def _unit_test_multiply_compile_slices_allowed() -> None:
    spec = MultiplyTransform().compile(
        {"from_a": "a::[:2]", "from_b": "b::[:2]", "to": "c::[:2]"},
        default_model="m",
    )
    assert spec.from_a_ref.slice_spec == "[:2]"
    assert spec.from_b_ref.slice_spec == "[:2]"
    assert spec.to_ref.slice_spec == "[:2]"


def _unit_test_multiply_shape_mismatch() -> None:
    class _Provider:
        def __init__(self) -> None:
            self._state_dict = {
                "a": torch.tensor([1.0, 2.0]),
                "b": torch.tensor([2.0]),
                "dst": torch.tensor([0.0, 0.0]),
            }

        def get_state_dict(self, model: str):
            assert model == "m"
            return self._state_dict

    item = ResolvedTernaryMapping(
        src_a_model="m",
        src_a_name="a",
        src_a_slice=None,
        src_b_model="m",
        src_b_name="b",
        src_b_slice=None,
        dst_model="m",
        dst_name="dst",
        dst_slice=None,
    )
    try:
        MultiplyTransform().apply_mapping(item, _Provider())
    except MultiplyTransformError as exc:
        assert "shape mismatch" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected shape mismatch")


__unit_tests__ = [
    _unit_test_multiply_apply_success,
    _unit_test_multiply_compile_slices_allowed,
    _unit_test_multiply_shape_mismatch,
]


register_transform(MultiplyTransform())
