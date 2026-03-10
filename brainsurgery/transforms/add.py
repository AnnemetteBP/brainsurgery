from __future__ import annotations

import torch

from ..ternary import (
    DestinationPolicy,
    ResolvedTernaryMapping,
    TernaryMappingSpec,
    TernaryMappingTransform,
)
from ..tensor_checks import require_same_shape_dtype_device3
from ..transform import (
    StateDictProvider,
    TensorRef,
    TransformError,
    parse_slice,
    register_transform,
    select_tensor,
)


class AddTransformError(TransformError):
    pass


class AddTransform(TernaryMappingTransform[TernaryMappingSpec]):
    name = "add"
    error_type = AddTransformError
    spec_type = TernaryMappingSpec
    destination_policy = DestinationPolicy.MUST_EXIST
    progress_desc = "Applying add transforms"
    help_text = (
        "Computes elementwise addition from 'from_a' and 'from_b' into 'to'. "
        "Destination tensors must already exist.\n"
        "\n"
        "All references may include slicing and may be regex/structured mappings.\n"
        "\n"
        "Examples:\n"
        "  add: { from_a: a.weight, from_b: b.weight, to: out.weight }\n"
        "  add: { from_a: '.*.weight', from_b: '.*.delta', to: '.*.weight' }"
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
            error_type=AddTransformError,
            op_name="adding",
            first_name=item.src_a_name,
            second_name=item.src_b_name,
            dest_name=item.dst_name,
            symbol="+",
        )
        dst_view.copy_(src_a_view + src_b_view)


def _unit_test_add_apply_success() -> None:
    class _Provider:
        def __init__(self) -> None:
            self._state_dict = {
                "a": torch.tensor([1.0, 2.0]),
                "b": torch.tensor([3.0, 4.0]),
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
    AddTransform().apply_mapping(item, provider)
    assert provider._state_dict["dst"].tolist() == [4.0, 6.0]


def _unit_test_add_compile_requires_from_b() -> None:
    try:
        AddTransform().compile({"from_a": "a", "to": "b"}, default_model="m")
    except TransformError as exc:
        assert "from_b" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected required-key validation error")


def _unit_test_add_dtype_mismatch() -> None:
    class _Provider:
        def __init__(self) -> None:
            self._state_dict = {
                "a": torch.tensor([1.0, 2.0], dtype=torch.float32),
                "b": torch.tensor([3.0, 4.0], dtype=torch.float16),
                "dst": torch.tensor([0.0, 0.0], dtype=torch.float32),
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
        AddTransform().apply_mapping(item, _Provider())
    except AddTransformError as exc:
        assert "dtype mismatch" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected dtype mismatch")


__unit_tests__ = [
    _unit_test_add_apply_success,
    _unit_test_add_compile_requires_from_b,
    _unit_test_add_dtype_mismatch,
]


register_transform(AddTransform())
