from __future__ import annotations

import torch

from ..ternary import (
    DestinationPolicy,
    ResolvedTernaryMapping,
    TernaryMappingSpec,
    TernaryMappingTransform,
)
from ..transform import (
    StateDictProvider,
    TensorRef,
    TransformError,
    parse_slice,
    register_transform,
    select_tensor,
)


class MatmulTransformError(TransformError):
    pass


class MatmulTransform(TernaryMappingTransform[TernaryMappingSpec]):
    name = "matmul"
    error_type = MatmulTransformError
    spec_type = TernaryMappingSpec
    destination_policy = DestinationPolicy.MUST_NOT_EXIST
    progress_desc = "Applying matmul transforms"
    help_text = (
        "Computes matrix multiplication from 'from_a' and 'from_b' into 'to'. "
        "Destination tensors must not already exist.\n"
        "\n"
        "All references may include slicing and may be regex/structured mappings.\n"
        "\n"
        "Example:\n"
        "  matmul: { from_a: a.weight, from_b: b.weight, to: out.weight }"
    )

    def validate_refs(self, from_a_ref: TensorRef, from_b_ref: TensorRef, to_ref: TensorRef) -> None:
        if from_a_ref.slice_spec is not None:
            parse_slice(from_a_ref.slice_spec)
        if from_b_ref.slice_spec is not None:
            parse_slice(from_b_ref.slice_spec)
        if to_ref.slice_spec is not None:
            raise MatmulTransformError("matmul destination must not be sliced")

    def apply_mapping(self, item: ResolvedTernaryMapping, provider: StateDictProvider) -> None:
        src_a_sd = provider.get_state_dict(item.src_a_model)
        src_b_sd = provider.get_state_dict(item.src_b_model)
        dst_sd = provider.get_state_dict(item.dst_model)

        src_a_view = select_tensor(src_a_sd[item.src_a_name], item.src_a_slice)
        src_b_view = select_tensor(src_b_sd[item.src_b_name], item.src_b_slice)

        if src_a_view.dtype != src_b_view.dtype:
            raise MatmulTransformError(
                f"dtype mismatch matmul {item.src_a_name} @ {item.src_b_name}: "
                f"{src_a_view.dtype} != {src_b_view.dtype}"
            )
        if src_a_view.device != src_b_view.device:
            raise MatmulTransformError(
                f"device mismatch matmul {item.src_a_name} @ {item.src_b_name}: "
                f"{src_a_view.device} != {src_b_view.device}"
            )

        try:
            result = torch.matmul(src_a_view, src_b_view)
        except RuntimeError as exc:
            raise MatmulTransformError(
                f"shape mismatch matmul {item.src_a_name} @ {item.src_b_name}: {exc}"
            ) from exc

        dst_sd[item.dst_name] = result.clone()


def _unit_test_matmul_compile_rejects_sliced_destination() -> None:
    try:
        MatmulTransform().compile(
            {"from_a": "a", "from_b": "b", "to": "c::[:]"},
            default_model="m",
        )
    except MatmulTransformError as exc:
        assert "destination must not be sliced" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected sliced destination rejection")


def _unit_test_matmul_apply_success() -> None:
    class _Provider:
        def __init__(self) -> None:
            self._state_dict = {
                "a": torch.tensor([[1.0, 2.0]], dtype=torch.float32),
                "b": torch.tensor([[3.0], [4.0]], dtype=torch.float32),
            }

        def get_state_dict(self, model: str):
            assert model == "m"
            return self._state_dict

    provider = _Provider()
    spec = MatmulTransform().compile({"from_a": "a", "from_b": "b", "to": "c"}, default_model="m")
    MatmulTransform().apply(spec, provider)
    assert provider._state_dict["c"].tolist() == [[11.0]]


__unit_tests__ = [
    _unit_test_matmul_compile_rejects_sliced_destination,
    _unit_test_matmul_apply_success,
]


register_transform(MatmulTransform())
