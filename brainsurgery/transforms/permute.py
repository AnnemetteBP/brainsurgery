from __future__ import annotations

from dataclasses import dataclass

import torch

from .binary import BinaryMappingSpec, BinaryMappingTransform, DestinationPolicy
from ..transform import (
    ResolvedMapping,
    StateDictProvider,
    TensorRef,
    TransformError,
    ensure_mapping_payload,
    parse_slice,
    register_transform,
    select_tensor,
    validate_payload_keys,
)


class PermuteTransformError(TransformError):
    pass


@dataclass(frozen=True)
class PermuteSpec(BinaryMappingSpec):
    order: tuple[int, ...]


class PermuteTransform(BinaryMappingTransform[PermuteSpec]):
    name = "permute"
    error_type = PermuteTransformError
    spec_type = PermuteSpec
    destination_policy = DestinationPolicy.MUST_NOT_EXIST
    allowed_keys = {"from", "to", "order"}
    required_keys = {"from", "to", "order"}
    progress_desc = "Applying permute transforms"
    help_text = (
        "Permutes source tensor dimensions into new destination tensors.\n"
        "\n"
        "Source references may be sliced. Destination tensors must not exist and "
        "must not be sliced. 'order' must be a full permutation of source dimensions.\n"
        "\n"
        "Example:\n"
        "  permute: { from: x, to: x_nhwc, order: [0, 2, 3, 1] }"
    )

    def compile(self, payload: dict, default_model: str | None) -> PermuteSpec:
        payload = ensure_mapping_payload(payload, self.name)
        validate_payload_keys(
            payload,
            op_name=self.name,
            allowed_keys=self.allowed_keys,
            required_keys=self.required_keys,
        )
        from_ref, to_ref = self.parse_refs(payload, default_model)
        self.validate_refs(from_ref, to_ref)
        order = _parse_order(payload.get("order"))
        return PermuteSpec(from_ref=from_ref, to_ref=to_ref, order=order)

    def validate_refs(self, from_ref: TensorRef, to_ref: TensorRef) -> None:
        if from_ref.slice_spec is not None:
            parse_slice(from_ref.slice_spec)
        if to_ref.slice_spec is not None:
            raise PermuteTransformError("permute destination must not be sliced")

    def apply_item(self, spec: PermuteSpec, item: ResolvedMapping, provider: StateDictProvider) -> None:
        src_sd = provider.get_state_dict(item.src_model)
        dst_sd = provider.get_state_dict(item.dst_model)
        src_view = select_tensor(src_sd[item.src_name], item.src_slice)
        order = spec.order
        if src_view.dim() != len(order):
            raise PermuteTransformError(
                f"permute.order rank mismatch for {item.src_name}: "
                f"tensor rank {src_view.dim()} vs order length {len(order)}"
            )
        if sorted(order) != list(range(src_view.dim())):
            raise PermuteTransformError(
                f"permute.order must be a permutation of [0..{src_view.dim()-1}], got {list(order)}"
            )
        dst_sd[item.dst_name] = src_view.permute(*order).clone()


def _parse_order(raw: object) -> tuple[int, ...]:
    if not isinstance(raw, list) or not raw:
        raise PermuteTransformError("permute.order must be a non-empty list of integers")
    if not all(isinstance(x, int) for x in raw):
        raise PermuteTransformError("permute.order must be a non-empty list of integers")
    return tuple(raw)


def _unit_test_permute_compile_rejects_non_list_order() -> None:
    try:
        PermuteTransform().compile({"from": "x", "to": "y", "order": "01"}, default_model="m")
    except PermuteTransformError as exc:
        assert "non-empty list of integers" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected order validation error")


def _unit_test_permute_apply_success() -> None:
    class _Provider:
        def __init__(self) -> None:
            self._state_dict = {"x": torch.arange(6).reshape(2, 3)}

        def get_state_dict(self, model: str):
            assert model == "m"
            return self._state_dict

    provider = _Provider()
    spec = PermuteTransform().compile({"from": "x", "to": "y", "order": [1, 0]}, default_model="m")
    PermuteTransform().apply(spec, provider)
    assert provider._state_dict["y"].shape == (3, 2)


__unit_tests__ = [
    _unit_test_permute_compile_rejects_non_list_order,
    _unit_test_permute_apply_success,
]


register_transform(PermuteTransform())
