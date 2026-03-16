from dataclasses import dataclass

from ..core import (
    BinaryMappingSpec,
    BinaryRefs,
    DeclarativeBinaryTransform,
    DestinationPolicy,
    Docs,
    StateDictProvider,
    TensorRef,
    TransformError,
    register_transform,
    state_dict_for_ref,
    view_for_ref_name,
)
from ..engine import emit_verbose_binary_activity


@dataclass(frozen=True)
class PermuteSpec(BinaryMappingSpec):
    order: tuple[int, ...]


def _build_permute_spec(from_ref: TensorRef, to_ref: TensorRef, payload: dict) -> PermuteSpec:
    order = _parse_order(payload.get("order"))
    return PermuteSpec(from_ref=from_ref, to_ref=to_ref, order=order)


def _permute_apply(
    spec: PermuteSpec, src_name: str, dst_name: str, provider: StateDictProvider
) -> None:
    _src_sd, src_view = view_for_ref_name(provider, spec.from_ref, src_name)
    dst_sd = state_dict_for_ref(provider, spec.to_ref)
    order = spec.order
    if src_view.dim() != len(order):
        raise TransformError(
            f"permute.order rank mismatch for {src_name}: "
            f"tensor rank {src_view.dim()} vs order length {len(order)}"
        )
    if sorted(order) != list(range(src_view.dim())):
        raise TransformError(
            f"permute.order must be a permutation of [0..{src_view.dim() - 1}], got {list(order)}"
        )
    dst_sd[dst_name] = src_view.permute(*order).clone()
    emit_verbose_binary_activity("permute", src_name, dst_name)


class PermuteTransform(DeclarativeBinaryTransform[PermuteSpec]):
    name = "permute"
    error_type = TransformError
    spec_type = PermuteSpec
    destination_policy = DestinationPolicy.MUST_NOT_EXIST
    allowed_keys = {"from", "to", "order"}
    required_keys = {"from", "to", "order"}
    docs = Docs(
        "Permutes source tensor dimensions into new destination tensors.",
        notes=("'order' must be a full permutation of source dimensions.",),
        examples=("permute: { from: x, to: x_nhwc, order: [0, 2, 3, 1] }",),
    )
    refs = BinaryRefs(from_slice=True)
    spec_builder = staticmethod(_build_permute_spec)
    apply_fn = staticmethod(_permute_apply)


def _parse_order(raw: object) -> tuple[int, ...]:
    if not isinstance(raw, list) or not raw:
        raise TransformError("permute.order must be a non-empty list of integers")
    if not all(isinstance(x, int) for x in raw):
        raise TransformError("permute.order must be a non-empty list of integers")
    return tuple(raw)


register_transform(PermuteTransform())
