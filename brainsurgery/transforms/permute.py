from __future__ import annotations

from dataclasses import dataclass

from ..core import BinaryMappingSpec, DestinationPolicy
from ..core import ResolvedMapping, StateDictProvider, TensorRef, TransformError, select_tensor
from ..core import register_transform
from ..core import BinaryRefs, DeclarativeBinaryTransform, Docs


@dataclass(frozen=True)
class PermuteSpec(BinaryMappingSpec):
    order: tuple[int, ...]


def _build_permute_spec(
    from_ref: TensorRef, to_ref: TensorRef, payload: dict
) -> PermuteSpec:
    order = _parse_order(payload.get("order"))
    return PermuteSpec(from_ref=from_ref, to_ref=to_ref, order=order)


def _permute_apply(
    spec: PermuteSpec, item: ResolvedMapping, provider: StateDictProvider
) -> None:
    src_sd = provider.get_state_dict(item.src_model)
    dst_sd = provider.get_state_dict(item.dst_model)
    src_view = select_tensor(src_sd[item.src_name], item.src_slice)
    order = spec.order
    if src_view.dim() != len(order):
        raise TransformError(
            f"permute.order rank mismatch for {item.src_name}: "
            f"tensor rank {src_view.dim()} vs order length {len(order)}"
        )
    if sorted(order) != list(range(src_view.dim())):
        raise TransformError(
            f"permute.order must be a permutation of [0..{src_view.dim() - 1}], got {list(order)}"
        )
    dst_sd[item.dst_name] = src_view.permute(*order).clone()


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
