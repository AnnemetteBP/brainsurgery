from __future__ import annotations

from dataclasses import dataclass

from ..core import BinaryMappingSpec, DestinationPolicy
from ..core import ResolvedMapping, StateDictProvider, TensorRef, TransformError, select_tensor
from ..core import register_transform
from ..core import BinaryRefs, DeclarativeBinaryTransform, Docs


@dataclass(frozen=True)
class ReshapeSpec(BinaryMappingSpec):
    shape: tuple[int, ...]


def _build_reshape_spec(
    from_ref: TensorRef, to_ref: TensorRef, payload: dict
) -> ReshapeSpec:
    shape = _parse_shape(
        payload.get("shape"), op_name="reshape", error_type=TransformError
    )
    return ReshapeSpec(from_ref=from_ref, to_ref=to_ref, shape=shape)


def _reshape_apply(
    spec: ReshapeSpec, item: ResolvedMapping, provider: StateDictProvider
) -> None:
    src_sd = provider.get_state_dict(item.src_model)
    dst_sd = provider.get_state_dict(item.dst_model)
    src_view = select_tensor(src_sd[item.src_name], item.src_slice)
    try:
        dst_sd[item.dst_name] = src_view.reshape(spec.shape).clone()
    except RuntimeError as exc:
        raise TransformError(
            f"reshape failed for {item.src_name} -> {item.dst_name}: {exc}"
        ) from exc


class ReshapeTransform(DeclarativeBinaryTransform[ReshapeSpec]):
    name = "reshape"
    error_type = TransformError
    spec_type = ReshapeSpec
    destination_policy = DestinationPolicy.MUST_NOT_EXIST
    allowed_keys = {"from", "to", "shape"}
    required_keys = {"from", "to", "shape"}
    docs = Docs(
        "Reshapes source tensors into new destination tensors.",
        notes=("Shape may include one '-1'.",),
        examples=("reshape: { from: x, to: x2d, shape: [1024, -1] }",),
    )
    refs = BinaryRefs(from_slice=True)
    spec_builder = staticmethod(_build_reshape_spec)
    apply_fn = staticmethod(_reshape_apply)


def _parse_shape(
    raw: object, *, op_name: str, error_type: type[TransformError]
) -> tuple[int, ...]:
    if not isinstance(raw, list) or not raw:
        raise error_type(f"{op_name}.shape must be a non-empty list of integers")
    if not all(isinstance(x, int) for x in raw):
        raise error_type(f"{op_name}.shape must be a non-empty list of integers")
    if sum(1 for x in raw if x == -1) > 1:
        raise error_type(f"{op_name}.shape may include at most one '-1'")
    if any(x == 0 or x < -1 for x in raw):
        raise error_type(f"{op_name}.shape dimensions must be positive integers or -1")
    return tuple(raw)


register_transform(ReshapeTransform())
