from dataclasses import dataclass

from ..core import (
    BinaryMappingSpec,
    BinaryRefs,
    DeclarativeBinaryTransform,
    DeclarativeUnaryTransform,
    DestinationPolicy,
    Docs,
    StateDictProvider,
    TensorRef,
    TransformError,
    UnaryRefs,
    UnarySpec,
    must_model,
    register_transform,
    state_dict_for_ref,
    view_for_ref_name,
)
from ..engine import emit_verbose_binary_activity, emit_verbose_unary_activity


@dataclass(frozen=True)
class ReshapeSpec(BinaryMappingSpec):
    shape: tuple[int, ...]


def _build_reshape_spec(from_ref: TensorRef, to_ref: TensorRef, payload: dict) -> ReshapeSpec:
    shape = _parse_shape(payload.get("shape"), op_name="reshape", error_type=TransformError)
    return ReshapeSpec(from_ref=from_ref, to_ref=to_ref, shape=shape)


def _reshape_apply(
    spec: ReshapeSpec, src_name: str, dst_name: str, provider: StateDictProvider
) -> None:
    _src_sd, src_view = view_for_ref_name(provider, spec.from_ref, src_name)
    dst_sd = state_dict_for_ref(provider, spec.to_ref)
    try:
        dst_sd[dst_name] = src_view.reshape(spec.shape).clone()
        emit_verbose_binary_activity("reshape", src_name, dst_name)
    except RuntimeError as exc:
        raise TransformError(f"reshape failed for {src_name} -> {dst_name}: {exc}") from exc


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


def _parse_shape(raw: object, *, op_name: str, error_type: type[TransformError]) -> tuple[int, ...]:
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


@dataclass(frozen=True)
class ReshapeInPlaceSpec(UnarySpec):
    shape: tuple[int, ...]


def _build_reshape_in_place_spec(target_ref: TensorRef, payload: dict) -> ReshapeInPlaceSpec:
    shape = _parse_shape(payload.get("shape"), op_name="reshape_", error_type=TransformError)
    return ReshapeInPlaceSpec(target_ref=target_ref, shape=shape)


def _reshape_in_place_apply(
    spec: ReshapeInPlaceSpec,
    name: str,
    provider: StateDictProvider,
) -> None:
    model = must_model(spec.target_ref)
    sd = provider.get_state_dict(model)
    try:
        sd[name] = sd[name].reshape(spec.shape).clone()
        emit_verbose_unary_activity("reshape_", name)
    except RuntimeError as exc:
        raise TransformError(f"reshape_ failed for {model}::{name}: {exc}") from exc


class ReshapeInPlaceTransform(DeclarativeUnaryTransform[ReshapeInPlaceSpec]):
    name = "reshape_"
    error_type = TransformError
    spec_type = ReshapeInPlaceSpec
    allowed_keys = {"target", "shape"}
    required_keys = {"target", "shape"}
    docs = Docs(
        "Reshapes target tensors in-place (rebinds the tensor at the same name).",
        notes=("Shape may include one '-1'.",),
        examples=("reshape_: { target: x, shape: [1024, -1] }",),
    )
    refs = UnaryRefs()
    spec_builder = staticmethod(_build_reshape_in_place_spec)
    apply_fn = staticmethod(_reshape_in_place_apply)


register_transform(ReshapeInPlaceTransform())
