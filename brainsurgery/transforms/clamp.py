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
    register_transform,
    require_numeric,
    state_dict_for_ref,
    unary_view_for_ref_name,
    view_for_ref_name,
)
from ..engine import emit_verbose_binary_activity, emit_verbose_unary_activity


@dataclass(frozen=True)
class ClampSpec(BinaryMappingSpec):
    min_value: float | None
    max_value: float | None


def _build_clamp_spec(from_ref: TensorRef, to_ref: TensorRef, payload: dict) -> ClampSpec:
    min_value, max_value = _parse_bounds(payload, "clamp", TransformError)
    return ClampSpec(from_ref=from_ref, to_ref=to_ref, min_value=min_value, max_value=max_value)


def _clamp_apply(
    spec: ClampSpec, src_name: str, dst_name: str, provider: StateDictProvider
) -> None:
    _src_sd, src_view = view_for_ref_name(provider, spec.from_ref, src_name)
    dst_sd = state_dict_for_ref(provider, spec.to_ref)
    dst_sd[dst_name] = src_view.clamp(min=spec.min_value, max=spec.max_value).clone()
    emit_verbose_binary_activity("clamp", src_name, dst_name)


class ClampTransform(DeclarativeBinaryTransform[ClampSpec]):
    name = "clamp"
    error_type = TransformError
    spec_type = ClampSpec
    destination_policy = DestinationPolicy.MUST_NOT_EXIST
    allowed_keys = {"from", "to", "min", "max"}
    required_keys = {"from", "to"}
    docs = Docs(
        "Clamps source tensors into new destination tensors.",
        notes=("At least one of 'min' or 'max' is required.",),
        examples=("clamp: { from: x, to: x_clamped, min: -1.0, max: 1.0 }",),
    )
    refs = BinaryRefs(from_slice=True)
    spec_builder = staticmethod(_build_clamp_spec)
    apply_fn = staticmethod(_clamp_apply)


def _parse_bounds(
    payload: dict,
    op_name: str,
    error_type: type[TransformError],
) -> tuple[float | None, float | None]:
    min_value = require_numeric(payload, op_name=op_name, key="min") if "min" in payload else None
    max_value = require_numeric(payload, op_name=op_name, key="max") if "max" in payload else None
    if min_value is None and max_value is None:
        raise error_type(f"{op_name} requires at least one of: min, max")
    if min_value is not None and max_value is not None and min_value > max_value:
        raise error_type(f"{op_name}.min must be <= {op_name}.max")
    return min_value, max_value


register_transform(ClampTransform())


@dataclass(frozen=True)
class ClampInPlaceSpec(UnarySpec):
    min_value: float | None
    max_value: float | None


def _build_clamp_in_place_spec(target_ref: TensorRef, payload: dict) -> ClampInPlaceSpec:
    min_value, max_value = _parse_bounds(payload, "clamp_", TransformError)
    return ClampInPlaceSpec(target_ref=target_ref, min_value=min_value, max_value=max_value)


def _clamp_in_place_apply(spec: ClampInPlaceSpec, name: str, provider: StateDictProvider) -> None:
    sd, view = unary_view_for_ref_name(provider, spec.target_ref, name)
    view.clamp_(min=spec.min_value, max=spec.max_value)
    sd.mark_write(name)
    emit_verbose_unary_activity("clamp_", name)


class ClampInPlaceTransform(DeclarativeUnaryTransform[ClampInPlaceSpec]):
    name = "clamp_"
    error_type = TransformError
    spec_type = ClampInPlaceSpec
    allowed_keys = {"target", "min", "max"}
    required_keys = {"target"}
    docs = Docs(
        "Clamps target tensors in-place.",
        notes=("At least one of 'min' or 'max' is required.",),
        examples=("clamp_: { target: x, min: -1.0, max: 1.0 }",),
    )
    refs = UnaryRefs(target_slice=True)
    spec_builder = staticmethod(_build_clamp_in_place_spec)
    apply_fn = staticmethod(_clamp_in_place_apply)


register_transform(ClampInPlaceTransform())
