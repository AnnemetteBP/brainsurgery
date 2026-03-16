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
class ScaleSpec(BinaryMappingSpec):
    factor: float


def _build_scale_spec(from_ref: TensorRef, to_ref: TensorRef, payload: dict) -> ScaleSpec:
    factor = require_numeric(payload, op_name="scale", key="by")
    return ScaleSpec(from_ref=from_ref, to_ref=to_ref, factor=factor)


def _scale_apply(
    spec: ScaleSpec, src_name: str, dst_name: str, provider: StateDictProvider
) -> None:
    _src_sd, src_view = view_for_ref_name(provider, spec.from_ref, src_name)
    dst_sd = state_dict_for_ref(provider, spec.to_ref)
    scaled = src_view.clone()
    scaled.mul_(spec.factor)
    dst_sd[dst_name] = scaled
    emit_verbose_binary_activity("scale", src_name, dst_name)


class ScaleTransform(DeclarativeBinaryTransform[ScaleSpec]):
    name = "scale"
    error_type = TransformError
    spec_type = ScaleSpec
    destination_policy = DestinationPolicy.MUST_NOT_EXIST
    allowed_keys = {"from", "to", "by"}
    required_keys = {"from", "to", "by"}
    docs = Docs(
        "Scales source tensors by a numeric factor into new destination tensors.",
        examples=(
            "scale: { from: ln_f.weight, to: ln_f_half.weight, by: 0.5 }",
            "scale: { from: '.*bias', to: 'scaled.\\\\g<0>', by: -1 }",
            "scale: { from: 'h.0.attn.c_attn.weight::[:, :10]', "
            "to: h.0.attn.c_attn.partial, by: 2.0 }",
        ),
    )
    refs = BinaryRefs(from_slice=True)
    spec_builder = staticmethod(_build_scale_spec)
    apply_fn = staticmethod(_scale_apply)


register_transform(ScaleTransform())


@dataclass(frozen=True)
class ScaleInPlaceSpec(UnarySpec):
    factor: float


def _build_scale_in_place_spec(target_ref: TensorRef, payload: dict) -> ScaleInPlaceSpec:
    factor = require_numeric(payload, op_name="scale_", key="by")
    return ScaleInPlaceSpec(target_ref=target_ref, factor=factor)


def _scale_in_place_apply(spec: ScaleInPlaceSpec, name: str, provider: StateDictProvider) -> None:
    sd, view = unary_view_for_ref_name(provider, spec.target_ref, name)
    view.mul_(spec.factor)
    sd.mark_write(name)
    emit_verbose_unary_activity("scale_", name)


class ScaleInPlaceTransform(DeclarativeUnaryTransform[ScaleInPlaceSpec]):
    name = "scale_"
    error_type = TransformError
    spec_type = ScaleInPlaceSpec
    allowed_keys = {"target", "by"}
    required_keys = {"target", "by"}
    docs = Docs(
        "Scales tensors in-place by a numeric factor.",
        notes=("The selected tensor (or slice) is multiplied by 'by'.",),
        examples=(
            "scale_: { target: ln_f.weight, by: 0.5 }",
            "scale_: { target: '.*bias', by: -1 }",
            "scale_: { target: 'h.0.attn.c_attn.weight::[:, :10]', by: 2.0 }",
        ),
    )
    refs = UnaryRefs(target_slice=True)
    spec_builder = staticmethod(_build_scale_in_place_spec)
    apply_fn = staticmethod(_scale_in_place_apply)


register_transform(ScaleInPlaceTransform())
