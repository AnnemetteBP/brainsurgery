from __future__ import annotations

from dataclasses import dataclass

from ..core import ResolvedMapping, StateDictProvider, TensorRef, TransformError, must_model, parse_slice, select_tensor
from ..core import register_transform
from ..core import require_numeric
from ..core import BinaryMappingSpec, DestinationPolicy
from ..core import BinaryRefs, DeclarativeBinaryTransform, Docs, UnaryRefs, UnarySpec, DeclarativeUnaryTransform


@dataclass(frozen=True)
class ScaleSpec(BinaryMappingSpec):
    factor: float


def _build_scale_spec(
    from_ref: TensorRef, to_ref: TensorRef, payload: dict
) -> ScaleSpec:
    factor = require_numeric(payload, op_name="scale", key="by")
    return ScaleSpec(from_ref=from_ref, to_ref=to_ref, factor=factor)


def _scale_apply(
    spec: ScaleSpec, item: ResolvedMapping, provider: StateDictProvider
) -> None:
    src_sd = provider.get_state_dict(item.src_model)
    dst_sd = provider.get_state_dict(item.dst_model)
    scaled = select_tensor(src_sd[item.src_name], item.src_slice).clone()
    scaled.mul_(spec.factor)
    dst_sd[item.dst_name] = scaled


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
            "scale: { from: 'h.0.attn.c_attn.weight::[:, :10]', to: h.0.attn.c_attn.partial, by: 2.0 }",
        ),
    )
    refs = BinaryRefs(from_slice=True)
    spec_builder = staticmethod(_build_scale_spec)
    apply_fn = staticmethod(_scale_apply)


register_transform(ScaleTransform())


@dataclass(frozen=True)
class ScaleInPlaceSpec(UnarySpec):
    factor: float


def _build_scale_in_place_spec(
    target_ref: TensorRef, payload: dict
) -> ScaleInPlaceSpec:
    factor = require_numeric(payload, op_name="scale_", key="by")
    return ScaleInPlaceSpec(target_ref=target_ref, factor=factor)


def _scale_in_place_apply(
    spec: ScaleInPlaceSpec, name: str, provider: StateDictProvider
) -> None:
    model = must_model(spec.target_ref)
    sd = provider.get_state_dict(model)
    tensor = sd[name]

    slice_spec = (
        parse_slice(spec.target_ref.slice_spec)
        if spec.target_ref.slice_spec is not None
        else None
    )
    view = select_tensor(tensor, slice_spec)
    view.mul_(spec.factor)
    sd.mark_write(name)


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
