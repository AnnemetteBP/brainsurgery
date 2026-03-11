from __future__ import annotations

from dataclasses import dataclass

from .unary import UnarySpec
from ..core import TensorRef, must_model, parse_slice, select_tensor
from ..core import (
    StateDictProvider,
    TransformError,
    register_transform,
    require_numeric,
)
from ..core import note_tensor_write
from ..utils.transforms import DeclarativeUnaryTransform, Docs, UnaryRefs


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
    note_tensor_write(sd, name)


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
