from __future__ import annotations

from dataclasses import dataclass

from .unary import UnarySpec
from .clamp import _parse_bounds
from ..core import TensorRef, must_model, parse_slice, select_tensor
from ..core import (
    register_transform,
)
from ..core import StateDictProvider, TransformError, note_tensor_write
from ..utils.transforms import DeclarativeUnaryTransform, Docs, UnaryRefs


@dataclass(frozen=True)
class ClampInPlaceSpec(UnarySpec):
    min_value: float | None
    max_value: float | None


def _build_clamp_in_place_spec(
    target_ref: TensorRef, payload: dict
) -> ClampInPlaceSpec:
    min_value, max_value = _parse_bounds(payload, "clamp_", TransformError)
    return ClampInPlaceSpec(
        target_ref=target_ref, min_value=min_value, max_value=max_value
    )


def _clamp_in_place_apply(
    spec: ClampInPlaceSpec, name: str, provider: StateDictProvider
) -> None:
    model = must_model(spec.target_ref)
    sd = provider.get_state_dict(model)
    slice_spec = (
        parse_slice(spec.target_ref.slice_spec)
        if spec.target_ref.slice_spec is not None
        else None
    )
    view = select_tensor(sd[name], slice_spec)
    view.clamp_(min=spec.min_value, max=spec.max_value)
    note_tensor_write(sd, name)


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
