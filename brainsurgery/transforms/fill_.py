from __future__ import annotations

from dataclasses import dataclass

from .fill import FillConfig, build_filled_tensor_like, parse_fill_config
from .unary import UnarySpec
from ..core import TensorRef, must_model, parse_slice, select_tensor
from ..core import (
    register_transform,
)
from ..core import StateDictProvider, TransformError, note_tensor_write
from ..utils import DeclarativeUnaryTransform, Docs, UnaryRefs


@dataclass(frozen=True)
class FillInPlaceSpec(UnarySpec):
    config: FillConfig


def _build_fill_in_place_spec(target_ref: TensorRef, payload: dict) -> FillInPlaceSpec:
    config = parse_fill_config(payload, op_name="fill_", error_type=TransformError)
    return FillInPlaceSpec(target_ref=target_ref, config=config)


def _fill_in_place_apply(
    spec: FillInPlaceSpec, name: str, provider: StateDictProvider
) -> None:
    model = must_model(spec.target_ref)
    sd = provider.get_state_dict(model)
    slice_spec = (
        parse_slice(spec.target_ref.slice_spec)
        if spec.target_ref.slice_spec is not None
        else None
    )
    view = select_tensor(sd[name], slice_spec)
    filled = build_filled_tensor_like(view, spec.config, TransformError)
    view.copy_(filled)
    note_tensor_write(sd, name)


class FillInPlaceTransform(DeclarativeUnaryTransform[FillInPlaceSpec]):
    name = "fill_"
    error_type = TransformError
    spec_type = FillInPlaceSpec
    allowed_keys = {
        "target",
        "mode",
        "value",
        "values",
        "distribution",
        "low",
        "high",
        "mean",
        "std",
        "seed",
    }
    required_keys = {"target", "mode"}
    docs = Docs(
        "Fills target tensors in-place.",
        notes=(
            "Modes:",
            "  - constant: uses scalar 'value'",
            "  - rand: random fill ('distribution': uniform|normal)",
            "  - tensor: uses concrete payload 'values' (broadcasted if needed)",
        ),
        examples=("fill_: { target: x, mode: constant, value: 0 }",),
    )
    refs = UnaryRefs(target_slice=True)
    spec_builder = staticmethod(_build_fill_in_place_spec)
    apply_fn = staticmethod(_fill_in_place_apply)


register_transform(FillInPlaceTransform())
