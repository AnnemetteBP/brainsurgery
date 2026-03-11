from __future__ import annotations

from dataclasses import dataclass

from .unary import UnarySpec
from .reshape import _parse_shape
from ..core import TensorRef, must_model
from ..core import (
    StateDictProvider,
    TransformError,
    register_transform,
)
from ..utils.transforms import DeclarativeUnaryTransform, Docs, UnaryRefs


@dataclass(frozen=True)
class ReshapeInPlaceSpec(UnarySpec):
    shape: tuple[int, ...]


def _build_reshape_in_place_spec(
    target_ref: TensorRef, payload: dict
) -> ReshapeInPlaceSpec:
    shape = _parse_shape(
        payload.get("shape"), op_name="reshape_", error_type=TransformError
    )
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
