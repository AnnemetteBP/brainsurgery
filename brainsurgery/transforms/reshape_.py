from __future__ import annotations

from dataclasses import dataclass

import torch

from .unary import UnarySpec, UnaryTransform
from .reshape import _parse_shape
from ..refs import TensorRef, must_model
from ..transform import (
    StateDictProvider,
    TransformError,
    register_transform,
)


class ReshapeInPlaceTransformError(TransformError):
    pass


@dataclass(frozen=True)
class ReshapeInPlaceSpec(UnarySpec):
    shape: tuple[int, ...]


class ReshapeInPlaceTransform(UnaryTransform[ReshapeInPlaceSpec]):
    name = "reshape_"
    error_type = ReshapeInPlaceTransformError
    spec_type = ReshapeInPlaceSpec
    allowed_keys = {"target", "shape"}
    required_keys = {"target", "shape"}
    progress_desc = "Applying reshape_ transforms"
    help_text = (
        "Reshapes target tensors in-place (rebinds the tensor at the same name).\n"
        "\n"
        "Slicing is not supported for reshape_. Shape may include one '-1'.\n"
        "\n"
        "Example:\n"
        "  reshape_: { target: x, shape: [1024, -1] }"
    )

    def build_spec(self, target_ref: TensorRef, payload: dict) -> ReshapeInPlaceSpec:
        shape = _parse_shape(payload.get("shape"), op_name=self.name, error_type=ReshapeInPlaceTransformError)
        return ReshapeInPlaceSpec(target_ref=target_ref, shape=shape)

    def apply_to_target(self, spec: ReshapeInPlaceSpec, name: str, provider: StateDictProvider) -> None:
        model = must_model(spec.target_ref)
        sd = provider.get_state_dict(model)
        try:
            sd[name] = sd[name].reshape(spec.shape).clone()
        except RuntimeError as exc:
            raise ReshapeInPlaceTransformError(
                f"reshape_ failed for {model}::{name}: {exc}"
            ) from exc






register_transform(ReshapeInPlaceTransform())
