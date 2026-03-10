from __future__ import annotations

from dataclasses import dataclass

import torch

from .unary import UnarySpec, UnaryTransform
from .clamp import _parse_bounds
from ..refs import TensorRef, must_model, parse_slice, select_tensor
from ..transform import (
    StateDictProvider,
    TransformError,
    register_transform,
)


class ClampInPlaceTransformError(TransformError):
    pass


@dataclass(frozen=True)
class ClampInPlaceSpec(UnarySpec):
    min_value: float | None
    max_value: float | None


class ClampInPlaceTransform(UnaryTransform[ClampInPlaceSpec]):
    name = "clamp_"
    error_type = ClampInPlaceTransformError
    spec_type = ClampInPlaceSpec
    allowed_keys = {"target", "min", "max"}
    required_keys = {"target"}
    slice_policy = "allow"
    progress_desc = "Applying clamp_ transforms"
    help_text = (
        "Clamps target tensors in-place.\n"
        "\n"
        "Targets may include slicing. At least one of 'min' or 'max' is required.\n"
        "\n"
        "Example:\n"
        "  clamp_: { target: x, min: -1.0, max: 1.0 }"
    )

    def build_spec(self, target_ref: TensorRef, payload: dict) -> ClampInPlaceSpec:
        min_value, max_value = _parse_bounds(payload, self.name, ClampInPlaceTransformError)
        return ClampInPlaceSpec(target_ref=target_ref, min_value=min_value, max_value=max_value)

    def apply_to_target(self, spec: ClampInPlaceSpec, name: str, provider: StateDictProvider) -> None:
        model = must_model(spec.target_ref)
        sd = provider.get_state_dict(model)
        slice_spec = (
            parse_slice(spec.target_ref.slice_spec)
            if spec.target_ref.slice_spec is not None
            else None
        )
        view = select_tensor(sd[name], slice_spec)
        view.clamp_(min=spec.min_value, max=spec.max_value)






register_transform(ClampInPlaceTransform())
