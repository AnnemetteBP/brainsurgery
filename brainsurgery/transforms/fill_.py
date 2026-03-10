from __future__ import annotations

from dataclasses import dataclass

import torch

from .fill import FillConfig, build_filled_tensor_like, parse_fill_config
from .unary import UnarySpec, UnaryTransform
from ..refs import TensorRef, must_model, parse_slice, select_tensor
from ..transform import (
    StateDictProvider,
    TransformError,
    register_transform,
)


class FillInPlaceTransformError(TransformError):
    pass


@dataclass(frozen=True)
class FillInPlaceSpec(UnarySpec):
    config: FillConfig


class FillInPlaceTransform(UnaryTransform[FillInPlaceSpec]):
    name = "fill_"
    error_type = FillInPlaceTransformError
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
    slice_policy = "allow"
    progress_desc = "Applying fill_ transforms"
    help_text = (
        "Fills target tensors in-place.\n"
        "\n"
        "Modes:\n"
        "  - constant: uses scalar 'value'\n"
        "  - rand: random fill ('distribution': uniform|normal)\n"
        "  - tensor: uses concrete payload 'values' (broadcasted if needed)\n"
        "\n"
        "Targets may include slicing.\n"
        "\n"
        "Example:\n"
        "  fill_: { target: x, mode: constant, value: 0 }"
    )

    def build_spec(self, target_ref: TensorRef, payload: dict) -> FillInPlaceSpec:
        config = parse_fill_config(payload, op_name=self.name, error_type=FillInPlaceTransformError)
        return FillInPlaceSpec(target_ref=target_ref, config=config)

    def apply_to_target(self, spec: FillInPlaceSpec, name: str, provider: StateDictProvider) -> None:
        model = must_model(spec.target_ref)
        sd = provider.get_state_dict(model)
        slice_spec = (
            parse_slice(spec.target_ref.slice_spec)
            if spec.target_ref.slice_spec is not None
            else None
        )
        view = select_tensor(sd[name], slice_spec)
        filled = build_filled_tensor_like(view, spec.config, FillInPlaceTransformError)
        view.copy_(filled)






register_transform(FillInPlaceTransform())
