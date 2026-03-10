from __future__ import annotations

from dataclasses import dataclass

import torch

from .unary import UnarySpec, UnaryTransform
from ..refs import TensorRef, must_model, parse_slice, select_tensor
from ..transform import (
    StateDictProvider,
    TransformError,
    register_transform,
    require_numeric,
)


class ScaleInPlaceTransformError(TransformError):
    pass


@dataclass(frozen=True)
class ScaleInPlaceSpec(UnarySpec):
    factor: float


class ScaleInPlaceTransform(UnaryTransform[ScaleInPlaceSpec]):
    name = "scale_"
    error_type = ScaleInPlaceTransformError
    spec_type = ScaleInPlaceSpec
    allowed_keys = {"target", "by"}
    required_keys = {"target", "by"}
    slice_policy = "allow"
    progress_desc = "Applying scale_ transforms"
    help_text = (
        "Scales tensors in-place by a numeric factor.\n"
        "\n"
        "Targets may be specified by name or pattern and may include slicing "
        "(written after '::'). The selected tensor (or slice) is multiplied by 'by'.\n"
        "\n"
        "Examples:\n"
        "  scale_: { target: ln_f.weight, by: 0.5 }\n"
        "  scale_: { target: '.*bias', by: -1 }\n"
        "  scale_: { target: 'h.0.attn.c_attn.weight::[:, :10]', by: 2.0 }"
    )

    def build_spec(self, target_ref: TensorRef, payload: dict) -> ScaleInPlaceSpec:
        factor = require_numeric(payload, op_name=self.name, key="by")
        return ScaleInPlaceSpec(target_ref=target_ref, factor=factor)

    def apply_to_target(self, spec: ScaleInPlaceSpec, name: str, provider: StateDictProvider) -> None:
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










register_transform(ScaleInPlaceTransform())
