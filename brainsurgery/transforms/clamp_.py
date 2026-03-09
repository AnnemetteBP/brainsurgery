from __future__ import annotations

from dataclasses import dataclass

import torch

from .unary import UnarySpec, UnaryTransform
from .clamp import _parse_bounds
from ..transform import (
    StateDictProvider,
    TensorRef,
    TransformError,
    must_model,
    parse_slice,
    register_transform,
    select_tensor,
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


def _unit_test_clamp_in_place_apply_with_slice() -> None:
    class _Provider:
        def __init__(self) -> None:
            self._state_dict = {"x": torch.tensor([-3.0, 0.0, 4.0])}

        def get_state_dict(self, model: str):
            assert model == "m"
            return self._state_dict

    provider = _Provider()
    spec = ClampInPlaceSpec(
        target_ref=TensorRef(model="m", expr="x", slice_spec="[0:2]"),
        min_value=-1.0,
        max_value=1.0,
    )
    ClampInPlaceTransform().apply_to_target(spec, "x", provider)
    assert provider._state_dict["x"].tolist() == [-1.0, 0.0, 4.0]


__unit_tests__ = [
    _unit_test_clamp_in_place_apply_with_slice,
]


register_transform(ClampInPlaceTransform())
