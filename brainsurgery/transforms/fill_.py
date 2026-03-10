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


def _unit_test_fill_in_place_tensor_mode() -> None:
    class _Provider:
        def __init__(self) -> None:
            self._state_dict = {"x": torch.zeros((2,), dtype=torch.float32)}

        def get_state_dict(self, model: str):
            assert model == "m"
            return self._state_dict

    provider = _Provider()
    spec = FillInPlaceSpec(
        target_ref=TensorRef(model="m", expr="x"),
        config=FillConfig(
            mode="tensor",
            constant_value=None,
            values_payload=[5.0, 6.0],
            distribution="uniform",
            low=0.0,
            high=1.0,
            mean=0.0,
            std=1.0,
            seed=None,
        ),
    )
    FillInPlaceTransform().apply_to_target(spec, "x", provider)
    assert provider._state_dict["x"].tolist() == [5.0, 6.0]


__unit_tests__ = [
    _unit_test_fill_in_place_tensor_mode,
]


register_transform(FillInPlaceTransform())
