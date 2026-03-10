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


def _unit_test_scale_in_place_compile_rejects_non_numeric_factor() -> None:
    try:
        ScaleInPlaceTransform().compile({"target": "x", "by": "nan?!"}, default_model="model")
    except TransformError as exc:
        assert "scale_.by must be numeric" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected scale_ numeric validation error")


def _unit_test_scale_in_place_compile_accepts_numeric_string_factor() -> None:
    spec = ScaleInPlaceTransform().compile({"target": "x", "by": "2.5"}, default_model="model")
    assert spec.factor == 2.5


def _unit_test_scale_in_place_apply_with_slice() -> None:
    class _Provider:
        def __init__(self) -> None:
            self._state_dict = {"x": torch.tensor([1.0, 2.0, 3.0, 4.0])}

        def get_state_dict(self, model: str):
            assert model == "model"
            return self._state_dict

    provider = _Provider()
    spec = ScaleInPlaceSpec(
        target_ref=TensorRef(model="model", expr="x", slice_spec="[1:3]"),
        factor=10.0,
    )
    ScaleInPlaceTransform().apply_to_target(spec, "x", provider)
    assert provider._state_dict["x"].tolist() == [1.0, 20.0, 30.0, 4.0]


__unit_tests__ = [
    _unit_test_scale_in_place_compile_rejects_non_numeric_factor,
    _unit_test_scale_in_place_compile_accepts_numeric_string_factor,
    _unit_test_scale_in_place_apply_with_slice,
]


register_transform(ScaleInPlaceTransform())
