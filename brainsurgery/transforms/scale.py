from __future__ import annotations

from dataclasses import dataclass

import torch

from .unary import UnarySpec, UnaryTransform
from ..transform import (
    StateDictProvider,
    TensorRef,
    TransformError,
    must_model,
    parse_slice,
    register_transform,
    require_numeric,
    select_tensor,
)


class ScaleTransformError(TransformError):
    pass


@dataclass(frozen=True)
class ScaleSpec(UnarySpec):
    factor: float


class ScaleTransform(UnaryTransform[ScaleSpec]):
    name = "scale"
    error_type = ScaleTransformError
    spec_type = ScaleSpec
    allowed_keys = {"target", "by"}
    required_keys = {"target", "by"}
    slice_policy = "allow"
    progress_desc = "Applying scale transforms"
    help_text = (
        "Scales tensors in-place by a numeric factor.\n"
        "\n"
        "Targets may be specified by name or pattern and may include slicing "
        "(written after '::'). The selected tensor (or slice) is multiplied by 'by'.\n"
        "\n"
        "Examples:\n"
        "  scale: { target: ln_f.weight, by: 0.5 }\n"
        "  scale: { target: '.*bias', by: -1 }\n"
        "  scale: { target: 'h.0.attn.c_attn.weight::[:, :10]', by: 2.0 }"
    )

    def build_spec(self, target_ref: TensorRef, payload: dict) -> ScaleSpec:
        factor = require_numeric(payload, op_name=self.name, key="by")
        return ScaleSpec(target_ref=target_ref, factor=factor)

    def apply_to_target(self, spec: ScaleSpec, name: str, provider: StateDictProvider) -> None:
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


def _unit_test_scale_compile_rejects_non_numeric_factor() -> None:
    try:
        ScaleTransform().compile({"target": "x", "by": "nan?!"}, default_model="model")
    except TransformError as exc:
        assert "scale.by must be numeric" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected scale numeric validation error")


def _unit_test_scale_compile_accepts_numeric_string_factor() -> None:
    spec = ScaleTransform().compile({"target": "x", "by": "2.5"}, default_model="model")
    assert spec.factor == 2.5


def _unit_test_scale_apply_with_slice() -> None:
    class _Provider:
        def __init__(self) -> None:
            self._state_dict = {"x": torch.tensor([1.0, 2.0, 3.0, 4.0])}

        def get_state_dict(self, model: str):
            assert model == "model"
            return self._state_dict

    provider = _Provider()
    spec = ScaleSpec(target_ref=TensorRef(model="model", expr="x", slice_spec="[1:3]"), factor=10.0)
    ScaleTransform().apply_to_target(spec, "x", provider)
    assert provider._state_dict["x"].tolist() == [1.0, 20.0, 30.0, 4.0]


__unit_tests__ = [
    _unit_test_scale_compile_rejects_non_numeric_factor,
    _unit_test_scale_compile_accepts_numeric_string_factor,
    _unit_test_scale_apply_with_slice,
]


register_transform(ScaleTransform())
