from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from ..expression import (
    collect_ref_models,
    compile_shape,
    compile_tensor_ref_expr,
    format_ref,
    register_assert_expr,
    resolve_tensors,
    require_mapping_assert_payload,
)
from ..expression import AssertTransformError
from ..refs import TensorRef
from ..transform import StateDictProvider


@dataclass(frozen=True)
class ShapeExpr:
    ref: TensorRef
    is_value: tuple[int, ...]

    def evaluate(self, provider: StateDictProvider) -> None:
        for ref, tensor in resolve_tensors(self.ref, provider, op_name="shape.of"):
            if tuple(tensor.shape) != self.is_value:
                raise AssertTransformError(
                    f"shape failed: {format_ref(ref)} has shape {tuple(tensor.shape)}, expected {self.is_value}"
                )

    def collect_models(self) -> set[str]:
        return collect_ref_models(self.ref)


@register_assert_expr(
    "shape",
    payload_kind="mapping",
    allowed_keys={"of", "is"},
    required_keys={"of", "is"},
    description="Succeeds if the tensor has the given shape.",
)
def compile_shape_expr(payload: Any, default_model: str | None) -> ShapeExpr:
    payload = require_mapping_assert_payload(
        payload,
        op_name="shape",
        allowed_keys={"of", "is"},
        required_keys={"of", "is"},
    )
    ref = compile_tensor_ref_expr(payload["of"], default_model, "shape.of")
    return ShapeExpr(ref=ref, is_value=compile_shape(payload["is"]))


def _unit_test_shape_compile_rejects_non_integer_shape() -> None:
    try:
        compile_shape_expr({"of": "x", "is": [1, "2"]}, default_model="model")
    except AssertTransformError as exc:
        assert "shape.is must be a list of integers" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected shape integer validation error")


def _unit_test_shape_evaluate_success() -> None:
    class _Provider:
        def get_state_dict(self, model: str):
            assert model == "model"
            return {"x": torch.ones((2, 3))}

    ShapeExpr(ref=TensorRef(model="model", expr="x"), is_value=(2, 3)).evaluate(_Provider())


def _unit_test_shape_evaluate_mismatch() -> None:
    class _Provider:
        def get_state_dict(self, model: str):
            assert model == "model"
            return {"x": torch.ones((2, 3))}

    try:
        ShapeExpr(ref=TensorRef(model="model", expr="x"), is_value=(3, 2)).evaluate(_Provider())
    except AssertTransformError as exc:
        assert "has shape (2, 3), expected (3, 2)" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected shape mismatch")


def _unit_test_shape_evaluate_pattern_checks_all_matches() -> None:
    class _Provider:
        def get_state_dict(self, model: str):
            assert model == "model"
            return {"x0": torch.ones((2, 3)), "x1": torch.ones((2, 4))}

    try:
        ShapeExpr(ref=TensorRef(model="model", expr="x.*"), is_value=(2, 3)).evaluate(_Provider())
    except AssertTransformError as exc:
        assert "model::x1" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected shape mismatch on one matched tensor")


__unit_tests__ = [
    _unit_test_shape_compile_rejects_non_integer_shape,
    _unit_test_shape_evaluate_success,
    _unit_test_shape_evaluate_mismatch,
    _unit_test_shape_evaluate_pattern_checks_all_matches,
]
