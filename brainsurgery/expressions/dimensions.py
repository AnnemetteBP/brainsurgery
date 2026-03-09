from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from ..expression import (
    AssertTransformError,
    collect_ref_models,
    compile_tensor_ref_expr,
    format_ref,
    register_assert_expr,
    require_mapping_assert_payload,
    resolve_single_tensor,
)
from ..transform import StateDictProvider, TensorRef


@dataclass(frozen=True)
class DimensionsExpr:
    ref: TensorRef
    is_value: int

    def evaluate(self, provider: StateDictProvider) -> None:
        tensor = resolve_single_tensor(self.ref, provider, op_name="dimensions.of")
        if len(tensor.shape) != self.is_value:
            raise AssertTransformError(
                f"dimensions failed: {format_ref(self.ref)} has {len(tensor.shape)} dims, expected {self.is_value}"
            )

    def collect_models(self) -> set[str]:
        return collect_ref_models(self.ref)


@register_assert_expr(
    "dimensions",
    payload_kind="mapping",
    allowed_keys={"of", "is"},
    required_keys={"of", "is"},
    description="Succeeds if the tensor has the given number of dimensions.",
)
def compile_dimensions_expr(payload: Any, default_model: str | None) -> DimensionsExpr:
    payload = require_mapping_assert_payload(
        payload,
        op_name="dimensions",
        allowed_keys={"of", "is"},
        required_keys={"of", "is"},
    )
    ref = compile_tensor_ref_expr(payload["of"], default_model, "dimensions.of")
    is_value = payload["is"]
    if not isinstance(is_value, int):
        raise AssertTransformError("dimensions.is must be an integer")
    return DimensionsExpr(ref=ref, is_value=is_value)


def _unit_test_dimensions_compile_rejects_non_int_is() -> None:
    try:
        compile_dimensions_expr({"of": "x", "is": "1"}, default_model="model")
    except AssertTransformError as exc:
        assert "dimensions.is must be an integer" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected dimensions.is integer validation error")


def _unit_test_dimensions_evaluate_success() -> None:
    class _Provider:
        def get_state_dict(self, model: str):
            assert model == "model"
            return {"x": torch.ones((2, 3, 4))}

    expr = DimensionsExpr(ref=TensorRef(model="model", expr="x"), is_value=3)
    expr.evaluate(_Provider())


def _unit_test_dimensions_evaluate_mismatch() -> None:
    class _Provider:
        def get_state_dict(self, model: str):
            assert model == "model"
            return {"x": torch.ones((2, 3))}

    expr = DimensionsExpr(ref=TensorRef(model="model", expr="x"), is_value=3)
    try:
        expr.evaluate(_Provider())
    except AssertTransformError as exc:
        assert "has 2 dims, expected 3" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected dimension mismatch error")


__unit_tests__ = [
    _unit_test_dimensions_compile_rejects_non_int_is,
    _unit_test_dimensions_evaluate_success,
    _unit_test_dimensions_evaluate_mismatch,
]
