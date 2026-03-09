from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from ..expression import (
    AssertExpr,
    AssertTransformError,
    collect_ref_models,
    compile_tensor_ref_expr,
    format_ref,
    register_assert_expr,
    resolve_single_tensor,
)
from ..transform import StateDictProvider, TensorRef


@dataclass(frozen=True)
class IsZeroExpr:
    ref: TensorRef

    def evaluate(self, provider: StateDictProvider) -> None:
        tensor = resolve_single_tensor(self.ref, provider, op_name="iszero")
        if not torch.all(tensor == 0):
            raise AssertTransformError(f"iszero failed: {format_ref(self.ref)} is not all zeros")

    def collect_models(self) -> set[str]:
        return collect_ref_models(self.ref)


@register_assert_expr(
    "iszero",
    payload_kind="tensor-ref",
    description="Succeeds if the selected tensor is all zeros.",
)
def compile_iszero_expr(payload: Any, default_model: str | None) -> AssertExpr:
    return IsZeroExpr(ref=compile_tensor_ref_expr(payload, default_model, "iszero"))


def _unit_test_iszero_compile_rejects_invalid_ref_type() -> None:
    try:
        compile_iszero_expr(123, default_model="model")
    except AssertTransformError as exc:
        assert "iszero must be a non-empty string reference" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected reference type validation error")


def _unit_test_iszero_evaluate_success() -> None:
    class _Provider:
        def get_state_dict(self, model: str):
            assert model == "model"
            return {"x": torch.zeros((2, 2))}

    IsZeroExpr(ref=TensorRef(model="model", expr="x")).evaluate(_Provider())


def _unit_test_iszero_evaluate_failure() -> None:
    class _Provider:
        def get_state_dict(self, model: str):
            assert model == "model"
            return {"x": torch.tensor([0.0, 1.0])}

    try:
        IsZeroExpr(ref=TensorRef(model="model", expr="x")).evaluate(_Provider())
    except AssertTransformError as exc:
        assert "not all zeros" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected iszero failure")


__unit_tests__ = [
    _unit_test_iszero_compile_rejects_invalid_ref_type,
    _unit_test_iszero_evaluate_success,
    _unit_test_iszero_evaluate_failure,
]
