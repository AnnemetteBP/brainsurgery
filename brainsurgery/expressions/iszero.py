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
    require_mapping_assert_payload,
    resolve_tensors,
)
from ..transform import StateDictProvider, TensorRef


@dataclass(frozen=True)
class IsZeroExpr:
    ref: TensorRef
    eps: float | None = None

    def evaluate(self, provider: StateDictProvider) -> None:
        for ref, tensor in resolve_tensors(self.ref, provider, op_name="iszero"):
            if self.eps is None:
                is_zero = bool(torch.all(tensor == 0).item())
            else:
                if tensor.is_complex():
                    diff = torch.abs(tensor.to(torch.complex128))
                else:
                    diff = torch.abs(tensor.to(torch.float64))
                is_zero = bool(torch.all(diff <= self.eps).item())

            if not is_zero:
                if self.eps is None:
                    raise AssertTransformError(f"iszero failed: {format_ref(ref)} is not all zeros")
                raise AssertTransformError(
                    f"iszero failed: {format_ref(ref)} is not all zeros within eps={self.eps}"
                )

    def collect_models(self) -> set[str]:
        return collect_ref_models(self.ref)


@register_assert_expr(
    "iszero",
    payload_kind="tensor-ref|mapping",
    allowed_keys={"of", "eps"},
    required_keys={"of"},
    description=(
        "Succeeds if selected tensor(s) are all zeros, "
        "or within optional absolute tolerance 'eps'."
    ),
)
def compile_iszero_expr(payload: Any, default_model: str | None) -> AssertExpr:
    if isinstance(payload, dict):
        payload = require_mapping_assert_payload(
            payload,
            op_name="iszero",
            allowed_keys={"of", "eps"},
            required_keys={"of"},
        )
        raw_ref = payload["of"]
        raw_eps = payload.get("eps")
    else:
        raw_ref = payload
        raw_eps = None

    ref = compile_tensor_ref_expr(raw_ref, default_model, "iszero")

    if raw_eps is None:
        eps = None
    else:
        if isinstance(raw_eps, bool) or not isinstance(raw_eps, (int, float)):
            raise AssertTransformError("iszero.eps must be a non-negative number")
        eps = float(raw_eps)
        if eps < 0:
            raise AssertTransformError("iszero.eps must be a non-negative number")

    return IsZeroExpr(ref=ref, eps=eps)


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


def _unit_test_iszero_evaluate_pattern_checks_all_matches() -> None:
    class _Provider:
        def get_state_dict(self, model: str):
            assert model == "model"
            return {
                "x0": torch.zeros((2, 2)),
                "x1": torch.tensor([0.0, 1.0]),
            }

    try:
        IsZeroExpr(ref=TensorRef(model="model", expr="x.*")).evaluate(_Provider())
    except AssertTransformError as exc:
        assert "model::x1" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected iszero failure on one matched tensor")


def _unit_test_iszero_compile_rejects_negative_eps() -> None:
    try:
        compile_iszero_expr({"of": "x", "eps": -1e-3}, default_model="model")
    except AssertTransformError as exc:
        assert "iszero.eps must be a non-negative number" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected iszero.eps validation error")


def _unit_test_iszero_within_eps_succeeds() -> None:
    class _Provider:
        def get_state_dict(self, model: str):
            assert model == "model"
            return {"x": torch.tensor([0.0, 5e-4])}

    IsZeroExpr(ref=TensorRef(model="model", expr="x"), eps=1e-3).evaluate(_Provider())


def _unit_test_iszero_outside_eps_fails() -> None:
    class _Provider:
        def get_state_dict(self, model: str):
            assert model == "model"
            return {"x": torch.tensor([0.0, 5e-4])}

    try:
        IsZeroExpr(ref=TensorRef(model="model", expr="x"), eps=1e-4).evaluate(_Provider())
    except AssertTransformError as exc:
        assert "within eps" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected iszero failure outside eps")


__unit_tests__ = [
    _unit_test_iszero_compile_rejects_invalid_ref_type,
    _unit_test_iszero_evaluate_success,
    _unit_test_iszero_evaluate_failure,
    _unit_test_iszero_evaluate_pattern_checks_all_matches,
    _unit_test_iszero_compile_rejects_negative_eps,
    _unit_test_iszero_within_eps_succeeds,
    _unit_test_iszero_outside_eps_fails,
]
