from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..expression import (
    AssertExpr,
    AssertTransformError,
    collect_ref_models,
    compile_tensor_ref_expr,
    format_ref,
    register_assert_expr,
    resolve_matches,
)
from ..refs import TensorRef
from ..transform import StateDictProvider


@dataclass(frozen=True)
class ExistsExpr:
    ref: TensorRef

    def evaluate(self, provider: StateDictProvider) -> None:
        matches = resolve_matches(self.ref, provider, op_name="exists")
        if not matches:
            raise AssertTransformError(f"exists failed: {format_ref(self.ref)} matched zero tensors")

    def collect_models(self) -> set[str]:
        return collect_ref_models(self.ref)


@register_assert_expr(
    "exists",
    payload_kind="tensor-ref",
    description="Succeeds if the reference matches at least one tensor.",
)
def compile_exists_expr(payload: Any, default_model: str | None) -> AssertExpr:
    return ExistsExpr(ref=compile_tensor_ref_expr(payload, default_model, "exists"))


def _unit_test_exists_compile_rejects_empty_ref() -> None:
    try:
        compile_exists_expr("", default_model="model")
    except AssertTransformError as exc:
        assert "exists must be a non-empty string reference" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected non-empty reference validation error")


def _unit_test_exists_evaluate_success() -> None:
    class _Provider:
        def get_state_dict(self, model: str):
            assert model == "model"
            return {"abc": object()}

    ExistsExpr(ref=TensorRef(model="model", expr="a.*")).evaluate(_Provider())


def _unit_test_exists_evaluate_failure() -> None:
    class _Provider:
        def get_state_dict(self, model: str):
            assert model == "model"
            return {"abc": object()}

    try:
        ExistsExpr(ref=TensorRef(model="model", expr="z.*")).evaluate(_Provider())
    except AssertTransformError as exc:
        assert "matched zero tensors" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected exists failure")


__unit_tests__ = [
    _unit_test_exists_compile_rejects_empty_ref,
    _unit_test_exists_evaluate_success,
    _unit_test_exists_evaluate_failure,
]
