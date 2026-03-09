from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..expression import AssertExpr, AssertTransformError, compile_assert_expr, register_assert_expr
from ..transform import StateDictProvider


@dataclass(frozen=True)
class NotExpr:
    expr: AssertExpr

    def evaluate(self, provider: StateDictProvider) -> None:
        try:
            self.expr.evaluate(provider)
        except AssertTransformError:
            return

        raise AssertTransformError(f"not failed: inner assertion succeeded: {self.expr!r}")

    def collect_models(self) -> set[str]:
        return self.expr.collect_models()


@register_assert_expr(
    "not",
    payload_kind="assert-expr",
    description="Succeeds if the inner assertion fails.",
)
def compile_not_expr(payload: Any, default_model: str | None) -> AssertExpr:
    return NotExpr(expr=compile_assert_expr(payload, default_model))


def _unit_test_not_compile_rejects_invalid_expr() -> None:
    try:
        compile_not_expr({"unknown": {}}, default_model="model")
    except AssertTransformError as exc:
        assert "unknown assert op" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected unknown assert op error")


def _unit_test_not_evaluate_succeeds_when_inner_fails() -> None:
    class _FailExpr:
        def evaluate(self, provider) -> None:
            del provider
            raise AssertTransformError("fail")

        def collect_models(self) -> set[str]:
            return {"model"}

    NotExpr(expr=_FailExpr()).evaluate(provider=None)  # type: ignore[arg-type]


def _unit_test_not_evaluate_fails_when_inner_succeeds() -> None:
    class _PassExpr:
        def evaluate(self, provider) -> None:
            del provider

        def collect_models(self) -> set[str]:
            return {"model"}

    try:
        NotExpr(expr=_PassExpr()).evaluate(provider=None)  # type: ignore[arg-type]
    except AssertTransformError as exc:
        assert "inner assertion succeeded" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected not failure")


__unit_tests__ = [
    _unit_test_not_compile_rejects_invalid_expr,
    _unit_test_not_evaluate_succeeds_when_inner_fails,
    _unit_test_not_evaluate_fails_when_inner_succeeds,
]
