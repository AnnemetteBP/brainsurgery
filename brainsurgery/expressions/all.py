from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..expression import AssertExpr, AssertTransformError, collect_expr_models, compile_assert_expr, register_assert_expr
from ..transform import StateDictProvider


@dataclass(frozen=True)
class AllExpr:
    exprs: list[AssertExpr]

    def evaluate(self, provider: StateDictProvider) -> None:
        for expr in self.exprs:
            expr.evaluate(provider)

    def collect_models(self) -> set[str]:
        return collect_expr_models(self.exprs)


@register_assert_expr(
    "all",
    payload_kind="list[assert-expr]",
    description="Succeeds if all inner assertions succeed.",
)
def compile_all_expr(payload: Any, default_model: str | None) -> AllExpr:
    if not isinstance(payload, list) or not payload:
        raise AssertTransformError("all must be a non-empty list")
    return AllExpr(exprs=[compile_assert_expr(item, default_model) for item in payload])


def _unit_test_all_compile_rejects_empty_list() -> None:
    try:
        compile_all_expr([], default_model="model")
    except AssertTransformError as exc:
        assert "all must be a non-empty list" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected non-empty list validation error")


def _unit_test_all_evaluate_short_success_path() -> None:
    class _Expr:
        def __init__(self) -> None:
            self.called = False

        def evaluate(self, provider) -> None:
            del provider
            self.called = True

        def collect_models(self) -> set[str]:
            return {"model"}

    left = _Expr()
    right = _Expr()
    expr = AllExpr(exprs=[left, right])
    expr.evaluate(provider=None)  # type: ignore[arg-type]
    assert left.called and right.called


__unit_tests__ = [
    _unit_test_all_compile_rejects_empty_list,
    _unit_test_all_evaluate_short_success_path,
]
