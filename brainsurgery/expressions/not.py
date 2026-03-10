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
