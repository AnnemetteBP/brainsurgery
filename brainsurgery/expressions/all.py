from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..expression import AssertExpr, AssertTransformError, collect_expr_models, compile_assert_expr, register_assert_expr
from ..transform_types import StateDictProvider


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
