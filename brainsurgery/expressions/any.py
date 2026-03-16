from dataclasses import dataclass
from typing import Any

from ..core import (
    Expression,
    StateDictProvider,
    TransformError,
    collect_expr_models,
    compile_assert_expr,
    register_assert_expr,
)


@dataclass(frozen=True)
class AnyExpr:
    exprs: list[Expression]

    def evaluate(self, provider: StateDictProvider) -> None:
        errors: list[str] = []
        for expr in self.exprs:
            try:
                expr.evaluate(provider)
                return
            except TransformError as exc:
                errors.append(str(exc))
        raise TransformError("any failed: all alternatives failed:\n- " + "\n- ".join(errors))

    def collect_models(self) -> set[str]:
        return collect_expr_models(self.exprs)


@register_assert_expr(
    "any",
    payload_kind="list[assert-expr]",
    description="Succeeds if any inner assertion succeeds.",
)
def compile_any_expr(payload: Any, default_model: str | None) -> AnyExpr:
    if not isinstance(payload, list) or not payload:
        raise TransformError("any must be a non-empty list")
    return AnyExpr(exprs=[compile_assert_expr(item, default_model) for item in payload])
