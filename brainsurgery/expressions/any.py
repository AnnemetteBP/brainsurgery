from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..expression import AssertExpr, AssertTransformError, collect_expr_models, compile_assert_expr, register_assert_expr
from ..transform import StateDictProvider


@dataclass(frozen=True)
class AnyExpr:
    exprs: list[AssertExpr]

    def evaluate(self, provider: StateDictProvider) -> None:
        errors: list[str] = []
        for expr in self.exprs:
            try:
                expr.evaluate(provider)
                return
            except AssertTransformError as exc:
                errors.append(str(exc))
        raise AssertTransformError("any failed: all alternatives failed:\n- " + "\n- ".join(errors))

    def collect_models(self) -> set[str]:
        return collect_expr_models(self.exprs)


@register_assert_expr(
    "any",
    payload_kind="list[assert-expr]",
    description="Succeeds if any inner assertion succeeds.",
)
def compile_any_expr(payload: Any, default_model: str | None) -> AnyExpr:
    if not isinstance(payload, list) or not payload:
        raise AssertTransformError("any must be a non-empty list")
    return AnyExpr(exprs=[compile_assert_expr(item, default_model) for item in payload])


def _unit_test_any_compile_rejects_empty_list() -> None:
    try:
        compile_any_expr([], default_model="model")
    except AssertTransformError as exc:
        assert "any must be a non-empty list" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected non-empty list validation error")


def _unit_test_any_evaluate_succeeds_when_one_branch_passes() -> None:
    class _FailExpr:
        def evaluate(self, provider) -> None:
            del provider
            raise AssertTransformError("fail")

        def collect_models(self) -> set[str]:
            return {"model"}

    class _PassExpr:
        def evaluate(self, provider) -> None:
            del provider

        def collect_models(self) -> set[str]:
            return {"model"}

    AnyExpr(exprs=[_FailExpr(), _PassExpr()]).evaluate(provider=None)  # type: ignore[arg-type]


def _unit_test_any_evaluate_reports_all_failures() -> None:
    class _FailExpr:
        def __init__(self, msg: str) -> None:
            self.msg = msg

        def evaluate(self, provider) -> None:
            del provider
            raise AssertTransformError(self.msg)

        def collect_models(self) -> set[str]:
            return {"model"}

    try:
        AnyExpr(exprs=[_FailExpr("a"), _FailExpr("b")]).evaluate(provider=None)  # type: ignore[arg-type]
    except AssertTransformError as exc:
        message = str(exc)
        assert "all alternatives failed" in message
        assert "- a" in message
        assert "- b" in message
    else:  # pragma: no cover
        raise AssertionError("expected aggregated failure")


__unit_tests__ = [
    _unit_test_any_compile_rejects_empty_list,
    _unit_test_any_evaluate_succeeds_when_one_branch_passes,
    _unit_test_any_evaluate_reports_all_failures,
]
