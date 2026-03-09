from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..expressions import AssertExpr, AssertTransformError, compile_assert_expr
from ..transform import (
    StateDictProvider,
    TypedTransform,
    TransformResult,
    ensure_mapping_payload,
    register_transform,
)


@dataclass(frozen=True)
class AssertSpec:
    expr: AssertExpr

    def collect_models(self) -> set[str]:
        return self.expr.collect_models()


class AssertTransform(TypedTransform[AssertSpec]):
    name = "assert"
    error_type = AssertTransformError
    spec_type = AssertSpec
    help_text = (
        "Checks conditions on tensors using a single assert expression. "
        "The operation fails if the assertion does not hold and does not modify tensors.\n"
        "\n"
        "Expressions can match tensors by name or pattern and may include slicing "
        "(written after '::') and nesting.\n"
        "\n"
        "Examples:\n"
        "  assert: { equal: { left: a.weight, right: b.weight } }\n"
        "  assert: { not: { equal: { left: a.weight, right: b.weight } } }\n"
        "  assert: { dtype: { of: ln_f.weight, is: float32 } }\n"
        "  assert: { shape: { of: 'ln_f.weight::[:8]', is: [8] } }\n"
        "  assert: { all: [ { exists: '.*weight' }, { dimensions: { of: ln_f.weight, is: 1 } } ] }"
    )

    def compile(self, payload: Any, default_model: str | None) -> AssertSpec:
        payload = ensure_mapping_payload(payload, self.name)
        expr = compile_assert_expr(payload, default_model)
        return AssertSpec(expr=expr)

    def apply(self, spec: object, provider: StateDictProvider) -> TransformResult:
        typed = self.require_spec(spec)
        typed.expr.evaluate(provider)
        return TransformResult(name=self.name, count=1)

    def infer_output_model(self, spec: object) -> str:
        models = self.require_spec(spec).collect_models()
        if len(models) != 1:
            raise self.error_type(
                f"{self.name} must refer to exactly one model to infer output, got {sorted(models)}"
            )

        return next(iter(models))


register_transform(AssertTransform())
