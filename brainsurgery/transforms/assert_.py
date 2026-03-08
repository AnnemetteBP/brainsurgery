from __future__ import annotations

from dataclasses import dataclass

from ..expressions import AssertExpr, AssertTransformError, compile_assert_expr
from ..transform import (
    BaseTransform,
    StateDictProvider,
    TransformResult,
    ensure_mapping_payload,
    register_transform,
)


@dataclass(frozen=True)
class AssertSpec:
    expr: AssertExpr


class AssertTransform(BaseTransform):
    name = "assert"

    def compile(self, payload: dict, default_model: str | None) -> AssertSpec:
        payload = ensure_mapping_payload(payload, self.name)
        if len(payload) != 1:
            raise AssertTransformError("assert payload must contain exactly one top-level assertion")
        return AssertSpec(expr=compile_assert_expr(payload, default_model))

    def apply(self, spec: object, provider: StateDictProvider) -> TransformResult:
        typed = self.require_spec(spec)
        typed.expr.evaluate(provider)
        return TransformResult(name=self.name, count=1)

    def infer_output_model(self, spec: object) -> str:
        typed = self.require_spec(spec)

        models = sorted(typed.expr.collect_models())
        if not models:
            raise AssertTransformError("assert does not reference any model")
        if len(models) != 1:
            raise AssertTransformError(
                f"assert references multiple models; cannot infer unique output model: {models}"
            )
        return models[0]

    def require_spec(self, spec: object) -> AssertSpec:
        if not isinstance(spec, AssertSpec):
            raise AssertTransformError(f"assert received wrong spec type: {type(spec).__name__}")
        return spec


register_transform(AssertTransform())
