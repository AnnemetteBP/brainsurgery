from __future__ import annotations

from dataclasses import dataclass

from ..transform import BaseTransform, StateDictProvider, TransformResult, ensure_mapping_payload, register_transform
from ..expressions import AssertExpr, AssertTransformError, compile_assert_expr


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
        if not isinstance(spec, AssertSpec):
            raise AssertTransformError(f"assert received wrong spec type: {type(spec).__name__}")
        spec.expr.evaluate(provider)
        return TransformResult(name=self.name, count=1)

    def infer_output_model(self, spec: object) -> str:
        if not isinstance(spec, AssertSpec):
            raise AssertTransformError(f"assert received wrong spec type: {type(spec).__name__}")

        models = sorted(spec.expr.collect_models())
        if not models:
            raise AssertTransformError("assert does not reference any model")
        if len(models) != 1:
            raise AssertTransformError(
                f"assert references multiple models; cannot infer unique output model: {models}"
            )
        return models[0]


register_transform(AssertTransform())

