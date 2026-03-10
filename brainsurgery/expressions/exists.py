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
