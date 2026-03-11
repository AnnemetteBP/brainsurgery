from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from ..core import (
    AssertTransformError,
    collect_ref_models,
    compile_tensor_ref_expr,
    format_ref,
    register_assert_expr,
    resolve_tensors,
    require_mapping_assert_payload,
)
from ..core import TensorRef
from ..core import StateDictProvider
from .scalar_compare import ScalarComparison, parse_scalar_comparison


@dataclass(frozen=True)
class DimensionsExpr:
    ref: TensorRef
    comparison: ScalarComparison

    def evaluate(self, provider: StateDictProvider) -> None:
        for ref, tensor in resolve_tensors(self.ref, provider, op_name="dimensions.of"):
            actual = len(tensor.shape)
            if not self.comparison.matches(actual):
                raise AssertTransformError(
                    f"dimensions failed: {format_ref(ref)} has {actual} dims, expected {self.comparison.describe()}"
                )

    def collect_models(self) -> set[str]:
        return collect_ref_models(self.ref)


@register_assert_expr(
    "dimensions",
    payload_kind="mapping",
    allowed_keys={"of", "is", "ge", "gt", "le", "lt"},
    required_keys={"of"},
    description="Succeeds if the tensor rank satisfies the requested comparison(s).",
)
def compile_dimensions_expr(payload: Any, default_model: str | None) -> DimensionsExpr:
    payload = require_mapping_assert_payload(
        payload,
        op_name="dimensions",
        allowed_keys={"of", "is", "ge", "gt", "le", "lt"},
        required_keys={"of"},
    )
    ref = compile_tensor_ref_expr(payload["of"], default_model, "dimensions.of")
    comparison = parse_scalar_comparison(payload, op_name="dimensions")
    return DimensionsExpr(ref=ref, comparison=comparison)
