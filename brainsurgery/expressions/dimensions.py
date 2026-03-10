from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from ..expression import (
    AssertTransformError,
    collect_ref_models,
    compile_tensor_ref_expr,
    format_ref,
    register_assert_expr,
    resolve_tensors,
    require_mapping_assert_payload,
)
from ..refs import TensorRef
from ..transform_types import StateDictProvider


@dataclass(frozen=True)
class DimensionsExpr:
    ref: TensorRef
    is_value: int

    def evaluate(self, provider: StateDictProvider) -> None:
        for ref, tensor in resolve_tensors(self.ref, provider, op_name="dimensions.of"):
            if len(tensor.shape) != self.is_value:
                raise AssertTransformError(
                    f"dimensions failed: {format_ref(ref)} has {len(tensor.shape)} dims, expected {self.is_value}"
                )

    def collect_models(self) -> set[str]:
        return collect_ref_models(self.ref)


@register_assert_expr(
    "dimensions",
    payload_kind="mapping",
    allowed_keys={"of", "is"},
    required_keys={"of", "is"},
    description="Succeeds if the tensor has the given number of dimensions.",
)
def compile_dimensions_expr(payload: Any, default_model: str | None) -> DimensionsExpr:
    payload = require_mapping_assert_payload(
        payload,
        op_name="dimensions",
        allowed_keys={"of", "is"},
        required_keys={"of", "is"},
    )
    ref = compile_tensor_ref_expr(payload["of"], default_model, "dimensions.of")
    is_value = payload["is"]
    if not isinstance(is_value, int):
        raise AssertTransformError("dimensions.is must be an integer")
    return DimensionsExpr(ref=ref, is_value=is_value)
