from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from ..core import TransformError, collect_ref_models, compile_tensor_ref_expr, format_ref, register_assert_expr, resolve_tensors, require_mapping_assert_payload
from ..core import parse_torch_dtype
from ..core import TensorRef
from ..core import StateDictProvider


@dataclass(frozen=True)
class DtypeExpr:
    ref: TensorRef
    is_value: torch.dtype

    def evaluate(self, provider: StateDictProvider) -> None:
        for ref, tensor in resolve_tensors(self.ref, provider, op_name="dtype.of"):
            if tensor.dtype != self.is_value:
                raise TransformError(
                    f"dtype failed: {format_ref(ref)} has dtype {tensor.dtype}, expected {self.is_value}"
                )

    def collect_models(self) -> set[str]:
        return collect_ref_models(self.ref)


@register_assert_expr(
    "dtype",
    payload_kind="mapping",
    allowed_keys={"of", "is"},
    required_keys={"of", "is"},
    description="Succeeds if the tensor has the given dtype.",
)
def compile_dtype_expr(payload: Any, default_model: str | None) -> DtypeExpr:
    payload = require_mapping_assert_payload(
        payload,
        op_name="dtype",
        allowed_keys={"of", "is"},
        required_keys={"of", "is"},
    )
    ref = compile_tensor_ref_expr(payload["of"], default_model, "dtype.of")

    raw_dtype = payload["is"]
    if not isinstance(raw_dtype, str) or not raw_dtype:
        raise TransformError("dtype.is must be a non-empty string")

    return DtypeExpr(
        ref=ref,
        is_value=parse_torch_dtype(
            raw_dtype,
            error_type=TransformError,
            op_name="dtype",
            field_name="is",
        ),
    )
