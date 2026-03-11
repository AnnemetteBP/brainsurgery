from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from ..expression import AssertExpr, AssertTransformError, collect_ref_models, compile_tensor_ref_expr, format_ref, register_assert_expr, require_mapping_assert_payload, resolve_tensors
from ..core import TensorRef
from ..core import StateDictProvider


@dataclass(frozen=True)
class IsZeroExpr:
    ref: TensorRef
    eps: float | None = None

    def evaluate(self, provider: StateDictProvider) -> None:
        for ref, tensor in resolve_tensors(self.ref, provider, op_name="iszero"):
            if self.eps is None:
                is_zero = bool(torch.all(tensor == 0).item())
            else:
                if tensor.is_complex():
                    diff = torch.abs(tensor.to(torch.complex128))
                else:
                    diff = torch.abs(tensor.to(torch.float64))
                is_zero = bool(torch.all(diff <= self.eps).item())

            if not is_zero:
                if self.eps is None:
                    raise AssertTransformError(f"iszero failed: {format_ref(ref)} is not all zeros")
                raise AssertTransformError(
                    f"iszero failed: {format_ref(ref)} is not all zeros within eps={self.eps}"
                )

    def collect_models(self) -> set[str]:
        return collect_ref_models(self.ref)


@register_assert_expr(
    "iszero",
    payload_kind="tensor-ref|mapping",
    allowed_keys={"of", "eps"},
    required_keys={"of"},
    description=(
        "Succeeds if selected tensor(s) are all zeros, "
        "or within optional absolute tolerance 'eps'."
    ),
)
def compile_iszero_expr(payload: Any, default_model: str | None) -> AssertExpr:
    if isinstance(payload, dict):
        payload = require_mapping_assert_payload(
            payload,
            op_name="iszero",
            allowed_keys={"of", "eps"},
            required_keys={"of"},
        )
        raw_ref = payload["of"]
        raw_eps = payload.get("eps")
    else:
        raw_ref = payload
        raw_eps = None

    ref = compile_tensor_ref_expr(raw_ref, default_model, "iszero")

    if raw_eps is None:
        eps = None
    else:
        if isinstance(raw_eps, bool) or not isinstance(raw_eps, (int, float)):
            raise AssertTransformError("iszero.eps must be a non-negative number")
        eps = float(raw_eps)
        if eps < 0:
            raise AssertTransformError("iszero.eps must be a non-negative number")

    return IsZeroExpr(ref=ref, eps=eps)
