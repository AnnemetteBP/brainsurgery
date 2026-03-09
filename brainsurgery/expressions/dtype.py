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
    require_mapping_assert_payload,
    resolve_single_tensor,
)
from ..dtypes import parse_torch_dtype
from ..transform import StateDictProvider, TensorRef


@dataclass(frozen=True)
class DtypeExpr:
    ref: TensorRef
    is_value: torch.dtype

    def evaluate(self, provider: StateDictProvider) -> None:
        tensor = resolve_single_tensor(self.ref, provider, op_name="dtype.of")
        if tensor.dtype != self.is_value:
            raise AssertTransformError(
                f"dtype failed: {format_ref(self.ref)} has dtype {tensor.dtype}, expected {self.is_value}"
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
        raise AssertTransformError("dtype.is must be a non-empty string")

    return DtypeExpr(
        ref=ref,
        is_value=parse_torch_dtype(
            raw_dtype,
            error_type=AssertTransformError,
            op_name="dtype",
            field_name="is",
        ),
    )


def _unit_test_dtype_compile_rejects_empty_is() -> None:
    try:
        compile_dtype_expr({"of": "x", "is": ""}, default_model="model")
    except AssertTransformError as exc:
        assert "dtype.is must be a non-empty string" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected dtype.is non-empty string validation error")


def _unit_test_dtype_evaluate_success() -> None:
    class _Provider:
        def get_state_dict(self, model: str):
            assert model == "model"
            return {"x": torch.ones((2,), dtype=torch.float32)}

    expr = DtypeExpr(ref=TensorRef(model="model", expr="x"), is_value=torch.float32)
    expr.evaluate(_Provider())


def _unit_test_dtype_evaluate_mismatch() -> None:
    class _Provider:
        def get_state_dict(self, model: str):
            assert model == "model"
            return {"x": torch.ones((2,), dtype=torch.float16)}

    expr = DtypeExpr(ref=TensorRef(model="model", expr="x"), is_value=torch.float32)
    try:
        expr.evaluate(_Provider())
    except AssertTransformError as exc:
        assert "has dtype torch.float16, expected torch.float32" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected dtype mismatch")


__unit_tests__ = [
    _unit_test_dtype_compile_rejects_empty_is,
    _unit_test_dtype_evaluate_success,
    _unit_test_dtype_evaluate_mismatch,
]
