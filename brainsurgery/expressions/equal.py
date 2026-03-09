from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from ..expression import (
    AssertTransformError,
    compile_tensor_ref_expr,
    format_ref,
    register_assert_expr,
    require_mapping_assert_payload,
    resolve_single_tensor,
)
from ..transform import StateDictProvider, TensorRef, must_model


@dataclass(frozen=True)
class EqualExpr:
    left: TensorRef
    right: TensorRef

    def evaluate(self, provider: StateDictProvider) -> None:
        left = resolve_single_tensor(self.left, provider, op_name="equal.left")
        right = resolve_single_tensor(self.right, provider, op_name="equal.right")

        if left.shape != right.shape:
            raise AssertTransformError(
                f"equal failed: shape mismatch {tuple(left.shape)} != {tuple(right.shape)}"
            )
        if left.dtype != right.dtype:
            raise AssertTransformError(f"equal failed: dtype mismatch {left.dtype} != {right.dtype}")
        if left.device != right.device:
            raise AssertTransformError(f"equal failed: device mismatch {left.device} != {right.device}")
        if not torch.equal(left, right):
            raise AssertTransformError(f"equal failed: {format_ref(self.left)} != {format_ref(self.right)}")

    def collect_models(self) -> set[str]:
        return {must_model(self.left), must_model(self.right)}


@register_assert_expr(
    "equal",
    payload_kind="mapping",
    allowed_keys={"left", "right"},
    required_keys={"left", "right"},
    description="Succeeds if two tensors have the same shape, dtype, and values.",
)
def compile_equal_expr(payload: Any, default_model: str | None) -> EqualExpr:
    payload = require_mapping_assert_payload(
        payload,
        op_name="equal",
        allowed_keys={"left", "right"},
        required_keys={"left", "right"},
    )
    left = compile_tensor_ref_expr(payload["left"], default_model, "equal.left")
    right = compile_tensor_ref_expr(payload["right"], default_model, "equal.right")
    return EqualExpr(left=left, right=right)


def _unit_test_equal_dtype_compatibility() -> None:
    class _Provider:
        def __init__(self) -> None:
            self._state_dict = {
                "left": torch.ones((2, 2), dtype=torch.float32),
                "right": torch.ones((2, 2), dtype=torch.float16),
            }

        def get_state_dict(self, model: str):
            assert model == "model"
            return self._state_dict

    expr = EqualExpr(
        left=TensorRef(model="model", expr="left"),
        right=TensorRef(model="model", expr="right"),
    )
    try:
        expr.evaluate(_Provider())
    except AssertTransformError as exc:
        assert "dtype mismatch" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected dtype mismatch error")


def _unit_test_equal_shape_compatibility() -> None:
    class _Provider:
        def __init__(self) -> None:
            self._state_dict = {
                "left": torch.ones((2, 2)),
                "right": torch.ones((3, 2)),
            }

        def get_state_dict(self, model: str):
            assert model == "model"
            return self._state_dict

    expr = EqualExpr(
        left=TensorRef(model="model", expr="left"),
        right=TensorRef(model="model", expr="right"),
    )
    try:
        expr.evaluate(_Provider())
    except AssertTransformError as exc:
        assert "shape mismatch" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected shape mismatch error")


def _unit_test_equal_value_mismatch() -> None:
    class _Provider:
        def __init__(self) -> None:
            self._state_dict = {
                "left": torch.tensor([1.0, 2.0]),
                "right": torch.tensor([1.0, 3.0]),
            }

        def get_state_dict(self, model: str):
            assert model == "model"
            return self._state_dict

    expr = EqualExpr(
        left=TensorRef(model="model", expr="left"),
        right=TensorRef(model="model", expr="right"),
    )
    try:
        expr.evaluate(_Provider())
    except AssertTransformError as exc:
        assert "!=" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected value mismatch")


__unit_tests__ = [
    _unit_test_equal_dtype_compatibility,
    _unit_test_equal_shape_compatibility,
    _unit_test_equal_value_mismatch,
]
