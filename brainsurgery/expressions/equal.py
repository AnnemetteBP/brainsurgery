from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from ..expression import (
    AssertTransformError,
    compile_tensor_ref_expr,
    format_ref,
    register_assert_expr,
    resolve_tensor_mappings,
    require_mapping_assert_payload,
)
from ..transform import StateDictProvider, TensorRef, must_model


@dataclass(frozen=True)
class EqualExpr:
    left: TensorRef
    right: TensorRef
    eps: float | None = None

    def evaluate(self, provider: StateDictProvider) -> None:
        mappings = resolve_tensor_mappings(self.left, self.right, provider, op_name="equal")
        for left_ref, left, right_ref, right in mappings:
            if left.shape != right.shape:
                raise AssertTransformError(
                    f"equal failed: shape mismatch {tuple(left.shape)} != {tuple(right.shape)} "
                    f"for {format_ref(left_ref)} vs {format_ref(right_ref)}"
                )
            if left.dtype != right.dtype:
                raise AssertTransformError(
                    f"equal failed: dtype mismatch {left.dtype} != {right.dtype} "
                    f"for {format_ref(left_ref)} vs {format_ref(right_ref)}"
                )
            if left.device != right.device:
                raise AssertTransformError(
                    f"equal failed: device mismatch {left.device} != {right.device} "
                    f"for {format_ref(left_ref)} vs {format_ref(right_ref)}"
                )

            if self.eps is None:
                is_equal = torch.equal(left, right)
            else:
                if left.is_complex():
                    diff = torch.abs(left.to(torch.complex128) - right.to(torch.complex128))
                else:
                    diff = torch.abs(left.to(torch.float64) - right.to(torch.float64))
                is_equal = bool(torch.all(diff <= self.eps).item())

            if not is_equal:
                raise AssertTransformError(
                    f"equal failed: {format_ref(left_ref)} != {format_ref(right_ref)}"
                )

    def collect_models(self) -> set[str]:
        return {must_model(self.left), must_model(self.right)}


@register_assert_expr(
    "equal",
    payload_kind="mapping",
    allowed_keys={"left", "right", "eps"},
    required_keys={"left", "right"},
    description=(
        "Succeeds if two tensors have the same shape and dtype, and their values are equal "
        "(or within 'eps' absolute tolerance when provided)."
    ),
)
def compile_equal_expr(payload: Any, default_model: str | None) -> EqualExpr:
    payload = require_mapping_assert_payload(
        payload,
        op_name="equal",
        allowed_keys={"left", "right", "eps"},
        required_keys={"left", "right"},
    )
    left = compile_tensor_ref_expr(payload["left"], default_model, "equal.left")
    right = compile_tensor_ref_expr(payload["right"], default_model, "equal.right")
    raw_eps = payload.get("eps")
    if raw_eps is None:
        eps = None
    else:
        if isinstance(raw_eps, bool) or not isinstance(raw_eps, (int, float)):
            raise AssertTransformError("equal.eps must be a non-negative number")
        eps = float(raw_eps)
        if eps < 0:
            raise AssertTransformError("equal.eps must be a non-negative number")
    return EqualExpr(left=left, right=right, eps=eps)


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


def _unit_test_equal_compile_rejects_negative_eps() -> None:
    try:
        compile_equal_expr({"left": "x", "right": "y", "eps": -1e-3}, default_model="model")
    except AssertTransformError as exc:
        assert "equal.eps must be a non-negative number" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected equal.eps validation error")


def _unit_test_equal_value_within_eps_succeeds() -> None:
    class _Provider:
        def __init__(self) -> None:
            self._state_dict = {
                "left": torch.tensor([1.0, 2.0]),
                "right": torch.tensor([1.0, 2.0005]),
            }

        def get_state_dict(self, model: str):
            assert model == "model"
            return self._state_dict

    expr = EqualExpr(
        left=TensorRef(model="model", expr="left"),
        right=TensorRef(model="model", expr="right"),
        eps=1e-3,
    )
    expr.evaluate(_Provider())


def _unit_test_equal_value_outside_eps_fails() -> None:
    class _Provider:
        def __init__(self) -> None:
            self._state_dict = {
                "left": torch.tensor([1.0, 2.0]),
                "right": torch.tensor([1.0, 2.0005]),
            }

        def get_state_dict(self, model: str):
            assert model == "model"
            return self._state_dict

    expr = EqualExpr(
        left=TensorRef(model="model", expr="left"),
        right=TensorRef(model="model", expr="right"),
        eps=1e-4,
    )
    try:
        expr.evaluate(_Provider())
    except AssertTransformError as exc:
        assert "!=" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected value mismatch with eps")


def _unit_test_equal_regex_mapping_success() -> None:
    class _Provider:
        def __init__(self) -> None:
            self._state_dicts = {
                "model": {
                    "a": torch.tensor([1.0, 2.0]),
                    "b": torch.tensor([3.0, 4.0]),
                },
                "orig": {
                    "a": torch.tensor([1.0, 2.0]),
                    "b": torch.tensor([3.0, 4.0]),
                },
            }

        def get_state_dict(self, model: str):
            return self._state_dicts[model]

    expr = EqualExpr(
        left=TensorRef(model="model", expr="(.+)"),
        right=TensorRef(model="orig", expr=r"\1"),
    )
    expr.evaluate(_Provider())


def _unit_test_equal_regex_mapping_missing_destination() -> None:
    class _Provider:
        def __init__(self) -> None:
            self._state_dicts = {
                "model": {
                    "a": torch.tensor([1.0, 2.0]),
                    "b": torch.tensor([3.0, 4.0]),
                },
                "orig": {
                    "a": torch.tensor([1.0, 2.0]),
                },
            }

        def get_state_dict(self, model: str):
            return self._state_dicts[model]

    expr = EqualExpr(
        left=TensorRef(model="model", expr="(.+)"),
        right=TensorRef(model="orig", expr=r"\1"),
    )
    try:
        expr.evaluate(_Provider())
    except AssertTransformError as exc:
        assert "destination missing" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected destination missing error")


def _unit_test_equal_structured_mapping_success() -> None:
    class _Provider:
        def __init__(self) -> None:
            self._state_dict = {
                "block.0.weight": torch.tensor([1.0, 2.0]),
                "block.1.weight": torch.tensor([3.0, 4.0]),
                "backup.0.weight": torch.tensor([1.0, 2.0]),
                "backup.1.weight": torch.tensor([3.0, 4.0]),
            }

        def get_state_dict(self, model: str):
            assert model == "model"
            return self._state_dict

    expr = EqualExpr(
        left=TensorRef(model="model", expr=["block", "$i", "weight"]),
        right=TensorRef(model="model", expr=["backup", "${i}", "weight"]),
    )
    expr.evaluate(_Provider())


__unit_tests__ = [
    _unit_test_equal_dtype_compatibility,
    _unit_test_equal_shape_compatibility,
    _unit_test_equal_value_mismatch,
    _unit_test_equal_compile_rejects_negative_eps,
    _unit_test_equal_value_within_eps_succeeds,
    _unit_test_equal_value_outside_eps_fails,
    _unit_test_equal_regex_mapping_success,
    _unit_test_equal_regex_mapping_missing_destination,
    _unit_test_equal_structured_mapping_success,
]
