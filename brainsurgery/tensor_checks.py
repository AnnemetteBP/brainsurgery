from __future__ import annotations

import torch

from .transform import TransformError


def require_same_shape_dtype_device(
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    error_type: type[TransformError],
    op_name: str,
    left_name: str,
    right_name: str,
) -> None:
    if left.shape != right.shape:
        raise error_type(
            f"shape mismatch {op_name} {left_name} -> {right_name}: "
            f"{tuple(left.shape)} != {tuple(right.shape)}"
        )
    if left.dtype != right.dtype:
        raise error_type(
            f"dtype mismatch {op_name} {left_name} -> {right_name}: "
            f"{left.dtype} != {right.dtype}"
        )
    if left.device != right.device:
        raise error_type(
            f"device mismatch {op_name} {left_name} -> {right_name}: "
            f"{left.device} != {right.device}"
        )


def require_same_shape_dtype_device3(
    first: torch.Tensor,
    second: torch.Tensor,
    dest: torch.Tensor,
    *,
    error_type: type[TransformError],
    op_name: str,
    first_name: str,
    second_name: str,
    dest_name: str,
    symbol: str,
) -> None:
    if first.shape != second.shape or first.shape != dest.shape:
        raise error_type(
            f"shape mismatch {op_name} {first_name} {symbol} {second_name} -> {dest_name}: "
            f"{tuple(first.shape)} {symbol} {tuple(second.shape)} -> {tuple(dest.shape)}"
        )
    if first.dtype != second.dtype or first.dtype != dest.dtype:
        raise error_type(
            f"dtype mismatch {op_name} {first_name} {symbol} {second_name} -> {dest_name}: "
            f"{first.dtype} {symbol} {second.dtype} -> {dest.dtype}"
        )
    if first.device != second.device or first.device != dest.device:
        raise error_type(
            f"device mismatch {op_name} {first_name} {symbol} {second_name} -> {dest_name}: "
            f"{first.device} {symbol} {second.device} -> {dest.device}"
        )


def _unit_test_require_same_shape_dtype_device_accepts_matching_tensors() -> None:
    left = torch.ones((2,), dtype=torch.float32)
    right = torch.ones((2,), dtype=torch.float32)
    require_same_shape_dtype_device(
        left,
        right,
        error_type=TransformError,
        op_name="assigning",
        left_name="a",
        right_name="b",
    )


def _unit_test_require_same_shape_dtype_device3_rejects_shape_mismatch() -> None:
    try:
        require_same_shape_dtype_device3(
            torch.ones((2,)),
            torch.ones((1,)),
            torch.ones((2,)),
            error_type=TransformError,
            op_name="adding",
            first_name="a",
            second_name="b",
            dest_name="dst",
            symbol="+",
        )
    except TransformError as exc:
        assert "shape mismatch" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected shape mismatch")


__unit_tests__ = [
    _unit_test_require_same_shape_dtype_device_accepts_matching_tensors,
    _unit_test_require_same_shape_dtype_device3_rejects_shape_mismatch,
]
