from __future__ import annotations

import torch

from ..core.transform_types import TransformError


def require_same_shape_dtype_device(
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    op_name: str,
    left_name: str,
    right_name: str,
) -> None:
    if left.shape != right.shape:
        raise TransformError(
            f"shape mismatch {op_name} {left_name} -> {right_name}: "
            f"{tuple(left.shape)} != {tuple(right.shape)}"
        )
    if left.dtype != right.dtype:
        raise TransformError(
            f"dtype mismatch {op_name} {left_name} -> {right_name}: "
            f"{left.dtype} != {right.dtype}"
        )
    if left.device != right.device:
        raise TransformError(
            f"device mismatch {op_name} {left_name} -> {right_name}: "
            f"{left.device} != {right.device}"
        )


def require_same_shape_dtype_device3(
    first: torch.Tensor,
    second: torch.Tensor,
    dest: torch.Tensor,
    *,
    op_name: str,
    first_name: str,
    second_name: str,
    dest_name: str,
    symbol: str,
) -> None:
    if first.shape != second.shape or first.shape != dest.shape:
        raise TransformError(
            f"shape mismatch {op_name} {first_name} {symbol} {second_name} -> {dest_name}: "
            f"{tuple(first.shape)} {symbol} {tuple(second.shape)} -> {tuple(dest.shape)}"
        )
    if first.dtype != second.dtype or first.dtype != dest.dtype:
        raise TransformError(
            f"dtype mismatch {op_name} {first_name} {symbol} {second_name} -> {dest_name}: "
            f"{first.dtype} {symbol} {second.dtype} -> {dest.dtype}"
        )
    if first.device != second.device or first.device != dest.device:
        raise TransformError(
            f"device mismatch {op_name} {first_name} {symbol} {second_name} -> {dest_name}: "
            f"{first.device} {symbol} {second.device} -> {dest.device}"
        )


__all__ = ["require_same_shape_dtype_device", "require_same_shape_dtype_device3"]
