from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from .transform_types import TransformError


Expr = str | list[str]


@dataclass(frozen=True)
class TensorRef:
    model: Optional[str]
    expr: Expr
    slice_spec: Optional[str] = None


def parse_model_expr(raw: object, default_model: Optional[str] = None) -> TensorRef:
    if isinstance(raw, list):
        if default_model is None:
            raise TransformError("missing model alias for structured reference")
        if not raw:
            raise TransformError("structured reference must be a non-empty list")
        if not all(isinstance(item, str) and item for item in raw):
            raise TransformError("structured reference must be a non-empty list of non-empty strings")
        return TensorRef(model=default_model, expr=raw, slice_spec=None)

    if not isinstance(raw, str) or not raw:
        raise TransformError("reference must be a non-empty string or a non-empty list of strings")

    parts = raw.split("::")
    if len(parts) == 1:
        if default_model is None:
            raise TransformError(f"missing model alias in reference: {raw!r}")
        return TensorRef(model=default_model, expr=parts[0], slice_spec=None)

    if len(parts) == 2:
        head, tail = parts
        if default_model is not None and looks_like_slice(tail):
            return TensorRef(model=default_model, expr=head, slice_spec=tail)
        return TensorRef(model=head or default_model, expr=tail, slice_spec=None)

    if len(parts) == 3:
        head, expr, slice_spec = parts
        if not looks_like_slice(slice_spec):
            raise TransformError(f"invalid slice syntax in reference: {raw!r}")
        model = head or default_model
        if model is None:
            raise TransformError(f"missing model alias in reference: {raw!r}")
        return TensorRef(model=model, expr=expr, slice_spec=slice_spec)

    raise TransformError(f"invalid reference syntax: {raw!r}")


def parse_slice(raw: str) -> Tuple[object, ...]:
    if not looks_like_slice(raw):
        raise TransformError(f"invalid slice syntax: {raw!r}")

    inner = raw[1:-1].strip()
    if not inner:
        return tuple()

    parts = [part.strip() for part in inner.split(",")]
    if any(part == "" for part in parts):
        raise TransformError(f"invalid empty slice component in {raw!r}")

    return tuple(parse_slice_component(part) for part in parts)


def parse_slice_component(raw: str) -> object:
    if raw == ":":
        return slice(None, None, None)

    if ":" not in raw:
        return parse_int(raw)

    parts = raw.split(":")
    if len(parts) not in (2, 3):
        raise TransformError(f"invalid slice component: {raw!r}")

    start = parse_optional_int(parts[0])
    stop = parse_optional_int(parts[1])
    step = parse_optional_int(parts[2]) if len(parts) == 3 else None
    return slice(start, stop, step)


def select_tensor(tensor: torch.Tensor, slice_spec: Optional[Tuple[object, ...]]) -> torch.Tensor:
    if slice_spec is None:
        return tensor
    try:
        return tensor[slice_spec]
    except Exception as exc:  # pragma: no cover
        raise TransformError(
            f"failed to apply slice {slice_spec!r} to tensor with shape {tuple(tensor.shape)}"
        ) from exc


def parse_int(raw: str) -> int:
    try:
        return int(raw)
    except ValueError as exc:
        raise TransformError(f"invalid integer in slice component: {raw!r}") from exc


def parse_optional_int(raw: str) -> Optional[int]:
    return None if raw == "" else parse_int(raw)


def looks_like_slice(raw: str) -> bool:
    return raw.startswith("[") and raw.endswith("]")


def must_model(ref: TensorRef) -> str:
    if ref.model is None:
        raise TransformError(f"reference is missing model alias: {ref}")
    return ref.model


def format_ref_expr(expr: Expr) -> str:
    if isinstance(expr, str):
        return expr
    return "[" + ", ".join(repr(part) for part in expr) + "]"


def format_tensor_ref(ref: TensorRef) -> str:
    model = must_model(ref)
    expr = format_ref_expr(ref.expr)
    if ref.slice_spec is None:
        return f"{model}::{expr}"
    return f"{model}::{expr}::{ref.slice_spec}"


def validate_expr_kind(
    *,
    expr: Expr,
    op_name: str,
    role: str,
) -> None:
    if isinstance(expr, str):
        if not expr:
            raise TransformError(f"{op_name} {role} regex must be non-empty")
        return

    if isinstance(expr, list):
        if not expr:
            raise TransformError(f"{op_name} structured {role} pattern must be non-empty")
        if not all(isinstance(item, str) and item for item in expr):
            raise TransformError(
                f"{op_name} structured {role} pattern must be a list of non-empty strings"
            )
        return

    raise TransformError(f"{op_name} {role} expression has invalid type: {type(expr).__name__}")
