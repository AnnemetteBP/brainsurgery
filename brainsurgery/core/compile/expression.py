from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import torch

from ..specs import (
    StateDictProvider,
    TensorRef,
    TransformError,
    ensure_mapping_payload,
    format_tensor_ref,
    must_model,
    parse_model_expr,
    parse_slice,
    validate_payload_keys,
)
from .name_mapping import match_expr_names
from .resolver import (
    _resolve_target_names as resolve_target_names_generic,
)
from .resolver import (
    _resolve_tensor_mappings as resolve_tensor_mappings_generic,
)
from .resolver import (
    _resolve_tensors as resolve_tensors_generic,
)


class Expression(Protocol):
    def evaluate(self, provider: StateDictProvider) -> None: ...
    def collect_models(self) -> set[str]: ...


@dataclass(frozen=True)
class ExpressionHelp:
    name: str
    payload_kind: str
    allowed_keys: set[str] | None = None
    required_keys: set[str] | None = None
    description: str | None = None


ExpressionCompiler = Callable[[Any, str | None], Expression]

_EXPRESSION_COMPILERS: dict[str, ExpressionCompiler] = {}
_EXPRESSION_HELP: dict[str, ExpressionHelp] = {}


def register_assert_expr(
    name: str,
    *,
    payload_kind: str,
    allowed_keys: set[str] | None = None,
    required_keys: set[str] | None = None,
    description: str | None = None,
) -> Callable[[ExpressionCompiler], ExpressionCompiler]:
    def decorator(fn: ExpressionCompiler) -> ExpressionCompiler:
        _EXPRESSION_COMPILERS[name] = fn
        _EXPRESSION_HELP[name] = ExpressionHelp(
            name=name,
            payload_kind=payload_kind,
            allowed_keys=None if allowed_keys is None else set(allowed_keys),
            required_keys=None if required_keys is None else set(required_keys),
            description=description,
        )
        return fn

    return decorator


def get_assert_expr_names() -> list[str]:
    return sorted(_EXPRESSION_COMPILERS)


def get_assert_expr_help(name: str | None = None) -> dict[str, ExpressionHelp] | ExpressionHelp:
    if name is None:
        return dict(_EXPRESSION_HELP)

    try:
        return _EXPRESSION_HELP[name]
    except KeyError as exc:
        raise TransformError(f"unknown assert op: {name!r}") from exc


def compile_assert_expr(raw: Any, default_model: str | None) -> Expression:
    if not isinstance(raw, dict) or len(raw) != 1:
        raise TransformError("assert expression must be a single-key mapping")

    op, payload = next(iter(raw.items()))

    try:
        compiler = _EXPRESSION_COMPILERS[op]
    except KeyError as exc:
        raise TransformError(f"unknown assert op: {op!r}") from exc

    return compiler(payload, default_model)


def require_mapping_assert_payload(
    payload: Any,
    *,
    op_name: str,
    allowed_keys: set[str],
    required_keys: set[str],
) -> dict[str, Any]:
    mapping = ensure_mapping_payload(payload, op_name)
    validate_payload_keys(
        mapping,
        op_name=op_name,
        allowed_keys=allowed_keys,
        required_keys=required_keys,
    )
    return mapping


def compile_tensor_ref_expr(raw: Any, default_model: str | None, op_name: str) -> TensorRef:
    if isinstance(raw, str):
        if not raw:
            raise TransformError(
                f"{op_name} must be a non-empty string reference or non-empty list of strings"
            )
    elif isinstance(raw, list):
        if not raw or not all(isinstance(item, str) and item for item in raw):
            raise TransformError(
                f"{op_name} must be a non-empty string reference or non-empty list of strings"
            )
    else:
        raise TransformError(
            f"{op_name} must be a non-empty string reference or non-empty list of strings"
        )

    ref = parse_model_expr(raw, default_model=default_model)
    if ref.slice_spec is not None:
        parse_slice(ref.slice_spec)
    return ref


def compile_shape(raw: Any) -> tuple[int, ...]:
    if not isinstance(raw, list) or not all(isinstance(x, int) for x in raw):
        raise TransformError("shape.is must be a list of integers")
    return tuple(raw)


def resolve_matches(
    ref: TensorRef,
    provider: StateDictProvider,
    *,
    op_name: str,
) -> list[str]:
    return resolve_target_names_generic(
        target_ref=ref,
        provider=provider,
        op_name=op_name,
        match_names=match_expr_names,
        error_type=TransformError,
    )


def resolve_tensors(
    ref: TensorRef,
    provider: StateDictProvider,
    *,
    op_name: str,
) -> list[tuple[TensorRef, torch.Tensor]]:
    return resolve_tensors_generic(
        ref,
        provider,
        op_name=op_name,
        resolve_names=resolve_matches,
    )


def resolve_tensor_mappings(
    from_ref: TensorRef,
    to_ref: TensorRef,
    provider: StateDictProvider,
    *,
    op_name: str,
) -> list[tuple[TensorRef, torch.Tensor, TensorRef, torch.Tensor]]:
    return resolve_tensor_mappings_generic(
        from_ref,
        to_ref,
        provider,
        op_name=op_name,
        error_type=TransformError,
    )


def collect_ref_models(ref: TensorRef) -> set[str]:
    return {must_model(ref)}


def collect_expr_models(exprs: list[Expression]) -> set[str]:
    models: set[str] = set()
    for expr in exprs:
        models.update(expr.collect_models())
    return models


def format_ref(ref: TensorRef) -> str:
    return format_tensor_ref(ref)


__all__ = [
    "Expression",
    "ExpressionHelp",
    "collect_expr_models",
    "collect_ref_models",
    "compile_assert_expr",
    "compile_shape",
    "compile_tensor_ref_expr",
    "format_ref",
    "get_assert_expr_help",
    "get_assert_expr_names",
    "register_assert_expr",
    "require_mapping_assert_payload",
    "resolve_matches",
    "resolve_tensor_mappings",
    "resolve_tensors",
]
