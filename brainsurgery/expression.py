from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol

import torch

from .resolver import (
    resolve_single_tensor as resolve_single_tensor_generic,
    resolve_target_names as resolve_target_names_generic,
    resolve_tensor_mappings as resolve_tensor_mappings_generic,
    resolve_tensors as resolve_tensors_generic,
)
from .transform import (
    StateDictProvider,
    TensorRef,
    TransformError,
    ensure_mapping_payload,
    format_tensor_ref,
    must_model,
    match_expr_names,
    parse_model_expr,
    parse_slice,
    validate_payload_keys,
)


class AssertTransformError(TransformError):
    pass


class AssertExpr(Protocol):
    def evaluate(self, provider: StateDictProvider) -> None: ...
    def collect_models(self) -> set[str]: ...


@dataclass(frozen=True)
class AssertExprHelp:
    name: str
    payload_kind: str
    allowed_keys: set[str] | None = None
    required_keys: set[str] | None = None
    description: str | None = None


AssertExprCompiler = Callable[[Any, str | None], AssertExpr]

_ASSERT_EXPR_COMPILERS: dict[str, AssertExprCompiler] = {}
_ASSERT_EXPR_HELP: dict[str, AssertExprHelp] = {}


def register_assert_expr(
    name: str,
    *,
    payload_kind: str,
    allowed_keys: set[str] | None = None,
    required_keys: set[str] | None = None,
    description: str | None = None,
) -> Callable[[AssertExprCompiler], AssertExprCompiler]:
    def decorator(fn: AssertExprCompiler) -> AssertExprCompiler:
        _ASSERT_EXPR_COMPILERS[name] = fn
        _ASSERT_EXPR_HELP[name] = AssertExprHelp(
            name=name,
            payload_kind=payload_kind,
            allowed_keys=None if allowed_keys is None else set(allowed_keys),
            required_keys=None if required_keys is None else set(required_keys),
            description=description,
        )
        return fn

    return decorator


def get_assert_expr_names() -> list[str]:
    return sorted(_ASSERT_EXPR_COMPILERS)


def get_assert_expr_help(name: str | None = None) -> dict[str, AssertExprHelp] | AssertExprHelp:
    if name is None:
        return dict(_ASSERT_EXPR_HELP)

    try:
        return _ASSERT_EXPR_HELP[name]
    except KeyError as exc:
        raise AssertTransformError(f"unknown assert op: {name!r}") from exc


def compile_assert_expr(raw: Any, default_model: str | None) -> AssertExpr:
    if not isinstance(raw, dict) or len(raw) != 1:
        raise AssertTransformError("assert expression must be a single-key mapping")

    op, payload = next(iter(raw.items()))

    try:
        compiler = _ASSERT_EXPR_COMPILERS[op]
    except KeyError as exc:
        raise AssertTransformError(f"unknown assert op: {op!r}") from exc

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
            raise AssertTransformError(
                f"{op_name} must be a non-empty string reference or non-empty list of strings"
            )
    elif isinstance(raw, list):
        if not raw or not all(isinstance(item, str) and item for item in raw):
            raise AssertTransformError(
                f"{op_name} must be a non-empty string reference or non-empty list of strings"
            )
    else:
        raise AssertTransformError(
            f"{op_name} must be a non-empty string reference or non-empty list of strings"
        )

    ref = parse_model_expr(raw, default_model=default_model)
    if ref.slice_spec is not None:
        parse_slice(ref.slice_spec)
    return ref


def compile_shape(raw: Any) -> tuple[int, ...]:
    if not isinstance(raw, list) or not all(isinstance(x, int) for x in raw):
        raise AssertTransformError("shape.is must be a list of integers")
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
        error_type=AssertTransformError,
    )


def resolve_single_tensor(ref: TensorRef, provider: StateDictProvider, op_name: str) -> torch.Tensor:
    return resolve_single_tensor_generic(
        ref,
        provider,
        op_name=op_name,
        resolve_names=resolve_matches,
        error_type=AssertTransformError,
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
        error_type=AssertTransformError,
    )


def collect_ref_models(ref: TensorRef) -> set[str]:
    return {must_model(ref)}


def collect_expr_models(exprs: list[AssertExpr]) -> set[str]:
    models: set[str] = set()
    for expr in exprs:
        models.update(expr.collect_models())
    return models


def format_ref(ref: TensorRef) -> str:
    return format_tensor_ref(ref)
