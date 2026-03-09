from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol

import torch

from .transforms.unary import format_target_ref, resolve_target_names
from .transform import (
    ResolvedMapping,
    StateDictProvider,
    TensorRef,
    TransformError,
    ensure_mapping_payload,
    must_model,
    parse_model_expr,
    parse_slice,
    require_dest_present,
    resolve_name_mappings,
    select_tensor,
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
    return resolve_target_names(
        target_ref=ref,
        provider=provider,
        op_name=op_name,
        error_type=AssertTransformError,
    )


def resolve_single_tensor(ref: TensorRef, provider: StateDictProvider, op_name: str) -> torch.Tensor:
    model = must_model(ref)
    sd = provider.get_state_dict(model)
    matches = resolve_matches(ref, provider, op_name=op_name)

    if len(matches) == 0:
        raise AssertTransformError(f"{op_name} failed: {format_ref(ref)} matched zero tensors")
    if len(matches) != 1:
        raise AssertTransformError(
            f"{op_name} failed: {format_ref(ref)} matched {len(matches)} tensors, expected 1"
        )

    tensor = sd[matches[0]]
    slice_spec = parse_slice(ref.slice_spec) if ref.slice_spec else None
    return select_tensor(tensor, slice_spec)


def resolve_tensors(
    ref: TensorRef,
    provider: StateDictProvider,
    *,
    op_name: str,
) -> list[tuple[TensorRef, torch.Tensor]]:
    model = must_model(ref)
    sd = provider.get_state_dict(model)
    matches = resolve_matches(ref, provider, op_name=op_name)
    slice_spec = parse_slice(ref.slice_spec) if ref.slice_spec else None
    resolved: list[tuple[TensorRef, torch.Tensor]] = []

    for name in matches:
        resolved_ref = TensorRef(model=model, expr=name, slice_spec=ref.slice_spec)
        resolved.append((resolved_ref, select_tensor(sd[name], slice_spec)))

    return resolved


def resolve_tensor_mappings(
    from_ref: TensorRef,
    to_ref: TensorRef,
    provider: StateDictProvider,
    *,
    op_name: str,
) -> list[tuple[TensorRef, torch.Tensor, TensorRef, torch.Tensor]]:
    try:
        mappings = resolve_name_mappings(
            from_ref=from_ref,
            to_ref=to_ref,
            provider=provider,
            op_name=op_name,
        )
        require_dest_present(mappings=mappings, provider=provider, op_name=op_name)
    except TransformError as exc:
        raise AssertTransformError(str(exc)) from exc

    resolved: list[tuple[TensorRef, torch.Tensor, TensorRef, torch.Tensor]] = []
    for item in mappings:
        resolved.append(_resolve_mapping_tensors(item, from_ref=from_ref, to_ref=to_ref, provider=provider))
    return resolved


def _resolve_mapping_tensors(
    item: ResolvedMapping,
    *,
    from_ref: TensorRef,
    to_ref: TensorRef,
    provider: StateDictProvider,
) -> tuple[TensorRef, torch.Tensor, TensorRef, torch.Tensor]:
    src_sd = provider.get_state_dict(item.src_model)
    dst_sd = provider.get_state_dict(item.dst_model)

    left_ref = TensorRef(model=item.src_model, expr=item.src_name, slice_spec=from_ref.slice_spec)
    right_ref = TensorRef(model=item.dst_model, expr=item.dst_name, slice_spec=to_ref.slice_spec)
    left = select_tensor(src_sd[item.src_name], item.src_slice)
    right = select_tensor(dst_sd[item.dst_name], item.dst_slice)
    return left_ref, left, right_ref, right


def collect_ref_models(ref: TensorRef) -> set[str]:
    return {must_model(ref)}


def collect_expr_models(exprs: list[AssertExpr]) -> set[str]:
    models: set[str] = set()
    for expr in exprs:
        models.update(expr.collect_models())
    return models


def format_ref(ref: TensorRef) -> str:
    return format_target_ref(ref)
