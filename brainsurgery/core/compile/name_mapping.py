import re
from collections.abc import Iterable
from dataclasses import dataclass

from ..specs import (
    StateDictProvider,
    TensorRef,
    TransformError,
    _Expr,
    _validate_expr_kind,
    format_tensor_ref,
    must_model,
    parse_slice,
)
from .matching import _MatchError, _StructuredPathMatcher

_MATCHER = _StructuredPathMatcher()


def match_expr_names(
    *,
    expr: _Expr,
    names: Iterable[str],
    op_name: str,
    role: str,
) -> list[str]:
    _validate_expr_kind(expr=expr, op_name=op_name, role=role)

    if isinstance(expr, str):
        try:
            return sorted(name for name in names if re.fullmatch(expr, name))
        except re.error as exc:
            raise TransformError(f"{op_name} invalid {role} regex {expr!r}: {exc}") from exc

    assert isinstance(expr, list)
    try:
        return sorted(name for name in names if _MATCHER.match(expr, name) is not None)
    except _MatchError as exc:
        raise TransformError(f"{op_name} invalid structured {role} pattern: {exc}") from exc


def _match_structured_expr(
    *,
    expr: list[str],
    name: str,
    op_name: str,
    role: str,
):
    _validate_expr_kind(expr=expr, op_name=op_name, role=role)
    try:
        return _MATCHER.match(expr, name)
    except _MatchError as exc:
        raise TransformError(f"{op_name} invalid structured {role} pattern: {exc}") from exc


def _rewrite_structured_expr(
    *,
    expr: list[str],
    match,
    op_name: str,
    role: str,
) -> str:
    _validate_expr_kind(expr=expr, op_name=op_name, role=role)
    try:
        return _MATCHER.rewrite(expr, match)
    except _MatchError as exc:
        raise TransformError(f"{op_name} invalid structured {role} pattern: {exc}") from exc


@dataclass(frozen=True)
class ResolvedMapping:
    src_model: str
    src_name: str
    src_slice: tuple[object, ...] | None
    dst_model: str
    dst_name: str
    dst_slice: tuple[object, ...] | None


def _resolve_name_mappings_regex(
    *,
    from_ref: TensorRef,
    to_ref: TensorRef,
    provider: StateDictProvider,
    op_name: str,
) -> list[ResolvedMapping]:
    if not isinstance(from_ref.expr, str) or not isinstance(to_ref.expr, str):
        raise TransformError(
            f"{op_name} internal error: regex resolver expected string expressions"
        )

    src_model = must_model(from_ref)
    dst_model = must_model(to_ref)
    src_sd = provider.get_state_dict(src_model)

    src_names = match_expr_names(
        expr=from_ref.expr,
        names=src_sd.keys(),
        op_name=op_name,
        role="source",
    )
    if not src_names:
        raise TransformError(
            f"{op_name} source matched zero tensors: "
            f"{format_tensor_ref(from_ref)}; available tensors: {sorted(src_sd.keys())}"
        )

    src_slice = parse_slice(from_ref.slice_spec) if from_ref.slice_spec else None
    dst_slice = parse_slice(to_ref.slice_spec) if to_ref.slice_spec else None

    dst_names_seen: set[str] = set()
    resolved: list[ResolvedMapping] = []

    for src_name in src_names:
        try:
            dst_name = re.sub(from_ref.expr, to_ref.expr, src_name)
        except re.error as exc:
            raise TransformError(
                f"{op_name} invalid regex rewrite from {from_ref.expr!r} to {to_ref.expr!r}: {exc}"
            ) from exc

        if dst_name in dst_names_seen:
            raise TransformError(f"{op_name} destination collision: {dst_model}::{dst_name}")

        dst_names_seen.add(dst_name)
        resolved.append(
            ResolvedMapping(
                src_model=src_model,
                src_name=src_name,
                src_slice=src_slice,
                dst_model=dst_model,
                dst_name=dst_name,
                dst_slice=dst_slice,
            )
        )

    return resolved


def _resolve_name_mappings_structured(
    *,
    from_ref: TensorRef,
    to_ref: TensorRef,
    provider: StateDictProvider,
    op_name: str,
) -> list[ResolvedMapping]:
    if not isinstance(from_ref.expr, list) or not isinstance(to_ref.expr, list):
        raise TransformError(
            f"{op_name} internal error: structured resolver expected list expressions"
        )

    src_model = must_model(from_ref)
    dst_model = must_model(to_ref)
    src_sd = provider.get_state_dict(src_model)

    src_slice = parse_slice(from_ref.slice_spec) if from_ref.slice_spec else None
    dst_slice = parse_slice(to_ref.slice_spec) if to_ref.slice_spec else None

    matched_any = False
    dst_names_seen: set[str] = set()
    resolved: list[ResolvedMapping] = []

    for src_name in sorted(src_sd.keys()):
        match = _match_structured_expr(
            expr=from_ref.expr,
            name=src_name,
            op_name=op_name,
            role="source",
        )
        if match is None:
            continue

        matched_any = True

        dst_name = _rewrite_structured_expr(
            expr=to_ref.expr,
            match=match,
            op_name=op_name,
            role="destination",
        )

        if dst_name in dst_names_seen:
            raise TransformError(f"{op_name} destination collision: {dst_model}::{dst_name}")

        dst_names_seen.add(dst_name)
        resolved.append(
            ResolvedMapping(
                src_model=src_model,
                src_name=src_name,
                src_slice=src_slice,
                dst_model=dst_model,
                dst_name=dst_name,
                dst_slice=dst_slice,
            )
        )

    if not matched_any:
        raise TransformError(
            f"{op_name} source matched zero tensors: "
            f"{format_tensor_ref(from_ref)}; available tensors: {sorted(src_sd.keys())}"
        )

    return resolved


def resolve_name_mappings(
    *,
    from_ref: TensorRef,
    to_ref: TensorRef,
    provider: StateDictProvider,
    op_name: str,
) -> list[ResolvedMapping]:
    src_slice = from_ref.slice_spec
    dst_slice = to_ref.slice_spec

    if src_slice is not None and not isinstance(src_slice, str):
        raise TransformError(f"{op_name} source slice must be a string")
    if dst_slice is not None and not isinstance(dst_slice, str):
        raise TransformError(f"{op_name} destination slice must be a string")

    if isinstance(from_ref.expr, str) and isinstance(to_ref.expr, str):
        resolved = _resolve_name_mappings_regex(
            from_ref=from_ref,
            to_ref=to_ref,
            provider=provider,
            op_name=op_name,
        )
        if not resolved:
            raise TransformError(f"{op_name} internal error: resolved zero mappings")
        return resolved

    if isinstance(from_ref.expr, list) and isinstance(to_ref.expr, list):
        resolved = _resolve_name_mappings_structured(
            from_ref=from_ref,
            to_ref=to_ref,
            provider=provider,
            op_name=op_name,
        )
        if not resolved:
            raise TransformError(f"{op_name} internal error: resolved zero mappings")
        return resolved

    raise TransformError(
        f"{op_name} requires from/to expressions of the same kind: "
        "either both strings (regex mode) or both lists (structured mode)"
    )


def _require_dest_missing(
    *,
    mappings: list[ResolvedMapping],
    provider: StateDictProvider,
    op_name: str,
) -> None:
    for item in mappings:
        dst_sd = provider.get_state_dict(item.dst_model)
        if item.dst_name in dst_sd:
            raise TransformError(
                f"{op_name} destination already exists: {item.dst_model}::{item.dst_name}"
            )


def _require_dest_present(
    *,
    mappings: list[ResolvedMapping],
    provider: StateDictProvider,
    op_name: str,
) -> None:
    for item in mappings:
        dst_sd = provider.get_state_dict(item.dst_model)
        if item.dst_name not in dst_sd:
            raise TransformError(
                f"{op_name} destination missing: {item.dst_model}::{item.dst_name}"
            )


__all__ = [
    "match_expr_names",
    "ResolvedMapping",
    "_require_dest_missing",
    "_require_dest_present",
    "resolve_name_mappings",
]
