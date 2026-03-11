from __future__ import annotations

from collections.abc import Iterable

import re

from .matching import MatchError, StructuredPathMatcher
from .refs import Expr, _validate_expr_kind
from .types import TransformError


_MATCHER = StructuredPathMatcher()


def match_expr_names(
    *,
    expr: Expr,
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
    except MatchError as exc:
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
    except MatchError as exc:
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
    except MatchError as exc:
        raise TransformError(f"{op_name} invalid structured {role} pattern: {exc}") from exc


__all__ = [
    "match_expr_names",
]
