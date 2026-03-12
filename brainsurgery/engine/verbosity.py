from __future__ import annotations

from ..core import ResolvedMapping
from ..core import ResolvedTernaryMapping
from .flags import get_runtime_flags
from .frontend import emit_line


def emit_verbose_binary_activity(transform_name: str, item: ResolvedMapping) -> None:
    flags = get_runtime_flags()
    if not flags.verbose:
        return
    prefix = "dry-run " if flags.dry_run else ""
    emit_line(f"{prefix}{transform_name}: {item.src_name} -> {item.dst_name}")


def emit_verbose_unary_activity(transform_name: str, target_name: str) -> None:
    flags = get_runtime_flags()
    if not flags.verbose:
        return
    prefix = "dry-run " if flags.dry_run else ""
    emit_line(f"{prefix}{transform_name}: {target_name}")


def emit_verbose_ternary_activity(transform_name: str, item: ResolvedTernaryMapping) -> None:
    flags = get_runtime_flags()
    if not flags.verbose:
        return
    prefix = "dry-run " if flags.dry_run else ""
    emit_line(f"{prefix}{transform_name}: {item.src_a_name}, {item.src_b_name} -> {item.dst_name}")


def emit_verbose_event(transform_name: str, detail: str | None = None) -> None:
    flags = get_runtime_flags()
    if not flags.verbose:
        return
    prefix = "dry-run " if flags.dry_run else ""
    if detail:
        emit_line(f"{prefix}{transform_name}: {detail}")
        return
    emit_line(f"{prefix}{transform_name}")


__all__ = [
    "emit_verbose_binary_activity",
    "emit_verbose_unary_activity",
    "emit_verbose_ternary_activity",
    "emit_verbose_event",
]
