from .flags import get_runtime_flags
from .frontend import emit_line


def emit_verbose_binary_activity(transform_name: str, src_name: str, dst_name: str) -> None:
    flags = get_runtime_flags()
    if not flags.verbose:
        return
    prefix = "dry-run " if flags.dry_run else ""
    emit_line(f"{prefix}{transform_name}: {src_name} -> {dst_name}")


def emit_verbose_unary_activity(transform_name: str, target_name: str) -> None:
    flags = get_runtime_flags()
    if not flags.verbose:
        return
    prefix = "dry-run " if flags.dry_run else ""
    emit_line(f"{prefix}{transform_name}: {target_name}")


def emit_verbose_ternary_activity(
    transform_name: str,
    src_a_name: str,
    src_b_name: str,
    dst_name: str,
) -> None:
    flags = get_runtime_flags()
    if not flags.verbose:
        return
    prefix = "dry-run " if flags.dry_run else ""
    emit_line(f"{prefix}{transform_name}: {src_a_name}, {src_b_name} -> {dst_name}")


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
