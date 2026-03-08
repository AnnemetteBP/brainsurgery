from __future__ import annotations

import re

from .matching import StructuredPathError, StructuredPathMatcher
from .transform import StateDictProvider, TensorRef, TransformError, must_model


_MATCHER = StructuredPathMatcher()


def format_target_ref(ref: TensorRef) -> str:
    model = must_model(ref)

    if isinstance(ref.expr, str):
        expr = ref.expr
    elif isinstance(ref.expr, list):
        expr = "[" + ", ".join(repr(part) for part in ref.expr) + "]"
    else:
        expr = repr(ref.expr)

    if ref.slice_spec is None:
        return f"{model}::{expr}"
    return f"{model}::{expr}::{ref.slice_spec}"


def resolve_target_names(
    *,
    target_ref: TensorRef,
    provider: StateDictProvider,
    op_name: str,
    error_type: type[TransformError],
) -> list[str]:
    model = must_model(target_ref)
    sd = provider.get_state_dict(model)

    if isinstance(target_ref.expr, str):
        try:
            matches = sorted(
                name for name in sd.keys() if re.fullmatch(target_ref.expr, name)
            )
        except re.error as exc:
            raise error_type(
                f"{op_name} invalid target regex {target_ref.expr!r}: {exc}"
            ) from exc

    elif isinstance(target_ref.expr, list):
        try:
            matches = sorted(
                name for name in sd.keys() if _MATCHER.match(target_ref.expr, name) is not None
            )
        except StructuredPathError as exc:
            raise error_type(f"{op_name} invalid structured target pattern: {exc}") from exc

    else:
        raise error_type(
            f"{op_name} target expression has invalid type: "
            f"{type(target_ref.expr).__name__}"
        )

    if not matches:
        raise error_type(f"{op_name} matched zero tensors: {format_target_ref(target_ref)}")

    return matches

