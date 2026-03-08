from __future__ import annotations

from .unary import UnarySpec, UnaryTransform
from ..matching import StructuredPathError, StructuredPathMatcher
from ..transform import (
    StateDictProvider,
    TensorRef,
    TransformError,
    must_model,
    register_transform,
)


class DeleteTransformError(TransformError):
    pass


_MATCHER = StructuredPathMatcher()


class DeleteTransform(UnaryTransform[UnarySpec]):
    name = "delete"
    error_type = DeleteTransformError
    spec_type = UnarySpec
    progress_desc = "Applying delete transforms"

    def validate_target_ref(self, target_ref: TensorRef) -> None:
        if target_ref.slice_spec is not None:
            raise DeleteTransformError("delete target must not be sliced")

    def resolve_targets(self, spec: UnarySpec, provider: StateDictProvider) -> list[str]:
        model = must_model(spec.target_ref)
        sd = provider.get_state_dict(model)

        if isinstance(spec.target_ref.expr, str):
            import re

            try:
                matches = sorted(
                    name for name in sd.keys() if re.fullmatch(spec.target_ref.expr, name)
                )
            except re.error as exc:
                raise DeleteTransformError(
                    f"delete invalid target regex {spec.target_ref.expr!r}: {exc}"
                ) from exc
        elif isinstance(spec.target_ref.expr, list):
            try:
                matches = sorted(
                    name
                    for name in sd.keys()
                    if _MATCHER.match(spec.target_ref.expr, name) is not None
                )
            except StructuredPathError as exc:
                raise DeleteTransformError(
                    f"delete invalid structured target pattern: {exc}"
                ) from exc
        else:
            raise DeleteTransformError(
                f"delete target expression has invalid type: "
                f"{type(spec.target_ref.expr).__name__}"
            )

        if not matches:
            raise DeleteTransformError(
                f"delete matched zero tensors: {format_target_ref(spec.target_ref)}"
            )

        return matches

    def apply_to_target(self, spec: UnarySpec, name: str, provider: StateDictProvider) -> None:
        model = must_model(spec.target_ref)
        sd = provider.get_state_dict(model)

        if name not in sd:
            raise DeleteTransformError(f"delete target disappeared during apply: {model}::{name}")

        del sd[name]


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


register_transform(DeleteTransform())
