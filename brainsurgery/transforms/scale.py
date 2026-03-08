from __future__ import annotations

from dataclasses import dataclass

from .unary import UnarySpec, UnaryTransform
from ..matching import StructuredPathError, StructuredPathMatcher
from ..transform import (
    StateDictProvider,
    TensorRef,
    TransformError,
    must_model,
    parse_slice,
    register_transform,
    require_numeric,
    select_tensor,
)


class ScaleTransformError(TransformError):
    pass


_MATCHER = StructuredPathMatcher()


@dataclass(frozen=True)
class ScaleSpec(UnarySpec):
    factor: float


class ScaleTransform(UnaryTransform[ScaleSpec]):
    name = "scale"
    error_type = ScaleTransformError
    spec_type = ScaleSpec
    allowed_keys = {"target", "by"}
    required_keys = {"target", "by"}
    progress_desc = "Applying scale transforms"

    def validate_target_ref(self, target_ref: TensorRef) -> None:
        if target_ref.slice_spec is not None:
            parse_slice(target_ref.slice_spec)

    def build_spec(self, target_ref: TensorRef, payload: dict) -> ScaleSpec:
        factor = require_numeric(payload, op_name=self.name, key="by")
        return ScaleSpec(target_ref=target_ref, factor=factor)

    def resolve_targets(self, spec: ScaleSpec, provider: StateDictProvider) -> list[str]:
        model = must_model(spec.target_ref)
        sd = provider.get_state_dict(model)

        if isinstance(spec.target_ref.expr, str):
            import re

            try:
                matches = sorted(
                    name for name in sd.keys() if re.fullmatch(spec.target_ref.expr, name)
                )
            except re.error as exc:
                raise ScaleTransformError(
                    f"scale invalid target regex {spec.target_ref.expr!r}: {exc}"
                ) from exc
        elif isinstance(spec.target_ref.expr, list):
            try:
                matches = sorted(
                    name
                    for name in sd.keys()
                    if _MATCHER.match(spec.target_ref.expr, name) is not None
                )
            except StructuredPathError as exc:
                raise ScaleTransformError(
                    f"scale invalid structured target pattern: {exc}"
                ) from exc
        else:
            raise ScaleTransformError(
                f"scale target expression has invalid type: "
                f"{type(spec.target_ref.expr).__name__}"
            )

        if not matches:
            raise ScaleTransformError(
                f"scale matched zero tensors: {format_target_ref(spec.target_ref)}"
            )

        return matches

    def apply_to_target(self, spec: ScaleSpec, name: str, provider: StateDictProvider) -> None:
        model = must_model(spec.target_ref)
        sd = provider.get_state_dict(model)
        tensor = sd[name]

        slice_spec = (
            parse_slice(spec.target_ref.slice_spec)
            if spec.target_ref.slice_spec is not None
            else None
        )
        view = select_tensor(tensor, slice_spec)
        view.mul_(spec.factor)


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


register_transform(ScaleTransform())
