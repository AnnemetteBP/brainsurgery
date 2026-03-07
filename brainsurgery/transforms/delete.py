from __future__ import annotations

from dataclasses import dataclass
from typing import List

from ..matching import StructuredPathError, StructuredPathMatcher
from ..transform import (
    BaseTransform,
    StateDictProvider,
    TensorRef,
    TransformError,
    TransformResult,
    ensure_mapping_payload,
    must_model,
    parse_model_expr,
    register_transform,
    require_expr,
    validate_payload_keys,
)


class DeleteTransformError(TransformError):
    pass


_MATCHER = StructuredPathMatcher()


@dataclass(frozen=True)
class DeleteSpec:
    target_ref: TensorRef


class DeleteTransform(BaseTransform):
    name = "delete"

    def compile(self, payload: dict, default_model: str | None) -> DeleteSpec:
        payload = ensure_mapping_payload(payload, self.name)
        validate_payload_keys(
            payload,
            op_name=self.name,
            allowed_keys={"target"},
            required_keys={"target"},
        )

        raw_target = require_expr(payload, op_name=self.name, key="target")
        target_ref = parse_model_expr(raw_target, default_model=default_model)

        if target_ref.slice_spec is not None:
            raise DeleteTransformError("delete target must not be sliced")

        assert target_ref.model is not None
        return DeleteSpec(target_ref=target_ref)

    def apply(self, spec: object, provider: StateDictProvider) -> TransformResult:
        if not isinstance(spec, DeleteSpec):
            raise DeleteTransformError(f"delete received wrong spec type: {type(spec).__name__}")

        targets = resolve_delete_targets(spec, provider)
        apply_delete_targets(spec, targets, provider)

        return TransformResult(name=self.name, count=len(targets))

    def infer_output_model(self, spec: object) -> str:
        if not isinstance(spec, DeleteSpec):
            raise DeleteTransformError(f"delete received wrong spec type: {type(spec).__name__}")

        model = spec.target_ref.model
        if model is None:
            raise DeleteTransformError("delete output model missing")
        return model


def resolve_delete_targets(spec: DeleteSpec, provider: StateDictProvider) -> List[str]:
    model = must_model(spec.target_ref)
    sd = provider.get_state_dict(model)

    if isinstance(spec.target_ref.expr, str):
        import re

        try:
            matches = sorted(name for name in sd.keys() if re.fullmatch(spec.target_ref.expr, name))
        except re.error as exc:
            raise DeleteTransformError(
                f"delete invalid target regex {spec.target_ref.expr!r}: {exc}"
            ) from exc
    elif isinstance(spec.target_ref.expr, list):
        try:
            matches = sorted(name for name in sd.keys() if _MATCHER.match(spec.target_ref.expr, name) is not None)
        except StructuredPathError as exc:
            raise DeleteTransformError(f"delete invalid structured target pattern: {exc}") from exc
    else:
        raise DeleteTransformError(
            f"delete target expression has invalid type: {type(spec.target_ref.expr).__name__}"
        )

    if not matches:
        raise DeleteTransformError(f"delete matched zero tensors: {format_target_ref(spec.target_ref)}")

    return matches


def apply_delete_targets(
    spec: DeleteSpec,
    targets: List[str],
    provider: StateDictProvider,
) -> None:
    model = must_model(spec.target_ref)
    sd = provider.get_state_dict(model)

    for name in targets:
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
