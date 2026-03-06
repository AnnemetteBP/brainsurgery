from __future__ import annotations

from dataclasses import dataclass
from typing import List

from ..transform import (
    BaseTransform,
    StateDictProvider,
    TensorRef,
    TransformError,
    TransformResult,
    ensure_mapping_payload,
    must_model,
    parse_model_expr,
    parse_slice,
    register_transform,
    require_nonempty_string,
    require_numeric,
    select_tensor,
    validate_payload_keys,
)
import re


class ScaleTransformError(TransformError):
    pass


@dataclass(frozen=True)
class ScaleSpec:
    target_ref: TensorRef
    factor: float


class ScaleTransform(BaseTransform):
    name = "scale"

    def compile(self, payload: dict, default_model: str | None) -> ScaleSpec:
        payload = ensure_mapping_payload(payload, self.name)
        validate_payload_keys(
            payload,
            op_name=self.name,
            allowed_keys={"target", "by"},
            required_keys={"target", "by"},
        )

        raw_target = require_nonempty_string(payload, op_name=self.name, key="target")
        factor = require_numeric(payload, op_name=self.name, key="by")

        target_ref = parse_model_expr(raw_target, default_model=default_model)
        if target_ref.slice_spec is not None:
            parse_slice(target_ref.slice_spec)

        assert target_ref.model is not None
        return ScaleSpec(target_ref=target_ref, factor=factor)

    def apply(self, spec: object, provider: StateDictProvider) -> TransformResult:
        if not isinstance(spec, ScaleSpec):
            raise ScaleTransformError(f"scale received wrong spec type: {type(spec).__name__}")

        targets = resolve_scale_targets(spec, provider)
        apply_scale_targets(spec, targets, provider)
        return TransformResult(name=self.name, count=len(targets))

    def infer_output_model(self, spec: object) -> str:
        if not isinstance(spec, ScaleSpec):
            raise ScaleTransformError(f"scale received wrong spec type: {type(spec).__name__}")

        model = spec.target_ref.model
        if model is None:
            raise ScaleTransformError("scale output model missing")
        return model


def resolve_scale_targets(spec: ScaleSpec, provider: StateDictProvider) -> List[str]:
    model = must_model(spec.target_ref)
    sd = provider.get_state_dict(model)

    matches = sorted(name for name in sd.keys() if re.fullmatch(spec.target_ref.expr, name))
    if not matches:
        raise ScaleTransformError(f"scale matched zero tensors: {model}::{spec.target_ref.expr}")

    return matches


def apply_scale_targets(
    spec: ScaleSpec,
    targets: List[str],
    provider: StateDictProvider,
) -> None:
    model = must_model(spec.target_ref)
    sd = provider.get_state_dict(model)
    slice_spec = parse_slice(spec.target_ref.slice_spec) if spec.target_ref.slice_spec else None

    for name in targets:
        tensor = sd[name]
        view = select_tensor(tensor, slice_spec)
        view.mul_(spec.factor)


register_transform(ScaleTransform())
