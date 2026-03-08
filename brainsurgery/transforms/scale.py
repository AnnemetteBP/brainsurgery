from __future__ import annotations

from dataclasses import dataclass

from .unary import UnarySpec, UnaryTransform, resolve_target_names
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
        return resolve_target_names(
            target_ref=spec.target_ref,
            provider=provider,
            op_name=self.name,
            error_type=ScaleTransformError,
        )

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


register_transform(ScaleTransform())
