from __future__ import annotations

from dataclasses import dataclass
from typing import List
import re

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
    require_nonempty_string,
    validate_payload_keys,
)


class DeleteTransformError(TransformError):
    pass


@dataclass(frozen=True)
class DeleteSpec:
    target_ref: TensorRef


class DeleteTransform(BaseTransform):
    name = "delete"

    # ---------------- compile ----------------

    def compile(self, payload: dict, default_model: str | None) -> DeleteSpec:
        payload = ensure_mapping_payload(payload, self.name)
        validate_payload_keys(
            payload,
            op_name=self.name,
            allowed_keys={"target"},
            required_keys={"target"},
        )

        raw_target = require_nonempty_string(payload, op_name=self.name, key="target")

        target_ref = parse_model_expr(raw_target, default_model=default_model)

        if target_ref.slice_spec is not None:
            raise DeleteTransformError("delete target must not be sliced")

        assert target_ref.model is not None
        return DeleteSpec(target_ref=target_ref)

    # ---------------- apply ----------------

    def apply(self, spec: object, provider: StateDictProvider) -> TransformResult:
        if not isinstance(spec, DeleteSpec):
            raise DeleteTransformError(f"delete received wrong spec type: {type(spec).__name__}")

        targets = resolve_delete_targets(spec, provider)
        apply_delete_targets(spec, targets, provider)

        return TransformResult(name=self.name, count=len(targets))

    # ---------------- output model ----------------

    def infer_output_model(self, spec: object) -> str:
        if not isinstance(spec, DeleteSpec):
            raise DeleteTransformError(f"delete received wrong spec type: {type(spec).__name__}")

        model = spec.target_ref.model
        if model is None:
            raise DeleteTransformError("delete output model missing")
        return model


# ============================================================
# Resolution + execution
# ============================================================

def resolve_delete_targets(spec: DeleteSpec, provider: StateDictProvider) -> List[str]:
    model = must_model(spec.target_ref)
    sd = provider.get_state_dict(model)

    matches = sorted(name for name in sd.keys() if re.fullmatch(spec.target_ref.expr, name))
    if not matches:
        raise DeleteTransformError(
            f"delete matched zero tensors: {model}::{spec.target_ref.expr}"
        )

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
            raise DeleteTransformError(
                f"delete target disappeared during apply: {model}::{name}"
            )
        del sd[name]


register_transform(DeleteTransform())

