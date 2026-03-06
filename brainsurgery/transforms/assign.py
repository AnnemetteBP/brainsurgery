from __future__ import annotations

from dataclasses import dataclass
from typing import List

from ..transform import (
    BaseTransform,
    ResolvedMapping,
    StateDictProvider,
    TensorRef,
    TransformError,
    TransformResult,
    ensure_mapping_payload,
    parse_model_expr,
    parse_slice,
    register_transform,
    require_dest_present,
    require_nonempty_string,
    resolve_name_mappings,
    select_tensor,
    validate_payload_keys,
)


class AssignTransformError(TransformError):
    pass


@dataclass(frozen=True)
class AssignSpec:
    from_ref: TensorRef
    to_ref: TensorRef


class AssignTransform(BaseTransform):
    name = "assign"

    def compile(self, payload: dict, default_model: str | None) -> AssignSpec:
        payload = ensure_mapping_payload(payload, self.name)
        validate_payload_keys(
            payload,
            op_name=self.name,
            allowed_keys={"from", "to"},
            required_keys={"from", "to"},
        )

        raw_from = require_nonempty_string(payload, op_name=self.name, key="from")
        raw_to = require_nonempty_string(payload, op_name=self.name, key="to")

        from_ref = parse_model_expr(raw_from, default_model=default_model)
        to_ref = parse_model_expr(raw_to, default_model=default_model)

        if from_ref.slice_spec is not None:
            parse_slice(from_ref.slice_spec)
        if to_ref.slice_spec is not None:
            parse_slice(to_ref.slice_spec)

        assert from_ref.model is not None
        assert to_ref.model is not None
        return AssignSpec(from_ref=from_ref, to_ref=to_ref)

    def apply(self, spec: object, provider: StateDictProvider) -> TransformResult:
        if not isinstance(spec, AssignSpec):
            raise AssignTransformError(f"assign received wrong spec type: {type(spec).__name__}")

        mappings = resolve_assign_mappings(spec, provider)
        apply_assign_mappings(mappings, provider)
        return TransformResult(name=self.name, count=len(mappings))

    def infer_output_model(self, spec: object) -> str:
        if not isinstance(spec, AssignSpec):
            raise AssignTransformError(f"assign received wrong spec type: {type(spec).__name__}")

        model = spec.to_ref.model
        if model is None:
            raise AssignTransformError("assign output model missing")
        return model


def resolve_assign_mappings(spec: AssignSpec, provider: StateDictProvider) -> List[ResolvedMapping]:
    mappings = resolve_name_mappings(
        from_ref=spec.from_ref,
        to_ref=spec.to_ref,
        provider=provider,
        op_name="assign",
    )
    require_dest_present(
        mappings=mappings,
        provider=provider,
        op_name="assign",
    )
    return mappings


def apply_assign_mappings(mappings: List[ResolvedMapping], provider: StateDictProvider) -> None:
    for item in mappings:
        src_sd = provider.get_state_dict(item.src_model)
        dst_sd = provider.get_state_dict(item.dst_model)

        src_tensor = src_sd[item.src_name]
        dst_tensor = dst_sd[item.dst_name]

        src_view = select_tensor(src_tensor, item.src_slice)
        dst_view = select_tensor(dst_tensor, item.dst_slice)

        if src_view.shape != dst_view.shape:
            raise AssignTransformError(
                f"shape mismatch assigning {item.src_name} -> {item.dst_name}: "
                f"{tuple(src_view.shape)} != {tuple(dst_view.shape)}"
            )

        dst_view.copy_(src_view)


register_transform(AssignTransform())
