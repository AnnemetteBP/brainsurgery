from __future__ import annotations

from dataclasses import dataclass

from ..core import BinaryMappingSpec, DestinationPolicy
from ..core import ResolvedMapping, StateDictProvider, TensorRef, TransformError, select_tensor
from ..core import register_transform
from ..core import require_numeric
from ..core import BinaryRefs, DeclarativeBinaryTransform, Docs


@dataclass(frozen=True)
class ClampSpec(BinaryMappingSpec):
    min_value: float | None
    max_value: float | None


def _build_clamp_spec(
    from_ref: TensorRef, to_ref: TensorRef, payload: dict
) -> ClampSpec:
    min_value, max_value = _parse_bounds(payload, "clamp", TransformError)
    return ClampSpec(
        from_ref=from_ref, to_ref=to_ref, min_value=min_value, max_value=max_value
    )


def _clamp_apply(
    spec: ClampSpec, item: ResolvedMapping, provider: StateDictProvider
) -> None:
    src_sd = provider.get_state_dict(item.src_model)
    dst_sd = provider.get_state_dict(item.dst_model)
    src_view = select_tensor(src_sd[item.src_name], item.src_slice)
    dst_sd[item.dst_name] = src_view.clamp(
        min=spec.min_value, max=spec.max_value
    ).clone()


class ClampTransform(DeclarativeBinaryTransform[ClampSpec]):
    name = "clamp"
    error_type = TransformError
    spec_type = ClampSpec
    destination_policy = DestinationPolicy.MUST_NOT_EXIST
    allowed_keys = {"from", "to", "min", "max"}
    required_keys = {"from", "to"}
    docs = Docs(
        "Clamps source tensors into new destination tensors.",
        notes=("At least one of 'min' or 'max' is required.",),
        examples=("clamp: { from: x, to: x_clamped, min: -1.0, max: 1.0 }",),
    )
    refs = BinaryRefs(from_slice=True)
    spec_builder = staticmethod(_build_clamp_spec)
    apply_fn = staticmethod(_clamp_apply)


def _parse_bounds(
    payload: dict,
    op_name: str,
    error_type: type[TransformError],
) -> tuple[float | None, float | None]:
    min_value = (
        require_numeric(payload, op_name=op_name, key="min")
        if "min" in payload
        else None
    )
    max_value = (
        require_numeric(payload, op_name=op_name, key="max")
        if "max" in payload
        else None
    )
    if min_value is None and max_value is None:
        raise error_type(f"{op_name} requires at least one of: min, max")
    if min_value is not None and max_value is not None and min_value > max_value:
        raise error_type(f"{op_name}.min must be <= {op_name}.max")
    return min_value, max_value


register_transform(ClampTransform())
