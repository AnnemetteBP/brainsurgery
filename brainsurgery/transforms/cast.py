from dataclasses import dataclass

import torch

from ..core import (
    BinaryMappingSpec,
    BinaryRefs,
    DeclarativeBinaryTransform,
    DeclarativeUnaryTransform,
    DestinationPolicy,
    Docs,
    StateDictProvider,
    TensorRef,
    TransformError,
    UnaryRefs,
    UnarySpec,
    must_model,
    parse_torch_dtype,
    register_transform,
    require_nonempty_string,
    state_dict_for_ref,
    view_for_ref_name,
)
from ..engine import emit_verbose_binary_activity, emit_verbose_unary_activity


@dataclass(frozen=True)
class CastSpec(BinaryMappingSpec):
    dtype: torch.dtype


def _build_cast_spec(from_ref: TensorRef, to_ref: TensorRef, payload: dict) -> CastSpec:
    raw_dtype = require_nonempty_string(payload, op_name="cast", key="dtype")
    dtype = parse_torch_dtype(
        raw_dtype,
        error_type=TransformError,
        op_name="cast",
        field_name="dtype",
    )
    return CastSpec(from_ref=from_ref, to_ref=to_ref, dtype=dtype)


def _cast_apply(spec: CastSpec, src_name: str, dst_name: str, provider: StateDictProvider) -> None:
    _src_sd, src_view = view_for_ref_name(provider, spec.from_ref, src_name)
    dst_sd = state_dict_for_ref(provider, spec.to_ref)
    dst_sd[dst_name] = src_view.to(dtype=spec.dtype)
    emit_verbose_binary_activity("cast", src_name, dst_name)


class CastTransform(DeclarativeBinaryTransform[CastSpec]):
    name = "cast"
    error_type = TransformError
    spec_type = CastSpec
    destination_policy = DestinationPolicy.MUST_NOT_EXIST
    allowed_keys = {"from", "to", "dtype"}
    required_keys = {"from", "to", "dtype"}
    docs = Docs(
        "Casts source tensors to a different dtype and writes new destination tensors.",
        examples=(
            "cast: { from: ln_f.weight, to: ln_f_fp16.weight, dtype: float16 }",
            "cast: { from: '.*weight', to: 'fp16.\\\\g<0>', dtype: bfloat16 }",
        ),
    )
    refs = BinaryRefs(from_slice=True)
    spec_builder = staticmethod(_build_cast_spec)
    apply_fn = staticmethod(_cast_apply)


register_transform(CastTransform())


@dataclass(frozen=True)
class CastInPlaceSpec(UnarySpec):
    dtype: torch.dtype


def _build_cast_in_place_spec(target_ref: TensorRef, payload: dict) -> CastInPlaceSpec:
    raw_dtype = require_nonempty_string(payload, op_name="cast_", key="to")
    dtype = parse_torch_dtype(
        raw_dtype,
        error_type=TransformError,
        op_name="cast_",
        field_name="to",
    )
    return CastInPlaceSpec(target_ref=target_ref, dtype=dtype)


def _cast_in_place_apply(spec: CastInPlaceSpec, name: str, provider: StateDictProvider) -> None:
    model = must_model(spec.target_ref)
    sd = provider.get_state_dict(model)
    sd[name] = sd[name].to(dtype=spec.dtype)
    emit_verbose_unary_activity("cast_", name)


class CastInPlaceTransform(DeclarativeUnaryTransform[CastInPlaceSpec]):
    name = "cast_"
    error_type = TransformError
    spec_type = CastInPlaceSpec
    allowed_keys = {"target", "to"}
    required_keys = {"target", "to"}
    docs = Docs(
        "Casts one or more tensors to a different dtype in-place.",
        notes=("The entire tensor is cast.",),
        examples=(
            "cast_: { target: ln_f.weight, to: float16 }",
            "cast_: { target: '.*weight', to: bfloat16 }",
        ),
    )
    refs = UnaryRefs()
    spec_builder = staticmethod(_build_cast_in_place_spec)
    apply_fn = staticmethod(_cast_in_place_apply)


register_transform(CastInPlaceTransform())
