from __future__ import annotations

from dataclasses import dataclass

import torch

from .unary import UnarySpec, UnaryTransform
from ..matching import StructuredPathError, StructuredPathMatcher
from ..transform import (
    StateDictProvider,
    TensorRef,
    TransformError,
    must_model,
    register_transform,
    require_nonempty_string,
)


class CastTransformError(TransformError):
    pass


_MATCHER = StructuredPathMatcher()


@dataclass(frozen=True)
class CastSpec(UnarySpec):
    dtype: torch.dtype


def parse_dtype(raw: str) -> torch.dtype:
    value = raw.strip().lower()

    aliases = {
        "float16": torch.float16,
        "half": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "float": torch.float32,
        "fp32": torch.float32,
        "float64": torch.float64,
        "double": torch.float64,
        "fp64": torch.float64,
        "uint8": torch.uint8,
        "int8": torch.int8,
        "int16": torch.int16,
        "short": torch.int16,
        "int32": torch.int32,
        "int": torch.int32,
        "int64": torch.int64,
        "long": torch.int64,
        "bool": torch.bool,
    }

    try:
        return aliases[value]
    except KeyError as exc:
        allowed = ", ".join(sorted(aliases))
        raise CastTransformError(f"unsupported dtype {raw!r}; expected one of: {allowed}") from exc


class CastTransform(UnaryTransform[CastSpec]):
    name = "cast"
    error_type = CastTransformError
    spec_type = CastSpec
    allowed_keys = {"target", "to"}
    required_keys = {"target", "to"}
    progress_desc = "Applying cast transforms"

    def validate_target_ref(self, target_ref: TensorRef) -> None:
        if target_ref.slice_spec is not None:
            raise CastTransformError("cast does not support tensor slices; cast the whole tensor")

    def build_spec(self, target_ref: TensorRef, payload: dict) -> CastSpec:
        raw_dtype = require_nonempty_string(payload, op_name=self.name, key="to")
        dtype = parse_dtype(raw_dtype)
        return CastSpec(target_ref=target_ref, dtype=dtype)

    def resolve_targets(self, spec: CastSpec, provider: StateDictProvider) -> list[str]:
        model = must_model(spec.target_ref)
        sd = provider.get_state_dict(model)

        if isinstance(spec.target_ref.expr, str):
            import re

            try:
                matches = sorted(
                    name for name in sd.keys() if re.fullmatch(spec.target_ref.expr, name)
                )
            except re.error as exc:
                raise CastTransformError(
                    f"cast invalid target regex {spec.target_ref.expr!r}: {exc}"
                ) from exc
        elif isinstance(spec.target_ref.expr, list):
            try:
                matches = sorted(
                    name
                    for name in sd.keys()
                    if _MATCHER.match(spec.target_ref.expr, name) is not None
                )
            except StructuredPathError as exc:
                raise CastTransformError(f"cast invalid structured target pattern: {exc}") from exc
        else:
            raise CastTransformError(
                f"cast target expression has invalid type: {type(spec.target_ref.expr).__name__}"
            )

        if not matches:
            raise CastTransformError(f"cast matched zero tensors: {format_target_ref(spec.target_ref)}")

        return matches

    def apply_to_target(self, spec: CastSpec, name: str, provider: StateDictProvider) -> None:
        model = must_model(spec.target_ref)
        sd = provider.get_state_dict(model)
        sd[name] = sd[name].to(dtype=spec.dtype)


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


register_transform(CastTransform())
