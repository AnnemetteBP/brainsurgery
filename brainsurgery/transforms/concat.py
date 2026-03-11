from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from ..core import match_expr_names
from ..core import TensorRef, must_model, parse_model_expr, parse_slice, select_tensor
from ..core import (
    BaseTransform,
    TransformError,
    TransformResult,
    ensure_mapping_payload,
    register_transform,
    validate_payload_keys,
)
from ..core import StateDictProvider


class ConcatTransformError(TransformError):
    pass


@dataclass(frozen=True)
class ConcatSpec:
    from_refs: list[TensorRef]
    to_ref: TensorRef
    dim: int

    def collect_models(self) -> set[str]:
        models = {must_model(self.to_ref)}
        models.update(must_model(ref) for ref in self.from_refs)
        return models


class ConcatTransform(BaseTransform):
    name = "concat"
    error_type = ConcatTransformError
    spec_type = ConcatSpec
    allowed_keys = {"from", "to", "dim"}
    required_keys = {"from", "to"}
    help_text = (
        "Concatenates multiple source tensors into one destination tensor.\n"
        "\n"
        "'from' must be a list of at least two references. Each source reference must resolve\n"
        "to exactly one tensor. 'to' must be a single destination tensor reference.\n"
        "\n"
        "Examples:\n"
        "  concat: { from: [a::x, a::y], to: a::xy, dim: 0 }\n"
        "  concat: { from: ['a::x::[:, :4]', 'a::x::[:, 4:]'], to: a::x_rebuilt, dim: 1 }"
    )

    def completion_reference_keys(self) -> list[str]:
        return ["from", "to"]

    def compile(self, payload: Any, default_model: str | None) -> ConcatSpec:
        payload = ensure_mapping_payload(payload, self.name)
        validate_payload_keys(
            payload,
            op_name=self.name,
            allowed_keys=self.allowed_keys,
            required_keys=self.required_keys,
        )

        raw_from = payload.get("from")
        if not isinstance(raw_from, list) or len(raw_from) < 2:
            raise ConcatTransformError("concat.from must be a list of at least two references")

        from_refs: list[TensorRef] = []
        for idx, item in enumerate(raw_from):
            ref = parse_model_expr(item, default_model=default_model)
            if ref.slice_spec is not None:
                parse_slice(ref.slice_spec)
            if ref.model is None:
                raise ConcatTransformError(f"concat.from[{idx}] missing model alias")
            from_refs.append(ref)

        to_ref = parse_model_expr(payload.get("to"), default_model=default_model)
        if to_ref.model is None:
            raise ConcatTransformError("concat.to missing model alias")
        if to_ref.slice_spec is not None:
            raise ConcatTransformError("concat.to must not be sliced")
        if not isinstance(to_ref.expr, str):
            raise ConcatTransformError("concat.to must resolve to a single tensor name")

        raw_dim = payload.get("dim", 0)
        if not isinstance(raw_dim, int):
            raise ConcatTransformError("concat.dim must be an integer")

        return ConcatSpec(from_refs=from_refs, to_ref=to_ref, dim=raw_dim)

    def apply(self, spec: object, provider: StateDictProvider) -> TransformResult:
        typed = self.require_spec(spec)
        dst_model = must_model(typed.to_ref)
        assert isinstance(typed.to_ref.expr, str)
        dst_name = typed.to_ref.expr
        dst_sd = provider.get_state_dict(dst_model)
        if dst_name in dst_sd:
            raise ConcatTransformError(f"concat destination already exists: {dst_model}::{dst_name}")

        source_tensors = [self._resolve_source_tensor(ref, provider) for ref in typed.from_refs]
        self._validate_sources(source_tensors, dim=typed.dim)
        rank = source_tensors[0].dim()
        cat_dim = typed.dim if typed.dim >= 0 else typed.dim + rank

        dst_sd[dst_name] = torch.cat(source_tensors, dim=cat_dim).clone()
        return TransformResult(name=self.name, count=1)

    def infer_output_model(self, spec: object) -> str:
        typed = self.require_spec(spec)
        return must_model(typed.to_ref)

    def require_spec(self, spec: object) -> ConcatSpec:
        if not isinstance(spec, ConcatSpec):
            raise ConcatTransformError(
                f"concat received wrong spec type: {type(spec).__name__}"
            )
        return spec

    def _resolve_source_tensor(self, ref: TensorRef, provider: StateDictProvider) -> torch.Tensor:
        src_model = must_model(ref)
        src_sd = provider.get_state_dict(src_model)
        matches = match_expr_names(
            expr=ref.expr,
            names=src_sd.keys(),
            op_name=self.name,
            role="source",
        )
        if not matches:
            raise ConcatTransformError(f"concat source matched zero tensors: {src_model}::{ref.expr}")
        if len(matches) != 1:
            raise ConcatTransformError(
                f"concat source must match exactly one tensor, got {len(matches)}: {src_model}::{ref.expr}"
            )
        src_name = matches[0]
        src_slice = parse_slice(ref.slice_spec) if ref.slice_spec is not None else None
        return select_tensor(src_sd[src_name], src_slice)

    def _validate_sources(self, tensors: list[torch.Tensor], *, dim: int) -> None:
        if not tensors:
            raise ConcatTransformError("concat requires at least one source tensor")

        first = tensors[0]
        rank = first.dim()
        cat_dim = dim if dim >= 0 else dim + rank
        if cat_dim < 0 or cat_dim >= rank:
            raise ConcatTransformError(
                f"concat.dim {dim} out of range for rank {rank} tensor"
            )

        for idx, tensor in enumerate(tensors[1:], start=1):
            if tensor.dim() != rank:
                raise ConcatTransformError(
                    f"concat source rank mismatch at index {idx}: {tensor.dim()} != {rank}"
                )
            if tensor.dtype != first.dtype:
                raise ConcatTransformError(
                    f"concat source dtype mismatch at index {idx}: {tensor.dtype} != {first.dtype}"
                )
            if tensor.device != first.device:
                raise ConcatTransformError(
                    f"concat source device mismatch at index {idx}: {tensor.device} != {first.device}"
                )
            for axis in range(rank):
                if axis == cat_dim:
                    continue
                if tensor.shape[axis] != first.shape[axis]:
                    raise ConcatTransformError(
                        "concat source shape mismatch outside concat dimension "
                        f"at index {idx}, axis {axis}: {tensor.shape[axis]} != {first.shape[axis]}"
                    )








register_transform(ConcatTransform())
