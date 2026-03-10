from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Generic, Iterable, TypeVar

import re

from .model import tqdm
from .transform import (
    BaseTransform,
    StateDictProvider,
    TensorRef,
    TransformError,
    TransformResult,
    ensure_mapping_payload,
    format_tensor_ref,
    match_expr_names,
    match_structured_expr,
    must_model,
    parse_model_expr,
    parse_slice,
    require_expr,
    rewrite_structured_expr,
    validate_payload_keys,
)


@dataclass(frozen=True)
class TernaryMappingSpec:
    from_a_ref: TensorRef
    from_b_ref: TensorRef
    to_ref: TensorRef

    def collect_models(self) -> set[str]:
        return {
            must_model(self.from_a_ref),
            must_model(self.from_b_ref),
            must_model(self.to_ref),
        }


@dataclass(frozen=True)
class ResolvedTernaryMapping:
    src_a_model: str
    src_a_name: str
    src_a_slice: tuple[object, ...] | None
    src_b_model: str
    src_b_name: str
    src_b_slice: tuple[object, ...] | None
    dst_model: str
    dst_name: str
    dst_slice: tuple[object, ...] | None


SpecT = TypeVar("SpecT", bound=TernaryMappingSpec)


class DestinationPolicy(Enum):
    ANY = "any"
    MUST_EXIST = "must_exist"
    MUST_NOT_EXIST = "must_not_exist"


class TernaryMappingTransform(BaseTransform, ABC, Generic[SpecT]):
    error_type: type[TransformError] = TransformError
    spec_type: type[SpecT]
    progress_desc: str | None = None
    progress_unit: str = "tensor"
    destination_policy: DestinationPolicy = DestinationPolicy.ANY

    allowed_keys = {"from_a", "from_b", "to"}
    required_keys = {"from_a", "from_b", "to"}

    def completion_reference_keys(self) -> list[str]:
        return ["from_a", "from_b", "to"]

    def compile(self, payload: dict, default_model: str | None) -> SpecT:
        payload = ensure_mapping_payload(payload, self.name)
        validate_payload_keys(
            payload,
            op_name=self.name,
            allowed_keys=self.allowed_keys,
            required_keys=self.required_keys,
        )

        from_a_ref, from_b_ref, to_ref = self.parse_refs(payload, default_model)
        self.validate_refs(from_a_ref, from_b_ref, to_ref)

        assert from_a_ref.model is not None
        assert from_b_ref.model is not None
        assert to_ref.model is not None
        return self.build_spec(from_a_ref=from_a_ref, from_b_ref=from_b_ref, to_ref=to_ref)

    def apply(self, spec: object, provider: StateDictProvider) -> TransformResult:
        typed = self.require_spec(spec)
        mappings = self.resolve_mappings(typed, provider)
        self.validate_resolved_mappings(mappings, provider)

        for item in self.iter_mappings(mappings):
            self.apply_mapping(item, provider)

        return TransformResult(name=self.name, count=len(mappings))

    def infer_output_model(self, spec: object) -> str:
        typed = self.require_spec(spec)
        model = typed.to_ref.model
        if model is None:
            raise self.error_type(f"{self.name} output model missing")
        return model

    def require_spec(self, spec: object) -> SpecT:
        if not isinstance(spec, self.spec_type):
            raise self.error_type(
                f"{self.name} received wrong spec type: {type(spec).__name__}"
            )
        return spec

    def parse_refs(
        self,
        payload: dict,
        default_model: str | None,
    ) -> tuple[TensorRef, TensorRef, TensorRef]:
        raw_from_a = require_expr(payload, op_name=self.name, key="from_a")
        raw_from_b = require_expr(payload, op_name=self.name, key="from_b")
        raw_to = require_expr(payload, op_name=self.name, key="to")

        from_a_ref = parse_model_expr(raw_from_a, default_model=default_model)
        from_b_ref = parse_model_expr(raw_from_b, default_model=default_model)
        to_ref = parse_model_expr(raw_to, default_model=default_model)
        return from_a_ref, from_b_ref, to_ref

    def build_spec(self, from_a_ref: TensorRef, from_b_ref: TensorRef, to_ref: TensorRef) -> SpecT:
        return self.spec_type(from_a_ref=from_a_ref, from_b_ref=from_b_ref, to_ref=to_ref)

    def iter_mappings(self, mappings: list[ResolvedTernaryMapping]) -> Iterable[ResolvedTernaryMapping]:
        if self.progress_desc is None:
            return mappings
        return tqdm(mappings, desc=self.progress_desc, unit=self.progress_unit)

    @abstractmethod
    def validate_refs(self, from_a_ref: TensorRef, from_b_ref: TensorRef, to_ref: TensorRef) -> None:
        ...

    def resolve_mappings(
        self,
        spec: SpecT,
        provider: StateDictProvider,
    ) -> list[ResolvedTernaryMapping]:
        from_a_ref = spec.from_a_ref
        from_b_ref = spec.from_b_ref
        to_ref = spec.to_ref

        a_slice = parse_slice(from_a_ref.slice_spec) if from_a_ref.slice_spec is not None else None
        b_slice = parse_slice(from_b_ref.slice_spec) if from_b_ref.slice_spec is not None else None
        dst_slice = parse_slice(to_ref.slice_spec) if to_ref.slice_spec is not None else None

        src_a_model = must_model(from_a_ref)
        src_b_model = must_model(from_b_ref)
        dst_model = must_model(to_ref)
        src_a_sd = provider.get_state_dict(src_a_model)
        src_b_sd = provider.get_state_dict(src_b_model)

        a_expr = from_a_ref.expr
        b_expr = from_b_ref.expr
        to_expr = to_ref.expr
        if isinstance(a_expr, str) and isinstance(b_expr, str) and isinstance(to_expr, str):
            src_a_names = match_expr_names(
                expr=a_expr,
                names=src_a_sd.keys(),
                op_name=self.name,
                role="source_a",
            )
            if not src_a_names:
                raise self.error_type(
                    f"{self.name} source_a matched zero tensors: "
                    f"{format_tensor_ref(from_a_ref)}; available tensors: {sorted(src_a_sd.keys())}"
                )

            resolved: list[ResolvedTernaryMapping] = []
            dst_names_seen: set[str] = set()
            for src_a_name in src_a_names:
                try:
                    src_b_name = re.sub(a_expr, b_expr, src_a_name)
                    dst_name = re.sub(a_expr, to_expr, src_a_name)
                except re.error as exc:
                    raise self.error_type(
                        f"{self.name} invalid regex rewrite from {a_expr!r}: {exc}"
                    ) from exc

                if src_b_name not in src_b_sd:
                    raise self.error_type(
                        f"{self.name} source_b missing: {src_b_model}::{src_b_name}"
                    )
                if dst_name in dst_names_seen:
                    raise self.error_type(f"{self.name} destination collision: {dst_model}::{dst_name}")
                dst_names_seen.add(dst_name)

                resolved.append(
                    ResolvedTernaryMapping(
                        src_a_model=src_a_model,
                        src_a_name=src_a_name,
                        src_a_slice=a_slice,
                        src_b_model=src_b_model,
                        src_b_name=src_b_name,
                        src_b_slice=b_slice,
                        dst_model=dst_model,
                        dst_name=dst_name,
                        dst_slice=dst_slice,
                    )
                )
            return resolved

        if isinstance(a_expr, list) and isinstance(b_expr, list) and isinstance(to_expr, list):
            resolved: list[ResolvedTernaryMapping] = []
            dst_names_seen: set[str] = set()
            matched_any = False
            for src_a_name in sorted(src_a_sd.keys()):
                match = match_structured_expr(
                    expr=a_expr,
                    name=src_a_name,
                    op_name=self.name,
                    role="source_a",
                )
                if match is None:
                    continue
                matched_any = True

                src_b_name = rewrite_structured_expr(
                    expr=b_expr,
                    match=match,
                    op_name=self.name,
                    role="source_b",
                )
                dst_name = rewrite_structured_expr(
                    expr=to_expr,
                    match=match,
                    op_name=self.name,
                    role="destination",
                )

                if src_b_name not in src_b_sd:
                    raise self.error_type(
                        f"{self.name} source_b missing: {src_b_model}::{src_b_name}"
                    )
                if dst_name in dst_names_seen:
                    raise self.error_type(f"{self.name} destination collision: {dst_model}::{dst_name}")
                dst_names_seen.add(dst_name)

                resolved.append(
                    ResolvedTernaryMapping(
                        src_a_model=src_a_model,
                        src_a_name=src_a_name,
                        src_a_slice=a_slice,
                        src_b_model=src_b_model,
                        src_b_name=src_b_name,
                        src_b_slice=b_slice,
                        dst_model=dst_model,
                        dst_name=dst_name,
                        dst_slice=dst_slice,
                    )
                )

            if not matched_any:
                raise self.error_type(
                    f"{self.name} source_a matched zero tensors: "
                    f"{format_tensor_ref(from_a_ref)}; available tensors: {sorted(src_a_sd.keys())}"
                )
            return resolved

        raise self.error_type(
            f"{self.name} requires from_a/from_b/to expressions of the same kind: "
            "either all strings (regex mode) or all lists (structured mode)"
        )

    def validate_resolved_mappings(
        self,
        mappings: list[ResolvedTernaryMapping],
        provider: StateDictProvider,
    ) -> None:
        if self.destination_policy is DestinationPolicy.ANY:
            return

        for item in mappings:
            dst_sd = provider.get_state_dict(item.dst_model)
            exists = item.dst_name in dst_sd

            if self.destination_policy is DestinationPolicy.MUST_EXIST and not exists:
                raise self.error_type(
                    f"{self.name} destination missing: {item.dst_model}::{item.dst_name}"
                )
            if self.destination_policy is DestinationPolicy.MUST_NOT_EXIST and exists:
                raise self.error_type(
                    f"{self.name} destination already exists: {item.dst_model}::{item.dst_name}"
                )

    @abstractmethod
    def apply_mapping(self, item: ResolvedTernaryMapping, provider: StateDictProvider) -> None:
        ...
