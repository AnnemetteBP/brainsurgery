from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Generic, Iterable, TypeVar

from ..model import tqdm
from ..transform import (
    BaseTransform,
    ResolvedMapping,
    StateDictProvider,
    TensorRef,
    TransformError,
    TransformResult,
    ensure_mapping_payload,
    must_model,
    parse_model_expr,
    require_dest_missing,
    require_dest_present,
    require_expr,
    resolve_name_mappings,
    validate_payload_keys,
)


@dataclass(frozen=True)
class BinaryMappingSpec:
    from_ref: TensorRef
    to_ref: TensorRef

    def collect_models(self) -> set[str]:
        return {must_model(self.from_ref), must_model(self.to_ref)}


SpecT = TypeVar("SpecT", bound=BinaryMappingSpec)


class DestinationPolicy(Enum):
    ANY = "any"
    MUST_EXIST = "must_exist"
    MUST_NOT_EXIST = "must_not_exist"


class BinaryMappingTransform(BaseTransform, ABC, Generic[SpecT]):
    error_type: type[TransformError] = TransformError
    spec_type: type[SpecT]
    progress_desc: str | None = None
    progress_unit: str = "tensor"
    destination_policy: DestinationPolicy = DestinationPolicy.ANY

    allowed_keys = {"from", "to"}
    required_keys = {"from", "to"}

    def compile(self, payload: dict, default_model: str | None) -> SpecT:
        payload = ensure_mapping_payload(payload, self.name)
        validate_payload_keys(
            payload,
            op_name=self.name,
            allowed_keys=self.allowed_keys,
            required_keys=self.required_keys,
        )

        from_ref, to_ref = self.parse_refs(payload, default_model)
        self.validate_refs(from_ref, to_ref)

        assert from_ref.model is not None
        assert to_ref.model is not None
        return self.build_spec(from_ref=from_ref, to_ref=to_ref)

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
    ) -> tuple[TensorRef, TensorRef]:
        raw_from = require_expr(payload, op_name=self.name, key="from")
        raw_to = require_expr(payload, op_name=self.name, key="to")

        from_ref = parse_model_expr(raw_from, default_model=default_model)
        to_ref = parse_model_expr(raw_to, default_model=default_model)
        return from_ref, to_ref

    def build_spec(self, from_ref: TensorRef, to_ref: TensorRef) -> SpecT:
        return self.spec_type(from_ref=from_ref, to_ref=to_ref)

    def iter_mappings(self, mappings: list[ResolvedMapping]) -> Iterable[ResolvedMapping]:
        if self.progress_desc is None:
            return mappings
        return tqdm(mappings, desc=self.progress_desc, unit=self.progress_unit)

    def resolve_mappings(
        self,
        spec: SpecT,
        provider: StateDictProvider,
    ) -> list[ResolvedMapping]:
        return resolve_name_mappings(
            from_ref=spec.from_ref,
            to_ref=spec.to_ref,
            provider=provider,
            op_name=self.name,
        )

    @abstractmethod
    def validate_refs(self, from_ref: TensorRef, to_ref: TensorRef) -> None:
        ...

    def validate_resolved_mappings(
        self,
        mappings: list[ResolvedMapping],
        provider: StateDictProvider,
    ) -> None:
        if self.destination_policy is DestinationPolicy.MUST_EXIST:
            require_dest_present(
                mappings=mappings,
                provider=provider,
                op_name=self.name,
            )
            return

        if self.destination_policy is DestinationPolicy.MUST_NOT_EXIST:
            require_dest_missing(
                mappings=mappings,
                provider=provider,
                op_name=self.name,
            )

    @abstractmethod
    def apply_mapping(self, item: ResolvedMapping, provider: StateDictProvider) -> None:
        ...
