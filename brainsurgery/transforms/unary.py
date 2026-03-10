from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, Literal, TypeVar

from ..model import tqdm
from ..transform import (
    BaseTransform,
    StateDictProvider,
    TensorRef,
    TransformError,
    TransformResult,
    ensure_mapping_payload,
    format_tensor_ref,
    match_expr_names,
    must_model,
    parse_model_expr,
    parse_slice,
    require_expr,
    validate_payload_keys,
)


@dataclass(frozen=True)
class UnarySpec:
    target_ref: TensorRef

    def collect_models(self) -> set[str]:
        return {must_model(self.target_ref)}


SpecT = TypeVar("SpecT", bound=UnarySpec)


def format_target_ref(ref: TensorRef) -> str:
    return format_tensor_ref(ref)


def resolve_target_names(
    *,
    target_ref: TensorRef,
    provider: StateDictProvider,
    op_name: str,
    error_type: type[TransformError],
) -> list[str]:
    model = must_model(target_ref)
    sd = provider.get_state_dict(model)

    try:
        matches = match_expr_names(
            expr=target_ref.expr,
            names=sd.keys(),
            op_name=op_name,
            role="target",
        )
    except TransformError as exc:
        raise error_type(str(exc)) from exc

    if not matches:
        raise error_type(f"{op_name} matched zero tensors: {format_target_ref(target_ref)}")

    return matches


class UnaryTransform(BaseTransform, ABC, Generic[SpecT]):
    error_type: type[TransformError] = TransformError
    spec_type: type[SpecT]
    progress_desc: str | None = None
    progress_unit: str = "tensor"

    allowed_keys: set[str] = {"target"}
    required_keys: set[str] = {"target"}
    target_key: str = "target"
    slice_policy: Literal["allow", "forbid"] = "forbid"

    def completion_reference_keys(self) -> list[str]:
        return [self.target_key]

    def compile(self, payload: dict, default_model: str | None) -> SpecT:
        payload = ensure_mapping_payload(payload, self.name)
        validate_payload_keys(
            payload,
            op_name=self.name,
            allowed_keys=self.allowed_keys,
            required_keys=self.required_keys,
        )

        raw_target = self.require_target_expr(payload)
        target_ref = parse_model_expr(raw_target, default_model=default_model)

        self.validate_target_ref(target_ref)

        assert target_ref.model is not None
        return self.build_spec(target_ref=target_ref, payload=payload)

    def apply(self, spec: object, provider: StateDictProvider) -> TransformResult:
        typed = self.require_spec(spec)
        targets = self.resolve_targets(typed, provider)

        items = targets
        if self.progress_desc is not None:
            items = tqdm(targets, desc=self.progress_desc, unit=self.progress_unit)

        for name in items:
            self.apply_to_target(typed, name, provider)

        return TransformResult(name=self.name, count=len(targets))

    def infer_output_model(self, spec: object) -> str:
        typed = self.require_spec(spec)
        model = typed.target_ref.model
        if model is None:
            raise self.error_type(f"{self.name} output model missing")
        return model

    def require_spec(self, spec: object) -> SpecT:
        if not isinstance(spec, self.spec_type):
            raise self.error_type(
                f"{self.name} received wrong spec type: {type(spec).__name__}"
            )
        return spec

    def require_target_expr(self, payload: dict) -> str | list[object]:
        return require_expr(payload, op_name=self.name, key=self.target_key)

    def build_spec(self, target_ref: TensorRef, payload: dict) -> SpecT:
        return self.spec_type(target_ref=target_ref)

    def validate_target_ref(self, target_ref: TensorRef) -> None:
        if target_ref.slice_spec is None:
            return

        if self.slice_policy == "forbid":
            raise self.error_type(f"{self.name} target must not be sliced")

        parse_slice(target_ref.slice_spec)

    def resolve_targets(self, spec: SpecT, provider: StateDictProvider) -> list[str]:
        return resolve_target_names(
            target_ref=spec.target_ref,
            provider=provider,
            op_name=self.name,
            error_type=self.error_type,
        )

    @abstractmethod
    def apply_to_target(self, spec: SpecT, name: str, provider: StateDictProvider) -> None:
        ...
