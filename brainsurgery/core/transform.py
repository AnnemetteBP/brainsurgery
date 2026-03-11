from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Dict, Generic, List, TypeVar

from .transform_types import StateDictLike, StateDictProvider, TransformError

if TYPE_CHECKING:
    from ..engine import SurgeryPlan


from .refs import Expr


class TransformControl(Enum):
    CONTINUE = "continue"
    EXIT = "exit"


@dataclass(frozen=True)
class TransformResult:
    name: str
    count: int
    control: TransformControl = TransformControl.CONTINUE
@dataclass(frozen=True)
class CompiledTransform:
    transform: "BaseTransform"
    spec: object


class BaseTransform(ABC):
    name: str
    completion_requires_payload: bool = True

    @abstractmethod
    def compile(self, payload: dict, default_model: str | None) -> object:
        raise NotImplementedError

    @abstractmethod
    def apply(self, spec: object, provider: StateDictProvider) -> TransformResult:
        raise NotImplementedError

    @abstractmethod
    def infer_output_model(self, spec: object) -> str:
        raise NotImplementedError

    def contributes_output_model(self, spec: object) -> bool:
        del spec
        return True

    def completion_reference_keys(self) -> list[str]:
        return []

    def completion_payload_start_candidates(self, prefix_text: str) -> list[str] | None:
        del prefix_text
        return None

    def completion_key_candidates(self, before_cursor: str, prefix_text: str) -> list[str] | None:
        del before_cursor, prefix_text
        return None

    def completion_value_candidates(
        self,
        value_key: str | None,
        prefix_text: str,
        model_aliases: list[str],
    ) -> list[str] | None:
        del value_key, prefix_text, model_aliases
        return None

    def completion_committed_next_candidates(self, value_key: str | None) -> list[str] | None:
        del value_key
        return None


SpecT = TypeVar("SpecT")


class TypedTransform(BaseTransform, ABC, Generic[SpecT]):
    error_type: type[TransformError] = TransformError
    spec_type: type[SpecT]

    def require_spec(self, spec: object) -> SpecT:
        if not isinstance(spec, self.spec_type):
            raise self.error_type(
                f"{self.name} expected {self.spec_type.__name__}, got {type(spec).__name__}"
            )
        return spec


_REGISTRY: Dict[str, BaseTransform] = {}


def register_transform(transform: BaseTransform) -> None:
    name = getattr(transform, "name", None)
    if not isinstance(name, str) or not name:
        raise TransformError("transform must define a non-empty string 'name'")
    if name in _REGISTRY:
        raise TransformError(f"transform already registered: {name}")
    _REGISTRY[name] = transform


def get_transform(name: str) -> BaseTransform:
    try:
        return _REGISTRY[name]
    except KeyError as exc:
        raise TransformError(f"unknown transform: {name}") from exc


def list_transforms() -> List[str]:
    return sorted(_REGISTRY.keys())


def _apply_transform(compiled: CompiledTransform, provider: StateDictProvider) -> TransformResult:
    return compiled.transform.apply(compiled.spec, provider)


def _infer_output_model(
    plan: SurgeryPlan,
    provider: StateDictProvider | None = None,
) -> str:
    destination_models = set()

    for compiled in plan.transforms:
        if not compiled.transform.contributes_output_model(compiled.spec):
            continue
        inferred_model = _infer_transform_output_model(compiled, provider)
        if provider is not None and not _has_any_tensor(provider, inferred_model):
            continue
        destination_models.add(inferred_model)

    if len(destination_models) != 1:
        raise TransformError(
            "cannot infer output model uniquely; expected exactly one destination model across all transforms"
        )

    return next(iter(destination_models))


def _infer_transform_output_model(
    compiled: CompiledTransform,
    provider: StateDictProvider | None,
) -> str:
    try:
        return compiled.transform.infer_output_model(compiled.spec)
    except TransformError:
        if provider is None:
            raise

        collect_models = getattr(compiled.spec, "collect_models", None)
        if not callable(collect_models):
            raise

        models = collect_models()
        if not isinstance(models, set):
            raise

        non_empty_models = [model for model in models if _has_any_tensor(provider, model)]
        if len(non_empty_models) == 1:
            return non_empty_models[0]
        raise


def _has_any_tensor(provider: StateDictProvider, model: str) -> bool:
    try:
        return len(provider.get_state_dict(model)) > 0
    except Exception:
        return False


from .mappings import (
    ResolvedMapping,
    match_expr_names,
    match_structured_expr,
    require_dest_missing,
    require_dest_present,
    resolve_name_mappings,
    rewrite_structured_expr,
)


def ensure_mapping_payload(payload: object, op_name: str) -> dict:
    if not isinstance(payload, dict):
        raise TransformError(f"{op_name} payload must be a mapping")
    return payload


def validate_payload_keys(
    payload: dict,
    *,
    op_name: str,
    allowed_keys: set[str],
    required_keys: set[str] | None = None,
) -> None:
    unknown = set(payload) - allowed_keys
    if unknown:
        raise TransformError(f"{op_name} received unknown keys: {sorted(unknown)}")

    if required_keys is None:
        required_keys = set()

    missing = required_keys - set(payload)
    if missing:
        missing_list = sorted(missing)
        if len(missing_list) == 1:
            raise TransformError(f"{op_name}.{missing_list[0]} is required")
        raise TransformError(f"{op_name} is missing required keys: {missing_list}")


def require_nonempty_string(payload: dict, *, op_name: str, key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise TransformError(f"{op_name}.{key} must be a non-empty string")
    return value


def require_expr(payload: dict, *, op_name: str, key: str) -> Expr:
    value = payload.get(key)

    if isinstance(value, str):
        if not value:
            raise TransformError(
                f"{op_name}.{key} must be a non-empty string or a non-empty list of non-empty strings"
            )
        return value

    if isinstance(value, list):
        if not value:
            raise TransformError(
                f"{op_name}.{key} must be a non-empty string or a non-empty list of non-empty strings"
            )
        if not all(isinstance(item, str) and item for item in value):
            raise TransformError(
                f"{op_name}.{key} must be a non-empty string or a non-empty list of non-empty strings"
            )
        return value

    raise TransformError(
        f"{op_name}.{key} must be a non-empty string or a non-empty list of non-empty strings"
    )


def require_numeric(payload: dict, *, op_name: str, key: str) -> float:
    value = payload.get(key)
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise TransformError(f"{op_name}.{key} must be numeric") from exc


__all__ = [
    "BaseTransform",
    "CompiledTransform",
    "TransformControl",
    "TransformResult",
    "_REGISTRY",
    "_apply_transform",
    "_infer_output_model",
    "TypedTransform",
    "ensure_mapping_payload",
    "get_transform",
    "list_transforms",
    "register_transform",
    "require_expr",
    "require_nonempty_string",
    "require_numeric",
    "validate_payload_keys",
]
