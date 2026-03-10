from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..model import load_tensor_from_path
from ..provider_utils import get_or_create_alias_state_dict
from ..providers import BaseStateDictProvider, ProviderError
from ..refs import parse_model_expr
from ..transform import (
    StateDictProvider,
    TypedTransform,
    TransformError,
    TransformResult,
    ensure_mapping_payload,
    register_transform,
    require_nonempty_string,
    validate_payload_keys,
)


class LoadTransformError(TransformError):
    pass


@dataclass(frozen=True)
class LoadSpec:
    path: Path
    alias: str
    tensor_name: str | None
    format: str

    def collect_models(self) -> set[str]:
        # load can introduce a new model alias
        return set()


class LoadTransform(TypedTransform[LoadSpec]):
    name = "load"
    error_type = LoadTransformError
    spec_type = LoadSpec
    allowed_keys = {"path", "alias", "to", "format"}
    required_keys = {"path"}
    help_text = (
        "Loads either a full state_dict or a single tensor from disk.\n"
        "\n"
        "Without 'to', loads a full state_dict into 'alias'. With 'to', loads one tensor "
        "into a tensor name (optionally with alias in 'to', e.g. model::name).\n"
        "\n"
        "Examples:\n"
        "  load: { path: /tmp/a.safetensors, alias: a }\n"
        "  load: { path: /tmp/tensor.npy, to: model::embed.weight }\n"
        "  load: /tmp/model.safetensors"
    )

    def completion_reference_keys(self) -> list[str]:
        return ["to"]

    def compile(self, payload: Any, default_model: str | None) -> LoadSpec:
        if isinstance(payload, str):
            payload = {"path": payload}
        payload = ensure_mapping_payload(payload, self.name)
        validate_payload_keys(
            payload,
            op_name=self.name,
            allowed_keys=self.allowed_keys,
            required_keys=self.required_keys,
        )

        path = Path(require_nonempty_string(payload, op_name=self.name, key="path"))
        raw_alias = payload.get("alias")
        if raw_alias is not None and (not isinstance(raw_alias, str) or not raw_alias):
            raise LoadTransformError("load.alias must be a non-empty string when provided")

        raw_format = payload.get("format", "auto")
        if not isinstance(raw_format, str) or not raw_format:
            raise LoadTransformError("load.format must be a non-empty string when provided")
        fmt = raw_format.strip().lower()
        if fmt not in {"auto", "torch", "safetensors", "numpy"}:
            raise LoadTransformError("load.format must be one of: auto, torch, safetensors, numpy")

        alias_default = raw_alias or default_model or "model"
        tensor_name: str | None = None
        raw_to = payload.get("to")
        if raw_to is not None:
            if not isinstance(raw_to, str) or not raw_to:
                raise LoadTransformError("load.to must be a non-empty string when provided")
            target_ref = parse_model_expr(raw_to, default_model=alias_default)
            if target_ref.slice_spec is not None:
                raise LoadTransformError("load.to must not be sliced")
            if not isinstance(target_ref.expr, str):
                raise LoadTransformError("load.to must resolve to a single tensor name")
            if raw_alias is not None and target_ref.model != raw_alias:
                raise LoadTransformError("load.alias conflicts with model alias in load.to")
            assert target_ref.model is not None
            alias_default = target_ref.model
            tensor_name = target_ref.expr
        elif fmt != "auto":
            raise LoadTransformError("load.format is only supported for tensor loads (with load.to)")

        return LoadSpec(path=path, alias=alias_default, tensor_name=tensor_name, format=fmt)

    def apply(self, spec: object, provider: StateDictProvider) -> TransformResult:
        typed = self.require_spec(spec)

        if typed.tensor_name is None:
            if not isinstance(provider, BaseStateDictProvider):
                raise LoadTransformError("load requires a provider that supports creating new aliases")
            try:
                state_dict = provider.load_alias_from_path(typed.alias, typed.path)
            except ProviderError as exc:
                message = str(exc).replace("model alias", "load alias")
                raise LoadTransformError(message) from exc
            except RuntimeError as exc:
                raise LoadTransformError(str(exc)) from exc
            return TransformResult(name=self.name, count=len(state_dict))

        try:
            tensor = load_tensor_from_path(typed.path, format=typed.format)  # type: ignore[arg-type]
        except RuntimeError as exc:
            raise LoadTransformError(str(exc)) from exc
        state_dict = get_or_create_alias_state_dict(
            provider,
            typed.alias,
            error_type=LoadTransformError,
            op_name=self.name,
        )
        if typed.tensor_name in state_dict:
            raise LoadTransformError(
                f"load destination already exists: {typed.alias}::{typed.tensor_name}"
            )
        state_dict[typed.tensor_name] = tensor
        return TransformResult(name=self.name, count=1)

    def infer_output_model(self, spec: object) -> str:
        return self.require_spec(spec).alias


def _unit_test_load_compile_defaults_alias_to_model_without_context() -> None:
    spec = LoadTransform().compile({"path": "/tmp/x.safetensors"}, default_model=None)
    assert spec.alias == "model"
    assert spec.tensor_name is None


def _unit_test_load_compile_to_conflict_raises() -> None:
    try:
        LoadTransform().compile(
            {"path": "/tmp/t.pt", "alias": "a", "to": "b::x"},
            default_model=None,
        )
    except LoadTransformError as exc:
        assert "conflicts" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected alias conflict error")


def _unit_test_load_rejects_non_auto_format_for_state_dict() -> None:
    try:
        LoadTransform().compile(
            {"path": "/tmp/x.safetensors", "format": "torch"},
            default_model="model",
        )
    except LoadTransformError as exc:
        assert "only supported for tensor loads" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected load.format validation error")


__unit_tests__ = [
    _unit_test_load_compile_defaults_alias_to_model_without_context,
    _unit_test_load_compile_to_conflict_raises,
    _unit_test_load_rejects_non_auto_format_for_state_dict,
]


register_transform(LoadTransform())
