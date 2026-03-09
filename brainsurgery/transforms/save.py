from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..model import save_state_dict_to_path, save_tensor_to_path
from ..providers import BaseStateDictProvider
from ..transform import (
    StateDictProvider,
    TypedTransform,
    TransformError,
    TransformResult,
    ensure_mapping_payload,
    parse_model_expr,
    register_transform,
    require_nonempty_string,
    validate_payload_keys,
)


class SaveTransformError(TransformError):
    pass


@dataclass(frozen=True)
class SaveSpec:
    path: Path
    alias: str | None
    tensor_name: str | None
    format: str | None

    def collect_models(self) -> set[str]:
        # alias can be omitted and resolved at runtime when only one model exists.
        return {self.alias} if self.alias is not None else set()


class SaveTransform(TypedTransform[SaveSpec]):
    name = "save"
    error_type = SaveTransformError
    spec_type = SaveSpec
    allowed_keys = {"path", "alias", "target", "format"}
    required_keys = {"path"}
    help_text = (
        "Saves either a full state_dict or a single tensor to disk.\n"
        "\n"
        "Without 'target', saves an alias state_dict (default format: safetensors). "
        "With 'target', saves one tensor from that alias.\n"
        "\n"
        "Examples:\n"
        "  save: { path: /tmp/out.safetensors }\n"
        "  save: { path: /tmp/a.pt, alias: a, format: torch }\n"
        "  save: { path: /tmp/emb.npy, target: model::embed.weight, format: numpy }"
    )

    def compile(self, payload: Any, default_model: str | None) -> SaveSpec:
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
            raise SaveTransformError("save.alias must be a non-empty string when provided")

        raw_format = payload.get("format")
        if raw_format is not None and (not isinstance(raw_format, str) or not raw_format):
            raise SaveTransformError("save.format must be a non-empty string when provided")
        fmt = raw_format.strip().lower() if isinstance(raw_format, str) else None

        alias: str | None = raw_alias or default_model
        tensor_name: str | None = None
        raw_target = payload.get("target")
        if raw_target is not None:
            if not isinstance(raw_target, str) or not raw_target:
                raise SaveTransformError("save.target must be a non-empty string when provided")
            ref = parse_model_expr(raw_target, default_model=alias)
            if ref.slice_spec is not None:
                raise SaveTransformError("save.target must not be sliced")
            if not isinstance(ref.expr, str):
                raise SaveTransformError("save.target must resolve to a single tensor name")
            if raw_alias is not None and ref.model != raw_alias:
                raise SaveTransformError("save.alias conflicts with model alias in save.target")
            assert ref.model is not None
            alias = ref.model
            tensor_name = ref.expr
            if fmt is not None and fmt not in {"safetensors", "torch", "numpy"}:
                raise SaveTransformError(
                    "save.format for tensor save must be one of: safetensors, torch, numpy"
                )
        else:
            if fmt is not None and fmt not in {"safetensors", "torch"}:
                raise SaveTransformError(
                    "save.format for state_dict save must be one of: safetensors, torch"
                )

        return SaveSpec(path=path, alias=alias, tensor_name=tensor_name, format=fmt)

    def apply(self, spec: object, provider: StateDictProvider) -> TransformResult:
        typed = self.require_spec(spec)
        alias = typed.alias or _resolve_single_alias(provider)

        if typed.tensor_name is None:
            state_dict = provider.get_state_dict(alias)
            format_name = typed.format or "safetensors"
            try:
                save_state_dict_to_path(
                    dict(state_dict.items()),
                    typed.path,
                    format=format_name,  # type: ignore[arg-type]
                )
            except RuntimeError as exc:
                raise SaveTransformError(str(exc)) from exc
            return TransformResult(name=self.name, count=len(state_dict))

        state_dict = provider.get_state_dict(alias)
        if typed.tensor_name not in state_dict:
            raise SaveTransformError(f"save target missing: {alias}::{typed.tensor_name}")

        format_name = typed.format or "safetensors"
        try:
            save_tensor_to_path(
                typed.tensor_name,
                state_dict[typed.tensor_name],
                typed.path,
                format=format_name,  # type: ignore[arg-type]
            )
        except RuntimeError as exc:
            raise SaveTransformError(str(exc)) from exc
        return TransformResult(name=self.name, count=1)

    def infer_output_model(self, spec: object) -> str:
        typed = self.require_spec(spec)
        if typed.alias is None:
            raise SaveTransformError("save cannot infer output model without explicit alias")
        return typed.alias

def _resolve_single_alias(provider: StateDictProvider) -> str:
    aliases = _list_aliases(provider)
    if len(aliases) != 1:
        raise SaveTransformError("save.alias is required when more than one model alias is available")
    return next(iter(aliases))


def _list_aliases(provider: StateDictProvider) -> set[str]:
    if isinstance(provider, BaseStateDictProvider):
        return provider.list_model_aliases()
    state_dicts = getattr(provider, "state_dicts", None)
    if isinstance(state_dicts, dict):
        return set(state_dicts.keys())
    shadow_state_dicts = getattr(provider, "_state_dicts", None)
    if isinstance(shadow_state_dicts, dict):
        return set(shadow_state_dicts.keys())
    return set()


def _unit_test_save_compile_defaults_to_default_model() -> None:
    spec = SaveTransform().compile({"path": "/tmp/x.safetensors"}, default_model="model")
    assert spec.alias == "model"
    assert spec.format is None
    assert spec.tensor_name is None


def _unit_test_save_compile_rejects_tensor_format_for_state_dict() -> None:
    try:
        SaveTransform().compile(
            {"path": "/tmp/x.safetensors", "format": "numpy"},
            default_model="model",
        )
    except SaveTransformError as exc:
        assert "state_dict save" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected save.format validation error")


def _unit_test_save_compile_rejects_alias_conflict() -> None:
    try:
        SaveTransform().compile(
            {"path": "/tmp/x.safetensors", "alias": "a", "target": "b::x"},
            default_model=None,
        )
    except SaveTransformError as exc:
        assert "conflicts" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected alias conflict error")


__unit_tests__ = [
    _unit_test_save_compile_defaults_to_default_model,
    _unit_test_save_compile_rejects_tensor_format_for_state_dict,
    _unit_test_save_compile_rejects_alias_conflict,
]


register_transform(SaveTransform())
