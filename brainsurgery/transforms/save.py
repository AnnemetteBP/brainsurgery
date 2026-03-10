from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..model import (
    parse_shard_size,
    persist_state_dict,
    save_tensor_to_path,
)
from ..provider_utils import resolve_single_model_alias
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


class SaveTransformError(TransformError):
    pass


@dataclass(frozen=True)
class SaveSpec:
    path: Path
    alias: str | None
    tensor_name: str | None
    format: str | None
    shard_size: int | None

    def collect_models(self) -> set[str]:
        # alias can be omitted and resolved at runtime when only one model exists.
        return {self.alias} if self.alias is not None else set()


class SaveTransform(TypedTransform[SaveSpec]):
    name = "save"
    error_type = SaveTransformError
    spec_type = SaveSpec
    allowed_keys = {"path", "alias", "target", "format", "shard"}
    required_keys = {"path"}
    help_text = (
        "Saves either a full state_dict or a single tensor to disk.\n"
        "\n"
        "Without 'target', saves an alias state_dict (default format: safetensors). "
        "Set 'shard' (for example '500MB') to write sharded safetensors in parallel.\n"
        "With 'target', saves one tensor from that alias.\n"
        "\n"
        "Examples:\n"
        "  save: { path: /tmp/out.safetensors }\n"
        "  save: { path: /tmp/out_dir, shard: 500MB }\n"
        "  save: { path: /tmp/a.pt, alias: a, format: torch }\n"
        "  save: { path: /tmp/emb.npy, target: model::embed.weight, format: numpy }"
    )

    def completion_reference_keys(self) -> list[str]:
        return ["target"]

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
        shard_size = _parse_save_shard(payload.get("shard"))

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
            if payload.get("shard") is not None:
                raise SaveTransformError("save.shard is only supported for state_dict save")
            if fmt is not None and fmt not in {"safetensors", "torch", "numpy"}:
                raise SaveTransformError(
                    "save.format for tensor save must be one of: safetensors, torch, numpy"
                )
        else:
            if fmt is not None and fmt not in {"safetensors", "torch"}:
                raise SaveTransformError(
                    "save.format for state_dict save must be one of: safetensors, torch"
                )
            if shard_size is not None and fmt == "torch":
                raise SaveTransformError("save.shard is only supported for safetensors state_dict save")

        return SaveSpec(path=path, alias=alias, tensor_name=tensor_name, format=fmt, shard_size=shard_size)

    def apply(self, spec: object, provider: StateDictProvider) -> TransformResult:
        typed = self.require_spec(spec)
        alias = typed.alias or resolve_single_model_alias(
            provider,
            error_type=SaveTransformError,
            op_name=self.name,
        )

        if typed.tensor_name is None:
            state_dict = provider.get_state_dict(alias)
            format_name = typed.format or "safetensors"
            try:
                persist_state_dict(
                    dict(state_dict.items()),
                    output_path=typed.path,
                    output_format=format_name,  # type: ignore[arg-type]
                    shard_size=typed.shard_size,
                    sharded_output_root=typed.path,
                    max_io_workers=_resolve_max_io_workers(provider),
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


def _resolve_max_io_workers(provider: StateDictProvider) -> int:
    value = getattr(provider, "max_io_workers", None)
    if isinstance(value, int) and value > 0:
        return value
    return 1


def _parse_save_shard(raw: object) -> int | None:
    if raw is None:
        return None
    if not isinstance(raw, str) or not raw:
        raise SaveTransformError("save.shard must be a non-empty string or 'none'")
    try:
        return parse_shard_size(raw)
    except RuntimeError as exc:
        message = str(exc).replace("output.shard", "save.shard")
        raise SaveTransformError(message) from exc


def _unit_test_save_compile_defaults_to_default_model() -> None:
    spec = SaveTransform().compile({"path": "/tmp/x.safetensors"}, default_model="model")
    assert spec.alias == "model"
    assert spec.format is None
    assert spec.tensor_name is None
    assert spec.shard_size is None


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


def _unit_test_save_compile_rejects_shard_for_tensor_save() -> None:
    try:
        SaveTransform().compile(
            {"path": "/tmp/x.safetensors", "target": "model::x", "shard": "1MB"},
            default_model="model",
        )
    except SaveTransformError as exc:
        assert "state_dict save" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected save.shard validation error")


__unit_tests__ = [
    _unit_test_save_compile_defaults_to_default_model,
    _unit_test_save_compile_rejects_tensor_format_for_state_dict,
    _unit_test_save_compile_rejects_alias_conflict,
    _unit_test_save_compile_rejects_shard_for_tensor_save,
]


register_transform(SaveTransform())
