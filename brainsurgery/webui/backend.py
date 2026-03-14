from __future__ import annotations

import json
from dataclasses import is_dataclass
from pathlib import Path
from typing import Any
from typing import get_args, get_origin, get_type_hints

from omegaconf import OmegaConf
from ..core import (
    BinaryMappingTransform,
    DestinationPolicy,
    TernaryMappingTransform,
    UnaryTransform,
    get_assert_expr_help,
    get_transform,
    get_assert_expr_names,
    list_transforms,
    match_expr_names,
    parse_model_expr,
)
from ..engine import list_model_aliases, reset_runtime_flags, use_output_emitter
from ..engine import get_runtime_flags


DISABLED_TRANSFORMS: set[str] = set()


def transform_items() -> list[dict[str, Any]]:
    specs = _transform_specs()
    items: list[dict[str, Any]] = []
    for name in list_transforms():
        if name in DISABLED_TRANSFORMS:
            continue
        spec = specs.get(name)
        items.append(
            {
                "name": name,
                "enabled": bool(spec and spec["enabled"]),
                "binary": bool(spec),
                "kind": spec["kind"] if spec else "other",
                "allowed_keys": spec["allowed_keys"] if spec else [],
                "required_keys": spec["required_keys"] if spec else [],
                "reference_keys": spec["reference_keys"] if spec else [],
                "to_must_exist": bool(spec and spec["to_must_exist"]),
                "help_commands": spec["help_commands"] if spec else [],
                "help_subcommands": spec["help_subcommands"] if spec else {},
                "assert_expressions": spec["assert_expressions"] if spec else [],
                "assert_expression_meta": spec["assert_expression_meta"] if spec else {},
                "boolean_keys": spec["boolean_keys"] if spec else [],
            }
        )
    return items


def _transform_specs() -> dict[str, dict[str, Any]]:
    specs: dict[str, dict[str, Any]] = {}
    for name in list_transforms():
        transform = get_transform(name)
        allowed = sorted(getattr(transform, "allowed_keys", set()) or set())
        required = sorted(getattr(transform, "required_keys", set()) or set())
        ref_keys = [key for key in transform.completion_reference_keys() if isinstance(key, str)]
        if allowed:
            ref_keys = [key for key in ref_keys if key in set(allowed)]
        if isinstance(transform, BinaryMappingTransform):
            kind = "binary"
        elif isinstance(transform, UnaryTransform):
            kind = "unary"
        elif isinstance(transform, TernaryMappingTransform):
            kind = "ternary"
        else:
            kind = "other"
        destination_policy = getattr(transform, "destination_policy", DestinationPolicy.ANY)
        specs[name] = {
            "enabled": name not in DISABLED_TRANSFORMS,
            "kind": kind,
            "allowed_keys": allowed,
            "required_keys": required,
            "reference_keys": ref_keys,
            "to_must_exist": destination_policy is DestinationPolicy.MUST_EXIST,
            "help_commands": sorted(list_transforms()) if name == "help" else [],
            "help_subcommands": {"assert": sorted(get_assert_expr_names())} if name == "help" else {},
            "assert_expressions": sorted(get_assert_expr_names()) if name == "assert" else [],
            "assert_expression_meta": _assert_expression_meta() if name == "assert" else {},
            "boolean_keys": _boolean_keys_for_transform(transform, allowed),
        }
    return specs


def _is_boolean_annotation(annotation: Any) -> bool:
    if annotation is bool:
        return True
    origin = get_origin(annotation)
    if origin is None:
        return False
    args = [arg for arg in get_args(annotation) if arg is not type(None)]
    return bool(args) and all(arg is bool for arg in args)


def _boolean_keys_for_transform(transform: Any, allowed_keys: list[str]) -> list[str]:
    out: set[str] = set()
    allowed = set(allowed_keys)

    spec_type = getattr(transform, "spec_type", None)
    if spec_type is not None and is_dataclass(spec_type):
        try:
            type_hints = get_type_hints(spec_type)
        except Exception:
            type_hints = {}
        for field_name, annotation in type_hints.items():
            if _is_boolean_annotation(annotation):
                key = field_name.replace("_", "-")
                if key in allowed:
                    out.add(key)

    help_text = str(getattr(transform, "help_text", "") or "")
    for line in help_text.splitlines():
        text = line.strip()
        if not text.startswith("-") or "(boolean)" not in text.lower():
            continue
        key = text[1:].split("(", 1)[0].strip()
        if key in allowed:
            out.add(key)

    return sorted(out)


def _assert_expression_meta() -> dict[str, dict[str, Any]]:
    meta: dict[str, dict[str, Any]] = {}
    for expr_name in sorted(get_assert_expr_names()):
        expr_help = get_assert_expr_help(expr_name)
        assert not isinstance(expr_help, dict)
        meta[expr_name] = {
            "payload_kind": expr_help.payload_kind,
            "allowed_keys": sorted(expr_help.allowed_keys or set()),
            "required_keys": sorted(expr_help.required_keys or set()),
            "description": expr_help.description or "",
        }
    return meta


def require_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string.")
    return value


def default_alias(provider: Any) -> str:
    aliases = set(list_model_aliases(provider))
    base = "model"
    if base not in aliases:
        return base
    index = 2
    while True:
        candidate = f"{base}_{index}"
        if candidate not in aliases:
            return candidate
        index += 1


def parse_filter_expr(raw: Any, *, alias: str) -> str | list[Any]:
    source: Any
    if raw is None:
        source = ".*"
    elif isinstance(raw, str):
        text = raw.strip()
        if not text:
            source = ".*"
        elif text.startswith("["):
            parsed = json.loads(text)
            source = parsed
        else:
            source = text
    elif isinstance(raw, list):
        source = raw
    else:
        raise ValueError("filter must be a string or JSON list.")

    ref = parse_model_expr(source, default_model=alias)
    return ref.expr


def apply_transform(
    *,
    provider: Any,
    transform_name: str,
    payload: Any,
    default_model: str | None = None,
) -> tuple[str, str | None]:
    transform = get_transform(transform_name)
    if transform_name in DISABLED_TRANSFORMS:
        raise ValueError(f"transform {transform_name!r} is disabled in webui.")

    if transform_name == "assert" and isinstance(payload, str):
        text = payload.strip()
        if not text:
            raise ValueError("assert payload cannot be empty.")
        try:
            parsed = OmegaConf.create(text)
            payload = OmegaConf.to_container(parsed, resolve=True)
        except Exception as exc:
            raise ValueError(f"invalid assert YAML payload: {exc}") from exc

    if default_model is None:
        aliases = sorted(list_model_aliases(provider))
        default_model = aliases[0] if len(aliases) == 1 else None

    reset_runtime_flags()
    lines: list[str] = []
    with use_output_emitter(lines.append):
        spec = transform.compile(payload, default_model=default_model)
        transform.apply(spec, provider)
    next_default_model = default_model
    try:
        if transform.contributes_output_model(spec):
            next_default_model = transform.infer_output_model(spec)
    except Exception:
        next_default_model = default_model
    return "\n".join(lines), next_default_model


def apply_load_transform(*, provider: Any, path: Path, alias: str) -> None:
    reset_runtime_flags()
    transform = get_transform("load")
    spec = transform.compile(
        {
            "path": str(path),
            "alias": alias,
        },
        default_model=None,
    )
    transform.apply(spec, provider)


def serialize_runtime_flags() -> dict[str, bool]:
    flags = get_runtime_flags()
    return {
        "dry_run": bool(flags.dry_run),
        "verbose": bool(flags.verbose),
    }


def render_dump_for_alias(
    *,
    provider: Any,
    alias: str,
    format_name: str,
    verbosity: str,
    target: str | list[Any],
) -> tuple[str, int, int]:
    names = provider.get_state_dict(alias).keys()
    matched = match_expr_names(
        expr=target,
        names=names,
        op_name="webui.dump",
        role="target",
    )
    total_count = len(list(names))
    matched_count = len(matched)
    if not matched:
        return "(no tensors matched filter)", 0, total_count

    dump_transform = get_transform("dump")
    spec = dump_transform.compile(
        {
            "target": target,
            "format": format_name,
            "verbosity": verbosity,
        },
        default_model=alias,
    )
    lines: list[str] = []
    with use_output_emitter(lines.append):
        dump_transform.apply(spec, provider)
    return "\n".join(lines), matched_count, total_count


def serialize_models(provider: Any) -> list[dict[str, Any]]:
    models: list[dict[str, Any]] = []
    for alias in sorted(list_model_aliases(provider)):
        state_dict = provider.get_state_dict(alias)
        dump_compact, matched_count, total_count = render_dump_for_alias(
            provider=provider,
            alias=alias,
            format_name="compact",
            verbosity="shape",
            target=".*",
        )
        dump_tree, _, _ = render_dump_for_alias(
            provider=provider,
            alias=alias,
            format_name="tree",
            verbosity="shape",
            target=".*",
        )
        models.append(
            {
                "alias": alias,
                "tensor_count": len(state_dict),
                "matched_count": matched_count,
                "total_count": total_count,
                "dump_compact": dump_compact,
                "dump_tree": dump_tree,
            }
        )
    return models


def render_execution_summary(*, provider: Any, executed_transforms: list[dict[str, Any]]) -> str:
    del provider
    summary_doc = {
        "transforms": [_normalize_summary_node(item) for item in executed_transforms],
    }
    return OmegaConf.to_yaml(summary_doc)


def _normalize_summary_node(value: Any) -> Any:
    if value is None:
        return {}
    if isinstance(value, dict):
        return {key: _normalize_summary_node(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize_summary_node(item) for item in value]
    return value


__all__ = [
    "DISABLED_TRANSFORMS",
    "apply_load_transform",
    "apply_transform",
    "default_alias",
    "parse_filter_expr",
    "render_dump_for_alias",
    "render_execution_summary",
    "require_string",
    "serialize_models",
    "transform_items",
]
