from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..core import (
    BinaryMappingTransform,
    DestinationPolicy,
    TernaryMappingTransform,
    UnaryTransform,
    get_transform,
    list_transforms,
    match_expr_names,
    parse_model_expr,
)
from ..engine import list_model_aliases, reset_runtime_flags, use_output_emitter


DISABLED_TRANSFORMS = {"dump", "help", "exit"}


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
        }
    return specs


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
    payload: dict[str, Any],
) -> str:
    transform = get_transform(transform_name)
    if transform_name in DISABLED_TRANSFORMS:
        raise ValueError(f"transform {transform_name!r} is disabled in webui2.")

    aliases = sorted(list_model_aliases(provider))
    default_model = aliases[0] if len(aliases) == 1 else None

    reset_runtime_flags()
    lines: list[str] = []
    with use_output_emitter(lines.append):
        spec = transform.compile(payload, default_model=default_model)
        transform.apply(spec, provider)
    return "\n".join(lines)


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
        op_name="webui2.dump",
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


__all__ = [
    "DISABLED_TRANSFORMS",
    "apply_load_transform",
    "apply_transform",
    "default_alias",
    "parse_filter_expr",
    "render_dump_for_alias",
    "require_string",
    "serialize_models",
    "transform_items",
]
