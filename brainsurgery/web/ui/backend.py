import json
import copy
from dataclasses import is_dataclass
from pathlib import Path
from typing import Any
from typing import get_args, get_origin, get_type_hints

from omegaconf import OmegaConf
from brainsurgery.core import (
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
from brainsurgery.engine import (
    SurgeryPlan,
    executed_plan_summary_yaml,
    get_runtime_flags,
    list_model_aliases,
    parse_summary_mode,
    reset_runtime_flags,
    use_output_emitter,
)


DISABLED_TRANSFORMS: set[str] = set()


def _transform_items() -> list[dict[str, Any]]:
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


def _require_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string.")
    return value


def _default_alias(provider: Any) -> str:
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


def _parse_filter_expr(raw: Any, *, alias: str) -> str | list[Any]:
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


def _apply_transform(
    *,
    provider: Any,
    plan: SurgeryPlan,
    transform_name: str,
    payload: Any,
    default_model: str | None = None,
    record_payload: Any | None = None,
) -> tuple[str, str | None]:
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

    raw_transform = {transform_name: copy.deepcopy(payload)}
    recorded_transform = {
        transform_name: copy.deepcopy(payload if record_payload is None else record_payload),
    }

    return _execute_plan_transform(
        provider=provider,
        plan=plan,
        raw_transform=raw_transform,
        recorded_transform=recorded_transform,
        default_model=default_model,
    )


def _apply_load_transform(*, provider: Any, plan: SurgeryPlan, path: Path, alias: str) -> None:
    _execute_plan_transform(
        provider=provider,
        plan=plan,
        raw_transform={"load": {"path": str(path), "alias": alias}},
        recorded_transform={"load": {"path": str(path), "alias": alias}},
        default_model=None,
    )


def _execute_plan_transform(
    *,
    provider: Any,
    plan: SurgeryPlan,
    raw_transform: dict[str, Any],
    recorded_transform: dict[str, Any],
    default_model: str | None,
) -> tuple[str, str | None]:
    reset_runtime_flags()
    step = plan.append_raw_transform(raw_transform)
    lines: list[str] = []
    with use_output_emitter(lines.append):
        plan.compile_pending(
            extra_known_models=set(list_model_aliases(provider)),
            default_model=default_model,
        )
        plan.execute_pending(provider, interactive=False)

    step.raw = recorded_transform

    next_default_model = default_model
    try:
        if step.compiled is not None:
            output_alias = step.compiled.transform._infer_output_model(step.compiled.spec)
            if output_alias:
                next_default_model = output_alias
    except Exception:
        next_default_model = default_model
    return "\n".join(lines), next_default_model


def _serialize_runtime_flags() -> dict[str, bool]:
    flags = get_runtime_flags()
    return {
        "dry_run": bool(flags.dry_run),
        "verbose": bool(flags.verbose),
    }


def _render_dump_for_alias(
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


def _serialize_models(provider: Any) -> list[dict[str, Any]]:
    models: list[dict[str, Any]] = []
    for alias in sorted(list_model_aliases(provider)):
        state_dict = provider.get_state_dict(alias)
        dump_compact, matched_count, total_count = _render_dump_for_alias(
            provider=provider,
            alias=alias,
            format_name="compact",
            verbosity="shape",
            target=".*",
        )
        dump_tree, _, _ = _render_dump_for_alias(
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


def _render_execution_summary(*, plan: SurgeryPlan, mode: str = "raw") -> str:
    summary_mode = parse_summary_mode(mode)
    return executed_plan_summary_yaml(plan, mode=summary_mode)
