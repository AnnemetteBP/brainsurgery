import logging
from collections.abc import Mapping
from typing import Any


def _normalize_single_transform_spec(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        if len(raw) != 1:
            raise ValueError("each transform spec must be a mapping with exactly one key")
        name, payload = next(iter(raw.items()))
        if payload is None:
            return {name: {}}
        return {name: payload}

    if isinstance(raw, str):
        name = raw.strip()
        if not name:
            raise ValueError("transform name must be a non-empty string")
        return {name: {}}

    raise ValueError(
        "transform spec must be either a YAML mapping or a bare transform name"
    )


def normalize_transform_specs(raw: Any) -> list[dict[str, Any]]:
    if raw is None:
        return []

    if isinstance(raw, list):
        return [_normalize_single_transform_spec(item) for item in raw]

    return [_normalize_single_transform_spec(raw)]


def normalize_raw_plan(raw_plan: Any) -> dict[str, Any]:
    if not isinstance(raw_plan, Mapping):
        raise ValueError("plan must be a mapping")

    planned_raw: dict[str, Any] = {
        "inputs": raw_plan.get("inputs", []),
        "transforms": normalize_transform_specs(raw_plan.get("transforms")),
    }
    if "output" in raw_plan:
        planned_raw["output"] = raw_plan.get("output")
    return planned_raw


def apply_log_level(log_level: str, *, logger_name: str = "brainsurgery") -> None:
    allowed = {"debug", "info", "warning", "error", "critical"}
    level_name = log_level.strip().lower()
    if level_name not in allowed:
        raise ValueError(f"log_level must be one of: {', '.join(sorted(allowed))}")

    level = getattr(logging, level_name.upper())
    logging.getLogger().setLevel(level)
    logging.getLogger(logger_name).setLevel(level)
