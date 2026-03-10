from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


class CompletionConfigError(RuntimeError):
    pass


def _as_string_list(value: Any, *, path: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise CompletionConfigError(f"{path} must be a list of non-empty strings")
    return value


def _as_mapping(value: Any, *, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CompletionConfigError(f"{path} must be a mapping")
    return value


def _validate_transform_rules(transforms: dict[str, Any]) -> dict[str, Any]:
    for transform_name, raw_rules in transforms.items():
        rules = _as_mapping(raw_rules, path=f"transforms.{transform_name}")

        payload_start = rules.get("payload_start")
        if payload_start is not None:
            payload_start = _as_mapping(
                payload_start,
                path=f"transforms.{transform_name}.payload_start",
            )
            empty_value_source = payload_start.get("empty_value_source")
            if empty_value_source is not None and not isinstance(empty_value_source, str):
                raise CompletionConfigError(
                    f"transforms.{transform_name}.payload_start.empty_value_source must be a string"
                )
            open_brace = payload_start.get("open_brace")
            if open_brace is not None and not isinstance(open_brace, str):
                raise CompletionConfigError(
                    f"transforms.{transform_name}.payload_start.open_brace must be a string"
                )

        key_context = rules.get("key_context")
        if key_context is not None:
            key_context = _as_mapping(key_context, path=f"transforms.{transform_name}.key_context")
            keys = key_context.get("keys")
            if keys is not None:
                _as_string_list(keys, path=f"transforms.{transform_name}.key_context.keys")
            dynamic_source = key_context.get("dynamic_source")
            if dynamic_source is not None and not isinstance(dynamic_source, str):
                raise CompletionConfigError(
                    f"transforms.{transform_name}.key_context.dynamic_source must be a string"
                )
            mode_selector = key_context.get("mode_selector")
            if mode_selector is not None:
                mode_selector = _as_mapping(
                    mode_selector,
                    path=f"transforms.{transform_name}.key_context.mode_selector",
                )
                if not isinstance(mode_selector.get("key"), str) or not mode_selector["key"]:
                    raise CompletionConfigError(
                        f"transforms.{transform_name}.key_context.mode_selector.key must be a non-empty string"
                    )
                _as_string_list(
                    mode_selector.get("default", []),
                    path=f"transforms.{transform_name}.key_context.mode_selector.default",
                )
                values = _as_mapping(
                    mode_selector.get("values", {}),
                    path=f"transforms.{transform_name}.key_context.mode_selector.values",
                )
                for mode_name, mode_keys in values.items():
                    _as_string_list(
                        mode_keys,
                        path=(
                            f"transforms.{transform_name}.key_context.mode_selector.values."
                            f"{mode_name}"
                        ),
                    )

        value_context = rules.get("value_context")
        if value_context is not None:
            value_context = _as_mapping(
                value_context, path=f"transforms.{transform_name}.value_context"
            )
            for value_key, raw_value_rule in value_context.items():
                value_rule = _as_mapping(
                    raw_value_rule,
                    path=f"transforms.{transform_name}.value_context.{value_key}",
                )
                source = value_rule.get("source")
                if not isinstance(source, str) or not source:
                    raise CompletionConfigError(
                        f"transforms.{transform_name}.value_context.{value_key}.source must be a non-empty string"
                    )
                values = value_rule.get("values")
                if values is not None:
                    _as_string_list(
                        values,
                        path=f"transforms.{transform_name}.value_context.{value_key}.values",
                    )
                committed_next = value_rule.get("committed_next")
                if committed_next is not None:
                    _as_string_list(
                        committed_next,
                        path=f"transforms.{transform_name}.value_context.{value_key}.committed_next",
                    )

    return transforms


@lru_cache(maxsize=1)
def load_completion_config() -> dict[str, Any]:
    path = Path(__file__).with_name("completion_rules.yaml")

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise CompletionConfigError(f"failed to read completion config: {path}") from exc
    except yaml.YAMLError as exc:
        raise CompletionConfigError(f"failed to parse completion config: {path}") from exc

    config = _as_mapping(raw, path="completion_rules")
    _as_string_list(config.get("reference_value_keys", []), path="reference_value_keys")
    _as_string_list(config.get("reference_key_order", []), path="reference_key_order")

    top_level = _as_mapping(config.get("top_level", {}), path="top_level")
    _as_string_list(
        top_level.get("no_payload_transforms", []),
        path="top_level.no_payload_transforms",
    )

    transforms = _as_mapping(config.get("transforms", {}), path="transforms")
    _validate_transform_rules(transforms)
    return config
