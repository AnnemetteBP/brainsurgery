from __future__ import annotations

from typing import Any

from omegaconf import OmegaConf


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


def parse_transform_block(block: str) -> list[dict[str, Any]]:
    try:
        # Keep `${name}` intact so structured-path destination templates can be
        # interpreted by the matcher rather than OmegaConf interpolation.
        loaded = OmegaConf.to_container(OmegaConf.create(block), resolve=False)
    except Exception as exc:
        raise ValueError(f"invalid YAML: {exc}") from exc

    return normalize_transform_specs(loaded)
