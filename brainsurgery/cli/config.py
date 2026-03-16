import re
from pathlib import Path
from typing import Any

import typer
from omegaconf import OmegaConf

_PATH_TOKEN_RE = re.compile(
    r"""
    ([^. \[\]]+)   # plain key
    |
    \[(\d+)\]      # numeric list index
    """,
    re.VERBOSE,
)


def is_yaml_file_arg(token: str) -> bool:
    path = Path(token)
    return path.suffix.lower() in {".yaml", ".yml"} and path.is_file()


def is_yaml_like_arg(token: str) -> bool:
    return Path(token).suffix.lower() in {".yaml", ".yml"}


def parse_override_path(path: str) -> list[str | int]:
    if not path:
        raise ValueError("override key must not be empty")

    tokens: list[str | int] = []
    pos = 0

    while pos < len(path):
        match = _PATH_TOKEN_RE.match(path, pos)
        if match is None:
            raise ValueError(f"invalid override path syntax: {path!r}")

        key, index = match.groups()
        if key is not None:
            tokens.append(key)
        else:
            tokens.append(int(index))

        pos = match.end()
        if pos < len(path):
            if path[pos] == ".":
                pos += 1
            elif path[pos] == "[":
                pass
            else:
                raise ValueError(f"invalid override path syntax: {path!r}")

    return tokens


def parse_override_value(raw_value: str) -> Any:
    try:
        parsed = OmegaConf.create({"_": raw_value})
        container = OmegaConf.to_container(parsed, resolve=True)
        if not isinstance(container, dict):
            raise ValueError("parsed override root is not a mapping")
        return container["_"]
    except Exception:
        return raw_value


def build_override_fragment(tokens: list[str | int], value: Any) -> Any:
    node: Any = value

    for token in reversed(tokens):
        if isinstance(token, int):
            arr = [None] * (token + 1)
            arr[token] = node
            node = arr
        else:
            node = {token: node}

    return node


def deep_merge_lists(base: list[Any], patch: list[Any]) -> list[Any]:
    size = max(len(base), len(patch))
    out: list[Any] = []

    for i in range(size):
        has_base = i < len(base)
        has_patch = i < len(patch)

        if has_patch and patch[i] is not None:
            if has_base and isinstance(base[i], dict) and isinstance(patch[i], dict):
                out.append(deep_merge_dicts(base[i], patch[i]))
            elif has_base and isinstance(base[i], list) and isinstance(patch[i], list):
                out.append(deep_merge_lists(base[i], patch[i]))
            else:
                out.append(patch[i])
        elif has_base:
            out.append(base[i])
        else:
            out.append(None)

    return out


def deep_merge_dicts(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)

    for key, patch_value in patch.items():
        if key not in out:
            out[key] = patch_value
            continue

        base_value = out[key]
        if isinstance(base_value, dict) and isinstance(patch_value, dict):
            out[key] = deep_merge_dicts(base_value, patch_value)
        elif isinstance(base_value, list) and isinstance(patch_value, list):
            out[key] = deep_merge_lists(base_value, patch_value)
        else:
            out[key] = patch_value

    return out


def apply_override(config: dict[str, Any], token: str) -> dict[str, Any]:
    if "=" not in token:
        raise ValueError("override must have the form key=value")

    raw_path, raw_value = token.split("=", 1)
    tokens = parse_override_path(raw_path)
    value = parse_override_value(raw_value)
    fragment = build_override_fragment(tokens, value)

    if not isinstance(fragment, dict):
        raise ValueError(f"top-level override must produce a mapping: {token!r}")

    return deep_merge_dicts(config, fragment)


def _load_cli_config(tokens: list[str]) -> dict[str, Any]:
    merged: dict[str, Any] = {}

    for token in tokens:
        if is_yaml_file_arg(token):
            loaded = OmegaConf.to_container(OmegaConf.load(token), resolve=True)
            if loaded is None:
                loaded = {}
            if not isinstance(loaded, dict):
                raise typer.BadParameter(f"YAML file must contain a mapping: {token}")
            loaded_mapping: dict[str, Any] = {str(key): value for key, value in loaded.items()}
            merged = deep_merge_dicts(merged, loaded_mapping)
            continue

        if is_yaml_like_arg(token):
            raise typer.BadParameter(f"YAML file does not exist: {token}")

        try:
            merged = apply_override(merged, token)
        except Exception as exc:
            raise typer.BadParameter(f"Invalid override {token!r}: {exc}") from exc

    return merged
