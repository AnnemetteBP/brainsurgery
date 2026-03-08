from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import typer
from omegaconf import OmegaConf

from .arena import ArenaError, SegmentedFileBackedArena
from .model import parse_shard_size
from .plan import compile_plan
from .providers import ArenaStateDictProvider, InMemoryStateDictProvider
from .transform import apply_transform

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("brainsurgery")

app = typer.Typer(help="Brain surgery CLI.")

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
    # Use OmegaConf/YAML parsing for numbers, booleans, lists, dicts, quoted strings.
    # Fall back to the raw string for literals like *all that YAML would treat specially.
    try:
        parsed = OmegaConf.create({"_": raw_value})
        container = OmegaConf.to_container(parsed, resolve=True)
        assert isinstance(container, dict)
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


def load_cli_config(tokens: list[str]) -> dict[str, Any]:
    merged: dict[str, Any] = {}

    for token in tokens:
        if is_yaml_file_arg(token):
            loaded = OmegaConf.to_container(OmegaConf.load(token), resolve=True)
            if loaded is None:
                loaded = {}
            if not isinstance(loaded, dict):
                raise typer.BadParameter(f"YAML file must contain a mapping: {token}")
            merged = deep_merge_dicts(merged, loaded)
            continue

        if is_yaml_like_arg(token):
            raise typer.BadParameter(f"YAML file does not exist: {token}")

        try:
            merged = apply_override(merged, token)
        except Exception as exc:
            raise typer.BadParameter(f"Invalid override {token!r}: {exc}") from exc

    return merged


@app.command()
def run(
    config_items: list[str] = typer.Argument(
        None,
        help=(
            "Zero or more YAML files and/or overrides in any order. Existing .yaml/.yml "
            "files are loaded and merged; everything else is treated as an override."
        ),
    ),
    shard_size: str = typer.Option("5GB", help="Default shard size for directory outputs"),
    num_workers: int = typer.Option(8, help="Max number of parallel I/O workers"),
    provider: str = typer.Option("inmemory", help="State-dict provider: inmemory or arena"),
    arena_root: Path = typer.Option(
        Path(".brainsurgery"),
        help="Arena directory when using the arena provider",
    ),
    arena_segment_size: str = typer.Option(
        "1GB",
        help="Arena segment size, e.g. 1GB, 4GB, 512MB",
    ),
) -> None:
    """Load a plan, execute it, and save the rewritten output checkpoint."""
    raw_plan = load_cli_config(config_items or [])

    logger.info("Scrubbing in. Surgical plan assembled from %d config item(s)", len(config_items or []))
    surgery_plan = compile_plan(raw_plan)
    logger.info(
        "Surgical plan ready: %d brain(s) prepped, %d procedure(s) scheduled, preservation %s",
        len(surgery_plan.inputs),
        len(surgery_plan.transforms),
        surgery_plan.output.path if surgery_plan.output else None,
    )

    provider_name = provider.strip().lower()

    try:
        if provider_name == "inmemory":
            state_dict_provider = InMemoryStateDictProvider(
                surgery_plan.inputs,
                max_io_workers=num_workers,
            )
        elif provider_name == "arena":
            segment_size_bytes = parse_shard_size(arena_segment_size)
            if segment_size_bytes is None:
                raise typer.BadParameter("arena-segment-size must not be 'none'")

            arena = SegmentedFileBackedArena(
                arena_root,
                segment_size_bytes=segment_size_bytes,
            )
            state_dict_provider = ArenaStateDictProvider(
                surgery_plan.inputs,
                arena=arena,
                max_io_workers=num_workers,
            )
        else:
            raise typer.BadParameter("provider must be either 'inmemory' or 'arena'")
    except ArenaError as exc:
        raise typer.BadParameter(str(exc)) from exc

    try:
        for transform_index, transform in enumerate(surgery_plan.transforms, start=1):
            logger.info(
                "Procedure %d/%d: positioning instruments for %s",
                transform_index,
                len(surgery_plan.transforms),
                type(transform.spec).__name__,
            )
            transform_result = apply_transform(transform, state_dict_provider)
            logger.info(
                "Procedure %d/%d complete: %s affected %d site(s)",
                transform_index,
                len(surgery_plan.transforms),
                transform_result.name,
                transform_result.count,
            )

        if surgery_plan.output is None:
            logger.info("No preservation requested; concluding operation without closure")
            return
        written_path = state_dict_provider.save_output(
            surgery_plan,
            default_shard_size=shard_size,
            max_io_workers=num_workers,
        )
        logger.info("Operation complete. Brain preserved at %s", written_path)
    finally:
        state_dict_provider.close()


if __name__ == "__main__":
    app()
