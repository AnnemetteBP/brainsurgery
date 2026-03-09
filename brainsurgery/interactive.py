from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import typer
from omegaconf import OmegaConf
from rich.console import Console
from rich.panel import Panel

from .history import add_history_entry
from .transform import get_transform, list_transforms

logger = logging.getLogger("brainsurgery")
console = Console()

try:
    import readline
except ImportError:  # pragma: no cover
    readline = None


def _normalize_single_transform_spec(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        if len(raw) != 1:
            raise ValueError("each transform spec must be a mapping with exactly one key")
        return raw

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
        loaded = OmegaConf.to_container(OmegaConf.create(block), resolve=True)
    except Exception as exc:
        raise ValueError(f"invalid YAML: {exc}") from exc

    return normalize_transform_specs(loaded)


def _collect_completion_candidates(state_dict_provider: Any | None) -> list[str]:
    del state_dict_provider
    commands = list_transforms()
    candidates: set[str] = set()
    for command in commands:
        transform = get_transform(command)
        required_keys = set(getattr(transform, "required_keys", set()) or set())
        if required_keys:
            candidates.add(f"{command}:")
        else:
            candidates.add(command)
    return sorted(candidate for candidate in candidates if candidate)


def _list_model_aliases(state_dict_provider: Any | None) -> set[str]:
    if state_dict_provider is None:
        return set()

    list_aliases = getattr(state_dict_provider, "list_model_aliases", None)
    if callable(list_aliases):
        try:
            return set(str(alias) for alias in list_aliases())
        except Exception:
            logger.debug("Could not list model aliases for completion", exc_info=True)

    aliases: set[str] = set()
    model_paths = getattr(state_dict_provider, "model_paths", None)
    if isinstance(model_paths, dict):
        aliases.update(str(alias) for alias in model_paths.keys())
    state_dicts = getattr(state_dict_provider, "state_dicts", None)
    if isinstance(state_dicts, dict):
        aliases.update(str(alias) for alias in state_dicts.keys())
    return aliases


def _list_loaded_tensor_names(state_dict_provider: Any | None) -> dict[str, set[str]]:
    if state_dict_provider is None:
        return {}

    state_dicts = getattr(state_dict_provider, "state_dicts", None)
    if not isinstance(state_dicts, dict):
        return {}

    loaded: dict[str, set[str]] = {}
    for alias, state_dict in state_dicts.items():
        keys = getattr(state_dict, "keys", None)
        if not callable(keys):
            continue
        try:
            loaded[str(alias)] = {str(name) for name in keys()}
        except Exception:
            logger.debug("Could not list tensor names for alias %s", alias, exc_info=True)
    return loaded


def _extract_transform_name(line: str) -> str | None:
    stripped = line.strip()
    if stripped.startswith("- "):
        stripped = stripped[2:].lstrip()
    if ":" not in stripped:
        return None
    name = stripped.split(":", 1)[0].strip()
    if not name:
        return None
    if name not in set(list_transforms()):
        return None
    return name


def _infer_active_transform(lines: list[str], line_buffer: str) -> str | None:
    inline = _extract_transform_name(line_buffer)
    if inline:
        return inline
    for line in lines:
        name = _extract_transform_name(line)
        if name:
            return name
    return None


def _collect_payload_candidates(
    *,
    active_transform: str | None,
    state_dict_provider: Any | None,
) -> list[str]:
    candidates: set[str] = {"{ ", "}", "[ ", "]", ", ", ": "}

    if active_transform:
        try:
            transform = get_transform(active_transform)
            required_keys = set(getattr(transform, "required_keys", set()) or set())
            allowed_keys = set(getattr(transform, "allowed_keys", set()) or set())
            for key in sorted(required_keys | allowed_keys):
                candidates.add(f"{key}: ")
        except Exception:
            pass

    aliases = _list_model_aliases(state_dict_provider)
    candidates.update(aliases)
    candidates.update(f"{alias}::" for alias in aliases)

    loaded_tensors = _list_loaded_tensor_names(state_dict_provider)
    for alias, names in loaded_tensors.items():
        for name in names:
            candidates.add(name)
            candidates.add(f"{alias}::{name}")

    return sorted(candidate for candidate in candidates if candidate)


def _is_top_level_completion_position(line_buffer: str, begidx: int) -> bool:
    if begidx > len(line_buffer):
        return False
    before_cursor = line_buffer[:begidx]
    stripped = before_cursor.lstrip()
    if not stripped:
        return True
    if stripped.startswith("- "):
        remainder = stripped[2:]
        return remainder == ""
    return False


@contextmanager
def _interactive_completion(
    *,
    top_level_candidates: list[str],
    lines: list[str],
    state_dict_provider: Any | None,
) -> Iterator[None]:
    if readline is None:
        yield
        return

    previous_completer = readline.get_completer()
    previous_delims = readline.get_completer_delims()
    matches: list[str] = []

    def completer(text: str, state: int) -> str | None:
        nonlocal matches
        if state == 0:
            try:
                line_buffer = readline.get_line_buffer()
                begidx = readline.get_begidx()
            except Exception:
                line_buffer = ""
                begidx = 0

            if _is_top_level_completion_position(line_buffer, begidx):
                matches = [
                    candidate for candidate in top_level_candidates if candidate.startswith(text)
                ]
            else:
                active_transform = _infer_active_transform(lines, line_buffer)
                payload_candidates = _collect_payload_candidates(
                    active_transform=active_transform,
                    state_dict_provider=state_dict_provider,
                )
                matches = [candidate for candidate in payload_candidates if candidate.startswith(text)]
        if state < len(matches):
            return matches[state]
        return None

    try:
        readline.parse_and_bind("tab: complete")
    except Exception:
        pass

    try:
        readline.set_completer_delims(" \t\n")
        readline.set_completer(completer)
    except Exception:
        logger.debug("Could not configure readline completion", exc_info=True)

    try:
        yield
    finally:
        try:
            readline.set_completer(previous_completer)
            readline.set_completer_delims(previous_delims)
        except Exception:
            logger.debug("Could not restore readline completion", exc_info=True)


def prompt_interactive_transform(state_dict_provider: Any | None = None) -> list[dict[str, Any]] | None:
    console.print()
    console.print(
        Panel.fit(
            "\n".join(
                [
                    "[bold]Interactive mode[/bold]",
                    "Enter one transform as YAML, or a YAML list of transforms.",
                    "Finish input with an empty line.",
                    "Tab completion: top-level transforms, then payload keys/aliases/tensors/YAML syntax.",
                    "Special transforms: [cyan]help[/cyan], [cyan]exit[/cyan]",
                    "Example: [dim]copy: { from: ln_f.weight, to: ln_f_copy.weight }[/dim]",
                ]
            ),
            border_style="cyan",
        )
    )

    while True:
        lines: list[str] = []
        prompt = typer.style("brainsurgery> ", fg=typer.colors.CYAN, bold=True)
        continuation = typer.style("... ", fg=typer.colors.BRIGHT_BLACK)
        top_level_candidates = _collect_completion_candidates(state_dict_provider)

        with _interactive_completion(
            top_level_candidates=top_level_candidates,
            lines=lines,
            state_dict_provider=state_dict_provider,
        ):
            while True:
                try:
                    line = input(prompt)
                except EOFError:
                    return None

                if line.strip() == "":
                    if not lines:
                        continue

                    block = "\n".join(lines)
                    try:
                        parsed = parse_transform_block(block)
                        add_history_entry(block)
                        return parsed
                    except ValueError as exc:
                        logger.error("Interactive transform rejected: %s", exc)
                        console.print(f"[red]Rejected:[/red] {exc}")
                        console.print("[yellow]Try again.[/yellow]")
                        break

                lines.append(line)
                prompt = continuation
