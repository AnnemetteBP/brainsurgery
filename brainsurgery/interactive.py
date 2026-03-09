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


def _list_model_aliases(state_dict_provider: Any | None) -> set[str]:
    if state_dict_provider is None:
        return set()

    list_aliases = getattr(state_dict_provider, "list_model_aliases", None)
    if callable(list_aliases):
        try:
            aliases = list_aliases()
            return set(aliases)
        except Exception:
            logger.debug("Could not list model aliases for completion", exc_info=True)

    aliases: set[str] = set()
    model_paths = getattr(state_dict_provider, "model_paths", None)
    if isinstance(model_paths, dict):
        aliases.update(model_paths.keys())
    state_dicts = getattr(state_dict_provider, "state_dicts", None)
    if isinstance(state_dicts, dict):
        aliases.update(state_dicts.keys())
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
            loaded[alias] = {str(name) for name in keys()}
        except Exception:
            logger.debug("Could not list tensor names for alias %s", alias, exc_info=True)

    return loaded


def _collect_completion_candidates(state_dict_provider: Any | None) -> list[str]:
    commands = list_transforms()
    candidates: set[str] = set(commands)
    candidates.update(f"{command}:" for command in commands)
    candidates.update({"help", "exit"})

    for command in commands:
        transform = get_transform(command)
        allowed_keys = set(getattr(transform, "allowed_keys", set()) or set())
        required_keys = set(getattr(transform, "required_keys", set()) or set())
        for key in allowed_keys | required_keys:
            candidates.add(key)
            candidates.add(f"{key}:")

    for alias in _list_model_aliases(state_dict_provider):
        candidates.add(alias)
        candidates.add(f"{alias}::")

    loaded_tensors = _list_loaded_tensor_names(state_dict_provider)
    for alias, names in loaded_tensors.items():
        for name in names:
            candidates.add(name)
            candidates.add(f"{alias}::{name}")

    return sorted(candidate for candidate in candidates if candidate)


@contextmanager
def _interactive_completion(candidates: list[str]) -> Iterator[None]:
    if readline is None:
        yield
        return

    previous_completer = readline.get_completer()
    previous_delims = readline.get_completer_delims()
    matches: list[str] = []

    def completer(text: str, state: int) -> str | None:
        nonlocal matches
        if state == 0:
            matches = [candidate for candidate in candidates if candidate.startswith(text)]
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
                    "Tab completion: commands, payload keys, model aliases, loaded tensor names.",
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
        candidates = _collect_completion_candidates(state_dict_provider)

        with _interactive_completion(candidates):
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
