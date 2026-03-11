from __future__ import annotations

import logging
import re
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import typer
from omegaconf import OmegaConf
from rich.console import Console
from rich.panel import Panel

from .history import _add_history_entry
from .provider_utils import (
    _list_loaded_tensor_names as list_loaded_tensor_names_from_provider,
    list_model_aliases as list_model_aliases_from_provider,
)
from ..core import get_transform, list_transforms

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
        if not getattr(transform, "completion_requires_payload", True):
            candidates.add(command)
        else:
            candidates.add(f"{command}: ")
    return sorted(candidate for candidate in candidates if candidate)


def _list_model_aliases(state_dict_provider: Any | None) -> set[str]:
    try:
        return list_model_aliases_from_provider(state_dict_provider)
    except Exception:
        logger.debug("Could not list model aliases for completion", exc_info=True)
        return set()


def _list_loaded_tensor_names(state_dict_provider: Any | None) -> dict[str, set[str]]:
    try:
        return list_loaded_tensor_names_from_provider(state_dict_provider)
    except Exception:
        logger.debug("Could not list tensor names for completion", exc_info=True)
        return {}


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
    candidates: set[str] = {"{ ", "}", "[ ", "]", ", "}

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


def _payload_context(before_cursor: str) -> str:
    trimmed = before_cursor.rstrip()
    if not trimmed:
        return "key"
    if trimmed.endswith("{") or trimmed.endswith(","):
        return "key"

    last_colon = trimmed.rfind(":")
    last_struct_sep = max(trimmed.rfind("{"), trimmed.rfind(","))
    if last_colon > last_struct_sep:
        return "value"
    return "any"


def _current_value_key(before_cursor: str) -> str | None:
    match = re.search(r"([A-Za-z_][A-Za-z0-9_]*)\s*:\s*[^,{}[\]]*$", before_cursor)
    if match is None:
        return None
    return match.group(1)


def _current_value_fragment(before_cursor: str) -> str | None:
    match = re.search(r"[A-Za-z_][A-Za-z0-9_]*\s*:\s*([^,{}[\]]*)$", before_cursor)
    if match is None:
        return None
    return match.group(1)


def _is_committed_value_fragment(raw_fragment: str) -> bool:
    if not raw_fragment:
        return False
    return raw_fragment.endswith(" ") and bool(raw_fragment.strip())


def _is_transform_payload_start(
    *,
    line_buffer: str,
    begidx: int,
    active_transform: str | None,
    endidx: int | None = None,
) -> bool:
    if active_transform is None:
        return False
    if endidx is None:
        endidx = begidx
    if endidx < 0:
        endidx = 0
    if endidx > len(line_buffer):
        endidx = len(line_buffer)
    before = line_buffer[:endidx].strip()
    return before in {f"{active_transform}:", f"{active_transform}: "}


def _match_payload_candidates(
    *,
    text: str,
    line_buffer: str,
    begidx: int,
    endidx: int | None = None,
    payload_candidates: list[str],
    active_transform: str | None = None,
    model_aliases: list[str] | None = None,
) -> list[str]:
    transform = None
    if active_transform is not None:
        try:
            transform = get_transform(active_transform)
        except Exception:
            transform = None

    def _dedupe_preserve_order(items: list[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for item in items:
            if item in seen:
                continue
            seen.add(item)
            out.append(item)
        return out

    def _used_keys(before_cursor: str) -> set[str]:
        return set(re.findall(r"([A-Za-z_][A-Za-z0-9_]*)\s*:", before_cursor))

    def _remaining_key_candidates(before_cursor: str) -> list[str]:
        used_keys = _used_keys(before_cursor)
        remaining = [
            candidate
            for candidate in payload_candidates
            if candidate.endswith(": ") and candidate[:-2] not in used_keys
        ]
        if not remaining:
            return ["}"]
        return [f", {candidate}" for candidate in remaining] + ["}"]

    def _ordered_transform_keys() -> list[str]:
        if transform is None:
            return []
        keys = getattr(transform, "completion_reference_keys", lambda: [])()
        return [key for key in keys if isinstance(key, str) and key]

    def _next_reference_key(before_cursor: str, current_key: str | None) -> str | None:
        if current_key is None:
            return None
        ordered_keys = _ordered_transform_keys()
        if current_key not in ordered_keys:
            return None
        used_keys = _used_keys(before_cursor)
        current_index = ordered_keys.index(current_key)
        for next_key in ordered_keys[current_index + 1 :]:
            if next_key not in used_keys:
                return next_key
        return None

    def _reference_candidates(prefix_text: str, value_key: str | None) -> list[str]:
        alias_prefix_candidates = [
            candidate
            for candidate in payload_candidates
            if candidate.endswith("::") and candidate[:-2] in set(model_aliases or [])
        ]
        if "::" not in prefix_text and len(alias_prefix_candidates) > 1:
            alias_matches = [
                candidate for candidate in alias_prefix_candidates if candidate.startswith(prefix_text)
            ]
            return _dedupe_preserve_order(alias_matches)

        ref_candidates = [
            candidate
            for candidate in payload_candidates
            if (
                not candidate.endswith(": ")
                and candidate not in {"{ ", "}", "[ ", "]", ", "}
            )
        ]
        # Prefer alias:: over bare alias in reference contexts.
        ref_candidates = [
            candidate for candidate in ref_candidates if not (f"{candidate}::" in ref_candidates)
        ]
        ref_matches = [candidate for candidate in ref_candidates if candidate.startswith(prefix_text)]
        next_key = _next_reference_key(before_cursor, value_key)
        if (
            next_key
            and prefix_text
            and prefix_text in ref_candidates
            and not prefix_text.endswith("::")
        ):
            ref_matches.append(f"{prefix_text}, {next_key}: ")
        return _dedupe_preserve_order(ref_matches)

    def _key_candidates_for_transform(
        *,
        before_cursor: str,
        prefix_text: str,
    ) -> list[str] | None:
        if transform is None:
            return None
        return transform.completion_key_candidates(before_cursor, prefix_text)

    def _value_candidates_for_transform(value_key: str | None, prefix_text: str) -> list[str] | None:
        if transform is None:
            return None
        return transform.completion_value_candidates(value_key, prefix_text, sorted(model_aliases or []))

    def _committed_next_candidates(value_key: str | None) -> list[str] | None:
        if transform is None:
            return None
        return transform.completion_committed_next_candidates(value_key)

    raw_text = text
    prefix = text
    if endidx is None:
        endidx = begidx + len(text)
    if endidx < 0:
        endidx = 0
    if endidx > len(line_buffer):
        endidx = len(line_buffer)
    before_cursor = line_buffer[:endidx]
    ctx = _payload_context(before_cursor)
    if ctx == "key":
        if "," in prefix:
            prefix = prefix.rsplit(",", 1)[-1]
        if "{" in prefix:
            prefix = prefix.rsplit("{", 1)[-1]
        if "}" in prefix:
            prefix = prefix.rsplit("}", 1)[-1]
        prefix = prefix.lstrip()
    if _is_transform_payload_start(
        line_buffer=line_buffer,
        begidx=begidx,
        active_transform=active_transform,
        endidx=endidx,
    ):
        if transform is not None:
            payload_start_candidates = transform.completion_payload_start_candidates(prefix)
            if payload_start_candidates is not None:
                return payload_start_candidates
        if not prefix:
            return ["{ "]
        return [candidate for candidate in ["{ "] if candidate.startswith(prefix)]

    value_key = _current_value_key(before_cursor)
    raw_value_fragment = _current_value_fragment(before_cursor) or ""
    if ctx == "key":
        key_candidates = _key_candidates_for_transform(before_cursor=before_cursor, prefix_text=prefix)
        if key_candidates is not None:
            if raw_text.rstrip().endswith("{"):
                return [f"{raw_text} {candidate}" for candidate in key_candidates if candidate != "}"]
            if before_cursor.rstrip().endswith(",") and not before_cursor.endswith(", "):
                if raw_text.rstrip().endswith(","):
                    return [
                        f"{raw_text} {candidate}" for candidate in key_candidates if candidate != "}"
                    ] + (["}"] if "}" in key_candidates else [])
                return [f" {candidate}" for candidate in key_candidates if candidate != "}"] + (
                    ["}"] if "}" in key_candidates else []
                )
            return key_candidates

    if not prefix:
        if ctx == "key":
            used_keys = _used_keys(before_cursor)
            key_candidates = [
                candidate
                for candidate in payload_candidates
                if (
                    candidate.endswith(": ")
                    and candidate[:-2] not in used_keys
                )
            ]
            if before_cursor.rstrip().endswith(",") and not before_cursor.endswith(", "):
                if raw_text.rstrip().endswith(","):
                    return [f"{raw_text} {candidate}" for candidate in key_candidates]
                return [f" {candidate}" for candidate in key_candidates]
            return key_candidates
        if ctx == "value":
            if _is_committed_value_fragment(raw_value_fragment):
                return _committed_next_candidates(value_key) or _remaining_key_candidates(before_cursor)
            transform_value_candidates = _value_candidates_for_transform(value_key, "")
            if transform_value_candidates is not None:
                return transform_value_candidates
            if value_key in _ordered_transform_keys():
                return _reference_candidates("", value_key)
            return [candidate for candidate in payload_candidates if candidate in {"{ ", "[ ", "]", "}"}]
        return payload_candidates

    if "::" in prefix:
        value_key = _current_value_key(before_cursor)
        if ctx == "value" and _value_candidates_for_transform(value_key, prefix) is not None:
            return _value_candidates_for_transform(value_key, prefix) or []
        if ctx == "value" and value_key in _ordered_transform_keys():
            return _reference_candidates(prefix, value_key)
        return [candidate for candidate in payload_candidates if "::" in candidate and candidate.startswith(prefix)]

    if prefix[0] in "{[]},:":
        return [candidate for candidate in payload_candidates if candidate.startswith(prefix)]

    if prefix.endswith(":") or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*:?", prefix):
        if ctx == "key":
            used_keys = _used_keys(before_cursor)
            key_candidates = [
                candidate
                for candidate in payload_candidates
                if (
                    candidate.endswith(": ")
                    and candidate.startswith(prefix)
                    and candidate[:-2] not in used_keys
                )
            ]
            if before_cursor.rstrip().endswith(",") and not before_cursor.endswith(", "):
                if raw_text.rstrip().endswith(","):
                    return [f"{raw_text} {candidate}" for candidate in key_candidates]
                return [f" {candidate}" for candidate in key_candidates]
            return key_candidates

    if ctx == "value":
        value_key = _current_value_key(before_cursor)
        raw_value_fragment = _current_value_fragment(before_cursor) or ""
        if _is_committed_value_fragment(raw_value_fragment):
            committed_next = _committed_next_candidates(value_key) or _remaining_key_candidates(before_cursor)
            return [candidate for candidate in committed_next if candidate.startswith(prefix)]
        transform_value_candidates = _value_candidates_for_transform(value_key, prefix)
        if transform_value_candidates is not None:
            return transform_value_candidates
        if value_key in _ordered_transform_keys():
            return _reference_candidates(prefix, value_key)
        return [
            candidate
            for candidate in payload_candidates
            if not candidate.endswith(": ") and candidate.startswith(prefix)
        ]

    return [candidate for candidate in payload_candidates if candidate.startswith(prefix)]


def _render_completion_preview(matches: list[str], limit: int = 16) -> str:
    if not matches:
        return ""
    shown = matches[:limit]
    remaining = len(matches) - len(shown)
    suffix = f" (+{remaining} more)" if remaining > 0 else ""
    return "  ".join(shown) + suffix


def _completion_display_hook(substitution: str, matches: list[str], longest_match_length: int) -> None:
    del substitution, longest_match_length
    preview = _render_completion_preview(matches)
    if not preview:
        return
    sys.stdout.write(f"\nCompletions: {preview}\n")
    try:
        if readline is not None:
            readline.redisplay()
    except Exception:
        pass


def _configure_readline_completion_bindings() -> None:
    if readline is None:
        return

    commands = [
        "set show-all-if-ambiguous on",
        "set show-all-if-unmodified on",
        "set completion-query-items 200",
        "set page-completions off",
        "set menu-complete-display-prefix on",
        "tab: menu-complete",
        '"\\e[Z": menu-complete-backward',
    ]

    for command in commands:
        try:
            readline.parse_and_bind(command)
        except Exception:
            logger.debug("Could not apply readline command %r", command, exc_info=True)


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
                get_endidx = getattr(readline, "get_endidx", None)
                if callable(get_endidx):
                    endidx = get_endidx()
                else:
                    endidx = begidx + len(text)
            except Exception:
                line_buffer = ""
                begidx = 0
                endidx = 0

            if _is_top_level_completion_position(line_buffer, begidx):
                matches = [
                    candidate for candidate in top_level_candidates if candidate.startswith(text)
                ]
                if not matches and ":" in line_buffer:
                    active_transform = _infer_active_transform(lines, line_buffer)
                    payload_candidates = _collect_payload_candidates(
                        active_transform=active_transform,
                        state_dict_provider=state_dict_provider,
                    )
                    matches = _match_payload_candidates(
                        text=text,
                        line_buffer=line_buffer,
                        begidx=begidx,
                        endidx=endidx,
                        payload_candidates=payload_candidates,
                        active_transform=active_transform,
                        model_aliases=sorted(_list_model_aliases(state_dict_provider)),
                    )
            else:
                active_transform = _infer_active_transform(lines, line_buffer)
                payload_candidates = _collect_payload_candidates(
                    active_transform=active_transform,
                    state_dict_provider=state_dict_provider,
                )
                matches = _match_payload_candidates(
                    text=text,
                    line_buffer=line_buffer,
                    begidx=begidx,
                    endidx=endidx,
                    payload_candidates=payload_candidates,
                    active_transform=active_transform,
                    model_aliases=sorted(_list_model_aliases(state_dict_provider)),
                )
        if state < len(matches):
            return matches[state]
        return None

    _configure_readline_completion_bindings()

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
                except KeyboardInterrupt:
                    console.print()
                    console.print("KeyboardInterrupt")
                    break
                except EOFError:
                    return None

                if line.strip() == "":
                    if not lines:
                        continue

                    block = "\n".join(lines)
                    try:
                        parsed = parse_transform_block(block)
                        _add_history_entry(block)
                        return parsed
                    except ValueError as exc:
                        logger.error("Interactive transform rejected: %s", exc)
                        console.print(f"[red]Rejected:[/red] {exc}")
                        console.print("[yellow]Try again.[/yellow]")
                        break

                lines.append(line)
                prompt = continuation
