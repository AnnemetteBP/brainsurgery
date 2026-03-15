import logging
import re
import sys
from dataclasses import dataclass
from typing import Any

from ..core import get_transform, list_transforms
from ..engine import (
    list_loaded_tensor_names as list_loaded_tensor_names_from_provider,
    list_model_aliases as list_model_aliases_from_provider,
)


logger = logging.getLogger("brainsurgery")
_KEY_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]*$")


@dataclass(frozen=True)
class _PayloadCursorState:
    context: str
    current_key: str | None
    raw_value_fragment: str | None
    used_keys: set[str]


def _find_top_level_colon(segment: str) -> int | None:
    brace_depth = 0
    bracket_depth = 0
    paren_depth = 0
    in_single = False
    in_double = False
    escaped = False

    for index, ch in enumerate(segment):
        if in_single:
            if ch == "'":
                in_single = False
            continue
        if in_double:
            if escaped:
                escaped = False
                continue
            if ch == "\\":
                escaped = True
                continue
            if ch == '"':
                in_double = False
            continue
        if ch == "'":
            in_single = True
            continue
        if ch == '"':
            in_double = True
            continue
        if ch == "(":
            paren_depth += 1
            continue
        if ch == ")" and paren_depth > 0:
            paren_depth -= 1
            continue
        if ch == "[":
            bracket_depth += 1
            continue
        if ch == "]" and bracket_depth > 0:
            bracket_depth -= 1
            continue
        if ch == "{":
            brace_depth += 1
            continue
        if ch == "}" and brace_depth > 0:
            brace_depth -= 1
            continue
        if ch == ":" and brace_depth == 0 and bracket_depth == 0 and paren_depth == 0:
            return index
    return None


def _parse_key_from_segment(segment: str) -> str | None:
    colon_index = _find_top_level_colon(segment)
    if colon_index is None:
        return None
    key = segment[:colon_index].strip()
    if not _KEY_IDENT_RE.fullmatch(key):
        return None
    return key


def _split_top_level_segments(payload: str) -> tuple[list[str], str]:
    working = payload
    stripped = payload.lstrip()
    if stripped.startswith("{"):
        brace_index = payload.find("{")
        if brace_index >= 0:
            working = payload[brace_index + 1 :]

    segments: list[str] = []
    start = 0
    brace_depth = 0
    bracket_depth = 0
    paren_depth = 0
    in_single = False
    in_double = False
    escaped = False

    for index, ch in enumerate(working):
        if in_single:
            if ch == "'":
                in_single = False
            continue
        if in_double:
            if escaped:
                escaped = False
                continue
            if ch == "\\":
                escaped = True
                continue
            if ch == '"':
                in_double = False
            continue
        if ch == "'":
            in_single = True
            continue
        if ch == '"':
            in_double = True
            continue
        if ch == "(":
            paren_depth += 1
            continue
        if ch == ")" and paren_depth > 0:
            paren_depth -= 1
            continue
        if ch == "[":
            bracket_depth += 1
            continue
        if ch == "]" and bracket_depth > 0:
            bracket_depth -= 1
            continue
        if ch == "{":
            brace_depth += 1
            continue
        if ch == "}":
            if brace_depth == 0:
                segments.append(working[start:index])
                return segments[:-1], segments[-1]
            brace_depth -= 1
            continue
        if (
            ch == ","
            and brace_depth == 0
            and bracket_depth == 0
            and paren_depth == 0
        ):
            segments.append(working[start:index])
            start = index + 1

    segments.append(working[start:])
    return segments[:-1], segments[-1]


def _payload_cursor_state(before_cursor: str) -> _PayloadCursorState:
    transform_match = re.match(
        r"^\s*(?:-\s*)?[A-Za-z_][A-Za-z0-9_-]*\s*:(.*)$",
        before_cursor,
        re.DOTALL,
    )
    if transform_match is None:
        context = "key" if not before_cursor.strip() else "any"
        return _PayloadCursorState(
            context=context,
            current_key=None,
            raw_value_fragment=None,
            used_keys=set(),
        )

    payload = transform_match.group(1)
    if not payload.strip():
        return _PayloadCursorState(
            context="value",
            current_key=None,
            raw_value_fragment="",
            used_keys=set(),
        )
    completed_segments, current_segment = _split_top_level_segments(payload)

    used_keys = {
        key
        for key in (_parse_key_from_segment(segment) for segment in completed_segments)
        if key is not None
    }

    current_key = _parse_key_from_segment(current_segment)
    if current_key is not None:
        used_keys.add(current_key)
    colon_index = _find_top_level_colon(current_segment)
    raw_value_fragment: str | None = None
    if colon_index is not None:
        raw_value_fragment = current_segment[colon_index + 1 :]

    if not current_segment.strip():
        context = "key"
    elif colon_index is None:
        context = "key"
    else:
        context = "value"

    return _PayloadCursorState(
        context=context,
        current_key=current_key,
        raw_value_fragment=raw_value_fragment,
        used_keys=used_keys,
    )


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


def list_loaded_tensor_names(state_dict_provider: Any | None) -> dict[str, set[str]]:
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

    loaded_tensors = list_loaded_tensor_names(state_dict_provider)
    for alias, names in loaded_tensors.items():
        for name in names:
            candidates.add(name)
            candidates.add(f"{alias}::{name}")

    return sorted(candidate for candidate in candidates if candidate)


def _payload_context(before_cursor: str) -> str:
    state = _payload_cursor_state(before_cursor)
    return state.context


def _current_value_key(before_cursor: str) -> str | None:
    state = _payload_cursor_state(before_cursor)
    return state.current_key


def _current_value_fragment(before_cursor: str) -> str | None:
    state = _payload_cursor_state(before_cursor)
    return state.raw_value_fragment


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
        state = _payload_cursor_state(before_cursor)
        return state.used_keys

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
        if current_key is None:  # pragma: no cover
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
                if raw_text.rstrip().endswith(","):  # pragma: no cover
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
                if raw_text.rstrip().endswith(","):  # pragma: no cover
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


def _completion_display_hook(
    substitution: str,
    matches: list[str],
    longest_match_length: int,
    readline_module: Any | None,
) -> None:
    del substitution, longest_match_length
    preview = _render_completion_preview(matches)
    if not preview:
        return
    sys.stdout.write(f"\nCompletions: {preview}\n")
    try:
        if readline_module is not None:
            readline_module.redisplay()
    except Exception:
        pass


def _configure_readline_completion_bindings(readline_module: Any | None) -> None:
    if readline_module is None:
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
            readline_module.parse_and_bind(command)
        except Exception:
            logger.debug("Could not apply readline command %r", command, exc_info=True)

    set_display_hook = getattr(readline_module, "set_completion_display_matches_hook", None)
    if callable(set_display_hook):
        try:
            set_display_hook(
                lambda substitution, matches, longest_match_length: _completion_display_hook(
                    substitution,
                    matches,
                    longest_match_length,
                    readline_module,
                )
            )
        except Exception:
            logger.debug("Could not configure readline display hook", exc_info=True)


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
