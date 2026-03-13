from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

import yaml


_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]*$")
_INT_RE = re.compile(r"^-?(0|[1-9][0-9]*)$")
_FLOAT_RE = re.compile(r"^-?(?:[0-9]+\.[0-9]*|\.[0-9]+|[0-9]+(?:[eE][+-]?[0-9]+)|[0-9]+\.[0-9]*[eE][+-]?[0-9]+)$")
_BARE_OUT_RE = re.compile(r"^[^\s,\[\]{}\"']+$")


class OlyParseError(ValueError):
    pass


@dataclass
class _Parser:
    text: str
    index: int = 0

    def _error(self, message: str) -> OlyParseError:
        return OlyParseError(f"{message} at position {self.index}")

    def _peek(self) -> str | None:
        if self.index >= len(self.text):
            return None
        return self.text[self.index]

    def _advance(self) -> str:
        if self.index >= len(self.text):
            raise self._error("unexpected end of input")
        ch = self.text[self.index]
        self.index += 1
        return ch

    def _skip_ws(self) -> None:
        while self._peek() in {" ", "\t"}:
            self.index += 1

    def _expect(self, ch: str) -> None:
        got = self._peek()
        if got != ch:
            raise self._error(f"expected {ch!r}")
        self.index += 1

    def _parse_ident(self) -> str:
        start = self.index
        ch = self._peek()
        if ch is None or not (ch.isalpha() or ch == "_"):
            raise self._error("expected identifier")
        self.index += 1
        while True:
            ch = self._peek()
            if ch is None:
                break
            if ch.isalnum() or ch in {"_", "-"}:
                self.index += 1
                continue
            break
        return self.text[start:self.index]

    def _parse_single_quoted(self) -> str:
        self._expect("'")
        out: list[str] = []
        while True:
            ch = self._peek()
            if ch is None:
                raise self._error("unclosed single-quoted string")
            self.index += 1
            if ch == "'":
                return "".join(out)
            out.append(ch)

    def _parse_double_quoted(self) -> str:
        self._expect('"')
        out: list[str] = []
        escapes = {
            '"': '"',
            "\\": "\\",
            "n": "\n",
            "r": "\r",
            "t": "\t",
        }
        while True:
            ch = self._peek()
            if ch is None:
                raise self._error("unclosed double-quoted string")
            self.index += 1
            if ch == '"':
                return "".join(out)
            if ch == "\\":
                nxt = self._peek()
                if nxt is None:
                    raise self._error("dangling escape in string")
                self.index += 1
                out.append(escapes.get(nxt, nxt))
                continue
            out.append(ch)

    def _parse_bare_scalar(self) -> Any:
        start = self.index
        paren_depth = 0
        bracket_depth = 0
        brace_depth = 0
        while True:
            ch = self._peek()
            if ch is None:
                break
            at_top = paren_depth == 0 and bracket_depth == 0 and brace_depth == 0
            if at_top and ch in {",", "]", "}", " ", "\t"}:
                break
            if ch == "(":
                paren_depth += 1
            elif ch == ")" and paren_depth > 0:
                paren_depth -= 1
            elif ch == "[":
                bracket_depth += 1
            elif ch == "]" and bracket_depth > 0:
                bracket_depth -= 1
            elif ch == "{":
                brace_depth += 1
            elif ch == "}" and brace_depth > 0:
                brace_depth -= 1
            self.index += 1
        token = self.text[start:self.index]
        if not token:
            raise self._error("expected value")
        return _coerce_bare_scalar(token)

    def _parse_value(self) -> Any:
        self._skip_ws()
        ch = self._peek()
        if ch is None:
            raise self._error("expected value")
        if ch == "[":
            return self._parse_list()
        if ch == "{":
            return self._parse_inline_map()
        if ch == "'":
            return self._parse_single_quoted()
        if ch == '"':
            return self._parse_double_quoted()
        return self._parse_bare_scalar()

    def _parse_list(self) -> list[Any]:
        self._expect("[")
        self._skip_ws()
        out: list[Any] = []
        if self._peek() == "]":
            self.index += 1
            return out
        while True:
            out.append(self._parse_value())
            self._skip_ws()
            ch = self._peek()
            if ch == ",":
                self.index += 1
                self._skip_ws()
                if self._peek() == "]":
                    raise self._error("trailing comma in list")
                continue
            if ch == "]":
                self.index += 1
                return out
            raise self._error("expected ',' or ']' in list")

    def _parse_kv_pairs(self, *, end_ch: str | None) -> dict[str, Any]:
        out: dict[str, Any] = {}
        while True:
            self._skip_ws()
            if end_ch is not None and self._peek() == end_ch:
                self.index += 1
                return out
            key = self._parse_ident()
            self._skip_ws()
            self._expect(":")
            self._skip_ws()
            value = self._parse_value()
            if key in out:
                raise self._error(f"duplicate key {key!r}")
            out[key] = value
            self._skip_ws()
            ch = self._peek()
            if ch == ",":
                self.index += 1
                self._skip_ws()
                if end_ch is not None and self._peek() == end_ch:
                    raise self._error("trailing comma in map")
                if self._peek() is None:
                    raise self._error("expected key/value pair after comma")
                continue
            if end_ch is not None:
                if ch == end_ch:
                    self.index += 1
                    return out
                raise self._error(f"expected ',' or {end_ch!r} in map")
            if ch is None:
                return out
            raise self._error("expected ',' between key/value pairs")

    def _parse_inline_map(self) -> dict[str, Any]:
        self._expect("{")
        self._skip_ws()
        return self._parse_kv_pairs(end_ch="}")

    def parse_line(self) -> dict[str, Any]:
        self._skip_ws()
        if self._peek() is None:
            raise self._error("empty input")
        transform = self._parse_ident()
        self._skip_ws()
        self._expect(":")
        self._skip_ws()
        if self._peek() is None:
            return {transform: {}}
        if self._peek() == "{":
            payload = self._parse_inline_map()
            self._skip_ws()
            if self._peek() is not None:
                raise self._error("unexpected token after top-level map")
            return {transform: payload}
        payload = self._parse_kv_pairs(end_ch=None)
        return {transform: payload}


def _coerce_bare_scalar(token: str) -> Any:
    lower = token.lower()
    if lower == "true":
        return True
    if lower == "false":
        return False
    if token in {"null", "NULL", "~"}:
        return None
    if _INT_RE.fullmatch(token):
        return int(token)
    if _FLOAT_RE.fullmatch(token):
        return float(token)
    return token


def parse_oly_line(line: str) -> dict[str, Any]:
    return _Parser(text=line).parse_line()


def _quote_string(value: str) -> str:
    escaped = (
        value.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )
    return f'"{escaped}"'


def _emit_value(value: Any) -> str:
    if isinstance(value, str):
        if value and _BARE_OUT_RE.fullmatch(value):
            return value
        return _quote_string(value)
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, list):
        return f"[{', '.join(_emit_value(item) for item in value)}]"
    if isinstance(value, dict):
        if not value:
            return "{}"
        return "{ " + ", ".join(f"{key}: {_emit_value(val)}" for key, val in value.items()) + " }"
    raise TypeError(f"unsupported OLY value type: {type(value).__name__}")


def emit_oly_line(transform_spec: dict[str, Any]) -> str:
    if len(transform_spec) != 1:
        raise ValueError("transform spec must contain exactly one transform")
    transform_name, payload = next(iter(transform_spec.items()))
    if not isinstance(transform_name, str) or not _IDENT_RE.fullmatch(transform_name):
        raise ValueError("transform name must be a valid identifier")
    if payload is None:
        payload = {}
    if not isinstance(payload, dict):
        raise ValueError("transform payload must be a mapping")
    return f"{transform_name}: {_emit_value(payload)}"


def emit_yaml_transform(transform_spec: dict[str, Any]) -> str:
    return yaml.safe_dump(transform_spec, sort_keys=False).strip()
