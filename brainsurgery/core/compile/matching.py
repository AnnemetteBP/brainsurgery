from dataclasses import dataclass
import re
from typing import Optional


class _MatchError(RuntimeError):
    pass


_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_INTERP_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


@dataclass(frozen=True)
class StructuredMatch:
    bindings: dict[str, object]


class _StructuredPathMatcher:
    """
    Structured path matcher for dot-separated tensor names.

    Source-pattern token semantics:
      - literal           exact segment match
      - $x                capture exactly one segment into x
      - *xs               capture zero or more segments into xs
      - ~REGEX            regex fullmatch on one segment, bind nothing
      - ~x::REGEX         bind x to the whole segment if REGEX has 0 groups,
                          or to group(1) if REGEX has 1 group
      - ~x,y,z::REGEX     bind variables to corresponding capturing groups

    Output-pattern token semantics:
      - literal / template string with ${x} interpolation
      - $x                interpolate scalar capture x (shorthand for ${x})
      - *xs               splice a captured variadic binding
      - regex tokens are forbidden
    """

    def split_name(self, name: str) -> list[str]:
        return name.split(".")

    def join_name(self, parts: list[str]) -> str:
        return ".".join(parts)

    def match(self, pattern: list[str], name: str) -> Optional[StructuredMatch]:
        segments = self.split_name(name)
        env = self._match_pattern(pattern, segments)
        if env is None:
            return None
        return StructuredMatch(bindings=env)

    def rewrite(self, pattern: list[str], match: StructuredMatch) -> str:
        return self._rewrite_name(pattern, match.bindings)

    def match_and_rewrite(
        self,
        *,
        from_pattern: list[str],
        to_pattern: list[str],
        name: str,
    ) -> Optional[str]:
        matched = self.match(from_pattern, name)
        if matched is None:
            return None
        return self.rewrite(to_pattern, matched)

    def _is_single_capture_token(self, token: str) -> bool:
        return token.startswith("$") and len(token) > 1

    def _is_variadic_capture_token(self, token: str) -> bool:
        return token.startswith("*") and len(token) > 1

    def _is_regex_token(self, token: str) -> bool:
        return token.startswith("~") and len(token) > 1

    def _validate_capture_name(self, name: str, *, token: str) -> None:
        if not _IDENT_RE.fullmatch(name):
            raise _MatchError(f"invalid capture name {name!r} in token: {token!r}")

    def _bind_scalar(self, env: dict[str, object], name: str, value: str) -> bool:
        existing = env.get(name)
        if existing is None:
            env[name] = value
            return True
        return existing == value

    def _bind_variadic(self, env: dict[str, object], name: str, value: list[str]) -> bool:
        existing = env.get(name)
        if existing is None:
            env[name] = value
            return True
        return existing == value

    def _parse_regex_token(self, token: str) -> tuple[list[str], str]:
        """
        ~REGEX              -> ([], REGEX)
        ~x::REGEX           -> (["x"], REGEX)
        ~x,y,z::REGEX       -> (["x", "y", "z"], REGEX)

        No whitespace stripping is performed.
        """
        body = token[1:]
        left, sep, right = body.partition("::")

        if sep == "":
            return [], body

        if left == "":
            raise _MatchError(f"invalid regex token: {token!r}")
        if right == "":
            raise _MatchError(f"missing regex body in token: {token!r}")

        names = left.split(",")
        if any(name == "" for name in names):
            raise _MatchError(f"invalid regex binding list in token: {token!r}")

        for name in names:
            self._validate_capture_name(name, token=token)

        return names, right

    def _match_regex_token(self, token: str, segment: str, env: dict[str, object]) -> bool:
        names, pattern = self._parse_regex_token(token)

        try:
            rx = re.compile(pattern)
        except re.error as exc:
            raise _MatchError(f"invalid structured regex in token {token!r}: {exc}") from exc

        if rx.groupindex:
            raise _MatchError(f"named groups are not allowed in structured regex token: {token!r}")

        m = rx.fullmatch(segment)
        if m is None:
            return False

        ngroups = rx.groups

        if not names:
            return True

        if len(names) == 1:
            if ngroups == 0:
                value = segment
            elif ngroups == 1:
                captured = m.group(1)
                if captured is None:
                    raise _MatchError(f"structured regex token {token!r} captured None")
                value = captured
            else:
                raise _MatchError(
                    f"regex token {token!r} binds 1 variable but regex has {ngroups} capturing groups"
                )
            return self._bind_scalar(env, names[0], value)

        if ngroups != len(names):
            raise _MatchError(
                f"regex token {token!r} binds {len(names)} variables but regex has {ngroups} capturing groups"
            )

        env2 = dict(env)
        for i, name in enumerate(names, start=1):
            captured = m.group(i)
            if captured is None:
                raise _MatchError(f"structured regex token {token!r} captured None")
            if not self._bind_scalar(env2, name, captured):
                return False

        env.clear()
        env.update(env2)
        return True

    def _match_segment_token(self, token: str, segment: str, env: dict[str, object]) -> bool:
        if self._is_single_capture_token(token):
            name = token[1:]
            self._validate_capture_name(name, token=token)
            return self._bind_scalar(env, name, segment)

        if self._is_regex_token(token):
            return self._match_regex_token(token, segment, env)

        return token == segment

    def _match_pattern(
        self,
        pattern: list[str],
        segments: list[str],
    ) -> Optional[dict[str, object]]:
        def rec(i: int, j: int, env: dict[str, object]) -> Optional[dict[str, object]]:
            if i == len(pattern) and j == len(segments):
                return env

            if i == len(pattern):
                return None

            token = pattern[i]

            if self._is_variadic_capture_token(token):
                name = token[1:]
                self._validate_capture_name(name, token=token)

                # Greedy backtracking: longest match first.
                for k in range(len(segments), j - 1, -1):
                    env2 = dict(env)
                    if not self._bind_variadic(env2, name, segments[j:k]):
                        continue
                    out = rec(i + 1, k, env2)
                    if out is not None:
                        return out
                return None

            if j >= len(segments):
                return None

            env2 = dict(env)
            if not self._match_segment_token(token, segments[j], env2):
                return None

            return rec(i + 1, j + 1, env2)

        return rec(0, 0, {})

    def _interpolate_segment(self, template: str, env: dict[str, object]) -> str:
        def repl(match: re.Match[str]) -> str:
            name = match.group(1)
            if name not in env:
                raise _MatchError(f"unknown interpolation variable: {name}")
            value = env[name]
            if not isinstance(value, str):
                raise _MatchError(
                    f"cannot interpolate non-scalar variable {name!r} into segment {template!r}"
                )
            return value

        return _INTERP_RE.sub(repl, template)

    def _rewrite_name(self, pattern: list[str], env: dict[str, object]) -> str:
        out: list[str] = []

        for token in pattern:
            if self._is_single_capture_token(token) and _INTERP_RE.fullmatch(token) is None:
                name = token[1:]
                self._validate_capture_name(name, token=token)
                if name not in env:
                    raise _MatchError(f"unknown interpolation variable: {name}")
                value = env[name]
                if not isinstance(value, str):
                    raise _MatchError(
                        f"cannot interpolate non-scalar variable {name!r} into segment {token!r}"
                    )
                out.append(value)
                continue

            if self._is_variadic_capture_token(token):
                name = token[1:]
                if name not in env:
                    raise _MatchError(f"unknown variadic variable in output pattern: {name}")
                value = env[name]
                if not isinstance(value, list):
                    raise _MatchError(f"output variable {name!r} is not variadic")
                if not all(isinstance(part, str) for part in value):
                    raise _MatchError(f"output variadic variable {name!r} contains non-string segments")
                out.extend(value)
                continue

            if self._is_regex_token(token):
                raise _MatchError(f"regex token not allowed in structured output pattern: {token!r}")

            out.append(self._interpolate_segment(token, env))

        return self.join_name(out)


__all__ = [
    "_MatchError",
    "StructuredMatch",
    "_StructuredPathMatcher",
]
