from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Literal

import typer

from ..engine import (
    find_alias_mapping,
    get_or_create_alias_state_dict,
    iter_alias_mappings,
    list_model_aliases,
    new_empty_state_dict,
)
from ..core import TransformError
from ..core import TypedTransform, TransformResult, register_transform
from ..core import ensure_mapping_payload, require_nonempty_string, validate_payload_keys
from ..core import StateDictProvider


class PrefixesTransformError(TransformError):
    pass


PrefixesMode = Literal["list", "add", "remove", "rename"]


@dataclass(frozen=True)
class PrefixesSpec:
    mode: PrefixesMode
    alias: str | None = None
    source_alias: str | None = None
    dest_alias: str | None = None

    def collect_models(self) -> set[str]:
        return set()


class PrefixesTransform(TypedTransform[PrefixesSpec]):
    name = "prefixes"
    error_type = PrefixesTransformError
    spec_type = PrefixesSpec
    completion_requires_payload = False
    allowed_keys = {"mode", "alias", "from", "to"}
    help_text = (
        "Lists or edits the currently available model prefixes (aliases).\n"
        "\n"
        "Modes:\n"
        "  - list (default): show all available aliases\n"
        "  - add: create a new empty alias (`alias`)\n"
        "  - remove: delete an existing alias (`alias`)\n"
        "  - rename: rename an existing alias (`from`, `to`)\n"
        "\n"
        "Examples:\n"
        "  prefixes\n"
        "  prefixes: { mode: list }\n"
        "  prefixes: { mode: add, alias: scratch }\n"
        "  prefixes: { mode: remove, alias: scratch }\n"
        "  prefixes: { mode: rename, from: scratch, to: edited }"
    )

    def compile(self, payload: Any, default_model: str | None) -> PrefixesSpec:
        del default_model

        if payload is None:
            return PrefixesSpec(mode="list")

        if isinstance(payload, dict) and not payload:
            return PrefixesSpec(mode="list")

        payload = ensure_mapping_payload(payload, self.name)
        validate_payload_keys(
            payload,
            op_name=self.name,
            allowed_keys=self.allowed_keys,
        )

        raw_mode = payload.get("mode", "list")
        if not isinstance(raw_mode, str) or not raw_mode:
            raise PrefixesTransformError("prefixes.mode must be a non-empty string when provided")

        mode = raw_mode.strip().lower()
        if mode == "list":
            _require_only_keys(payload, allowed={"mode"})
            return PrefixesSpec(mode="list")

        if mode == "add":
            _require_only_keys(payload, allowed={"mode", "alias"})
            return PrefixesSpec(
                mode="add",
                alias=require_nonempty_string(payload, op_name=self.name, key="alias"),
            )

        if mode == "remove":
            _require_only_keys(payload, allowed={"mode", "alias"})
            return PrefixesSpec(
                mode="remove",
                alias=require_nonempty_string(payload, op_name=self.name, key="alias"),
            )

        if mode == "rename":
            _require_only_keys(payload, allowed={"mode", "from", "to"})
            return PrefixesSpec(
                mode="rename",
                source_alias=require_nonempty_string(payload, op_name=self.name, key="from"),
                dest_alias=require_nonempty_string(payload, op_name=self.name, key="to"),
            )

        raise PrefixesTransformError("prefixes.mode must be one of: list, add, remove, rename")

    def apply(self, spec: object, provider: StateDictProvider) -> TransformResult:
        typed = self.require_spec(spec)

        if typed.mode == "list":
            aliases = sorted(list_model_aliases(provider))
            if aliases:
                typer.echo("Available model prefixes:")
                for alias in aliases:
                    typer.echo(f"  {alias}::")
            else:
                typer.echo("No model prefixes available.")
            return TransformResult(name=self.name, count=len(aliases))

        if typed.mode == "add":
            assert typed.alias is not None
            _create_empty_alias(provider, typed.alias)
            return TransformResult(name=self.name, count=1)

        if typed.mode == "remove":
            assert typed.alias is not None
            _delete_alias(provider, typed.alias)
            return TransformResult(name=self.name, count=1)

        if typed.mode == "rename":
            assert typed.source_alias is not None
            assert typed.dest_alias is not None
            _rename_alias(provider, source=typed.source_alias, dest=typed.dest_alias)
            return TransformResult(name=self.name, count=1)

        raise PrefixesTransformError(f"unsupported prefixes mode: {typed.mode}")

    def infer_output_model(self, spec: object) -> str:
        typed = self.require_spec(spec)
        if typed.mode == "add":
            assert typed.alias is not None
            return typed.alias
        if typed.mode == "rename":
            assert typed.dest_alias is not None
            return typed.dest_alias
        raise PrefixesTransformError("prefixes does not infer an output model in this mode")

    def contributes_output_model(self, spec: object) -> bool:
        typed = self.require_spec(spec)
        return typed.mode in {"add", "rename"}

    def completion_key_candidates(self, before_cursor: str, prefix_text: str) -> list[str] | None:
        used_keys = set(re.findall(r"([A-Za-z_][A-Za-z0-9_]*)\s*:", before_cursor))
        mode = _prefixes_mode(before_cursor)
        if mode is None:
            options = ["mode: "]
        elif mode == "list":
            options = []
        elif mode in {"add", "remove"}:
            options = ["alias: "]
        elif mode == "rename":
            options = ["from: ", "to: "]
        else:
            options = ["mode: "]
        filtered = [
            candidate
            for candidate in options
            if candidate[:-2] not in used_keys and candidate.startswith(prefix_text)
        ]
        if filtered:
            return filtered
        if options and not all(option[:-2] in used_keys for option in options):
            return []
        return ["}"]

    def completion_value_candidates(
        self,
        value_key: str | None,
        prefix_text: str,
        model_aliases: list[str],
    ) -> list[str] | None:
        if value_key == "mode":
            return [
                mode for mode in ("list", "add", "remove", "rename") if mode.startswith(prefix_text)
            ]
        if value_key in {"alias", "from", "to"}:
            return [alias for alias in sorted(model_aliases) if alias.startswith(prefix_text)]
        return None

    def completion_reference_keys(self) -> list[str]:
        return []


def _require_only_keys(payload: dict[str, object], *, allowed: set[str]) -> None:
    unexpected = set(payload) - allowed
    if unexpected:
        raise PrefixesTransformError(f"prefixes received unknown keys: {sorted(unexpected)}")


def _prefixes_mode(before_cursor: str) -> str | None:
    match = re.search(r"\bmode\s*:\s*([A-Za-z_][A-Za-z0-9_]*)", before_cursor)
    if match is None:
        return None
    return match.group(1).lower()


def _list_aliases(provider: StateDictProvider) -> set[str]:
    return list_model_aliases(provider)


def _iter_alias_mappings(provider: StateDictProvider) -> list[tuple[str, dict[str, object]]]:
    return iter_alias_mappings(provider)


def _find_alias_mapping(provider: StateDictProvider, alias: str) -> tuple[str, dict[str, object], object]:
    return find_alias_mapping(provider, alias, error_type=PrefixesTransformError)


def _create_empty_alias(provider: StateDictProvider, alias: str) -> None:
    if alias in _list_aliases(provider):
        raise PrefixesTransformError(f"model prefix already exists: {alias!r}")

    try:
        get_or_create_alias_state_dict(
            provider,
            alias,
            error_type=PrefixesTransformError,
            op_name="prefixes add",
        )
        return
    except PrefixesTransformError:
        pass

    mappings = _iter_alias_mappings(provider)
    if not mappings:
        raise PrefixesTransformError("prefixes add requires a provider that supports editable aliases")

    _, mapping = mappings[-1]
    mapping[alias] = _new_empty_state_dict(mappings)


def _new_empty_state_dict(mappings: list[tuple[str, dict[str, object]]]) -> object:
    return new_empty_state_dict(mappings)


def _delete_alias(provider: StateDictProvider, alias: str) -> None:
    _, mapping, _ = _find_alias_mapping(provider, alias)
    del mapping[alias]


def _rename_alias(provider: StateDictProvider, *, source: str, dest: str) -> None:
    if source == dest:
        raise PrefixesTransformError("prefixes rename source and destination must differ")
    if dest in _list_aliases(provider):
        raise PrefixesTransformError(f"model prefix already exists: {dest!r}")
    _, mapping, value = _find_alias_mapping(provider, source)
    del mapping[source]
    mapping[dest] = value






register_transform(PrefixesTransform())
