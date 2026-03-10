from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import typer

from ..transform import (
    StateDictProvider,
    TypedTransform,
    TransformError,
    TransformResult,
    ensure_mapping_payload,
    register_transform,
    require_nonempty_string,
    validate_payload_keys,
)


class PrefixesTransformError(TransformError):
    pass


PrefixesMode = Literal["list", "add", "remove", "rename"]


@dataclass(frozen=True)
class PrefixesSpec:
    mode: PrefixesMode
    alias: str | None = None
    from_alias: str | None = None
    to_alias: str | None = None

    def collect_models(self) -> set[str]:
        return set()


class PrefixesTransform(TypedTransform[PrefixesSpec]):
    name = "prefixes"
    error_type = PrefixesTransformError
    spec_type = PrefixesSpec
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
                from_alias=require_nonempty_string(payload, op_name=self.name, key="from"),
                to_alias=require_nonempty_string(payload, op_name=self.name, key="to"),
            )

        raise PrefixesTransformError("prefixes.mode must be one of: list, add, remove, rename")

    def apply(self, spec: object, provider: StateDictProvider) -> TransformResult:
        typed = self.require_spec(spec)

        if typed.mode == "list":
            aliases = sorted(_list_aliases(provider))
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
            assert typed.from_alias is not None
            assert typed.to_alias is not None
            _rename_alias(provider, source=typed.from_alias, dest=typed.to_alias)
            return TransformResult(name=self.name, count=1)

        raise PrefixesTransformError(f"unsupported prefixes mode: {typed.mode}")

    def infer_output_model(self, spec: object) -> str:
        typed = self.require_spec(spec)
        if typed.mode == "add":
            assert typed.alias is not None
            return typed.alias
        if typed.mode == "rename":
            assert typed.to_alias is not None
            return typed.to_alias
        raise PrefixesTransformError("prefixes does not infer an output model in this mode")


def _require_only_keys(payload: dict[str, object], *, allowed: set[str]) -> None:
    unexpected = set(payload) - allowed
    if unexpected:
        raise PrefixesTransformError(f"prefixes received unknown keys: {sorted(unexpected)}")


def _list_aliases(provider: StateDictProvider) -> set[str]:
    list_aliases = getattr(provider, "list_model_aliases", None)
    if callable(list_aliases):
        aliases = list_aliases()
        if isinstance(aliases, set):
            return aliases
        return set(aliases)

    aliases: set[str] = set()
    for _, mapping in _iter_alias_mappings(provider):
        aliases.update(mapping.keys())
    return aliases


def _iter_alias_mappings(provider: StateDictProvider) -> list[tuple[str, dict[str, object]]]:
    mappings: list[tuple[str, dict[str, object]]] = []
    for attr_name in ("model_paths", "state_dicts", "_state_dicts"):
        value = getattr(provider, attr_name, None)
        if isinstance(value, dict):
            mappings.append((attr_name, value))
    return mappings


def _find_alias_mapping(provider: StateDictProvider, alias: str) -> tuple[str, dict[str, object], object]:
    for attr_name, mapping in _iter_alias_mappings(provider):
        if alias in mapping:
            return attr_name, mapping, mapping[alias]
    raise PrefixesTransformError(f"unknown model prefix: {alias!r}")


def _create_empty_alias(provider: StateDictProvider, alias: str) -> None:
    if alias in _list_aliases(provider):
        raise PrefixesTransformError(f"model prefix already exists: {alias!r}")

    get_or_create = getattr(provider, "get_or_create_alias_state_dict", None)
    if callable(get_or_create):
        get_or_create(alias)
        return

    mappings = _iter_alias_mappings(provider)
    if not mappings:
        raise PrefixesTransformError("prefixes add requires a provider that supports editable aliases")

    _, mapping = mappings[-1]
    mapping[alias] = _new_empty_state_dict(mappings)


def _new_empty_state_dict(mappings: list[tuple[str, dict[str, object]]]) -> object:
    for _, mapping in mappings:
        for value in mapping.values():
            state_dict_type = type(value)
            try:
                return state_dict_type()
            except Exception:
                continue
    return {}


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


def _unit_test_prefixes_compile_accepts_none_and_empty_mapping() -> None:
    transform = PrefixesTransform()
    assert isinstance(transform.compile(None, default_model=None), PrefixesSpec)
    assert isinstance(transform.compile({}, default_model=None), PrefixesSpec)


def _unit_test_prefixes_compile_add_mode() -> None:
    spec = PrefixesTransform().compile(
        {"mode": "add", "alias": "scratch"},
        default_model=None,
    )
    assert spec.mode == "add"
    assert spec.alias == "scratch"


def _unit_test_prefixes_compile_rejects_invalid_mode() -> None:
    try:
        PrefixesTransform().compile({"mode": "explode"}, default_model=None)
    except PrefixesTransformError as exc:
        assert "must be one of" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected prefixes mode error")


def _unit_test_prefixes_list_aliases_from_provider_state() -> None:
    class _Provider:
        model_paths = {"base": object()}
        state_dicts = {"scratch": object()}

    assert _list_aliases(_Provider()) == {"base", "scratch"}


__unit_tests__ = [
    _unit_test_prefixes_compile_accepts_none_and_empty_mapping,
    _unit_test_prefixes_compile_add_mode,
    _unit_test_prefixes_compile_rejects_invalid_mode,
    _unit_test_prefixes_list_aliases_from_provider_state,
]


register_transform(PrefixesTransform())
