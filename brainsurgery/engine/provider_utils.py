from collections.abc import Mapping

from ..core import StateDictLike, StateDictProvider, TransformError


def _is_base_provider_instance(provider: object) -> bool:
    try:
        from .providers import BaseStateDictProvider
    except Exception:
        return False
    return isinstance(provider, BaseStateDictProvider)


def iter_alias_mappings(provider: StateDictProvider) -> list[tuple[str, dict[str, object]]]:
    mappings: list[tuple[str, dict[str, object]]] = []
    for attr_name in ("model_paths", "state_dicts", "_state_dicts"):
        value = getattr(provider, attr_name, None)
        if isinstance(value, dict):
            mappings.append((attr_name, value))
    return mappings


def list_model_aliases(provider: StateDictProvider | None) -> set[str]:
    if provider is None:
        return set()

    if _is_base_provider_instance(provider):
        return provider.list_model_aliases()

    list_aliases = getattr(provider, "list_model_aliases", None)
    if callable(list_aliases):
        aliases = list_aliases()
        if isinstance(aliases, set):
            return {str(alias) for alias in aliases}
        return {str(alias) for alias in aliases}

    aliases: set[str] = set()
    for _, mapping in iter_alias_mappings(provider):
        aliases.update(str(alias) for alias in mapping.keys())
    return aliases


def _has_model_alias(provider: StateDictProvider, alias: str) -> bool:
    if _is_base_provider_instance(provider):
        return provider.has_model_alias(alias)
    return alias in list_model_aliases(provider)


def get_or_create_alias_state_dict(
    provider: StateDictProvider,
    alias: str,
    *,
    error_type: type[TransformError],
    op_name: str,
) -> StateDictLike:
    if _is_base_provider_instance(provider):
        return provider.get_or_create_alias_state_dict(alias)
    if _has_model_alias(provider, alias):
        return provider.get_state_dict(alias)
    raise error_type(f"{op_name} requires a provider that supports creating new aliases")


def list_loaded_tensor_names(provider: StateDictProvider | None) -> dict[str, set[str]]:
    if provider is None:
        return {}

    if _is_base_provider_instance(provider):
        items: Mapping[str, StateDictLike] = provider.state_dicts
    else:
        state_dicts = getattr(provider, "state_dicts", None)
        if not isinstance(state_dicts, dict):
            return {}
        items = state_dicts

    loaded: dict[str, set[str]] = {}
    for alias, state_dict in items.items():
        keys = getattr(state_dict, "keys", None)
        if not callable(keys):
            continue
        try:
            loaded[str(alias)] = {str(name) for name in keys()}
        except Exception:
            continue
    return loaded


def resolve_single_model_alias(
    provider: StateDictProvider,
    *,
    error_type: type[TransformError],
    op_name: str,
) -> str:
    aliases = list_model_aliases(provider)
    if len(aliases) != 1:
        raise error_type(f"{op_name}.alias is required when more than one model alias is available")
    return next(iter(aliases))


def find_alias_mapping(
    provider: StateDictProvider,
    alias: str,
    *,
    error_type: type[TransformError],
) -> tuple[str, dict[str, object], object]:
    for attr_name, mapping in iter_alias_mappings(provider):
        if alias in mapping:
            return attr_name, mapping, mapping[alias]
    raise error_type(f"unknown model prefix: {alias!r}")


def new_empty_state_dict(mappings: list[tuple[str, dict[str, object]]]) -> object:
    for _, mapping in mappings:
        for value in mapping.values():
            state_dict_type = type(value)
            try:
                return state_dict_type()
            except Exception:
                continue
    return {}


__all__ = [
    "iter_alias_mappings",
    "list_model_aliases",
    "get_or_create_alias_state_dict",
    "resolve_single_model_alias",
    "find_alias_mapping",
    "new_empty_state_dict",
]
