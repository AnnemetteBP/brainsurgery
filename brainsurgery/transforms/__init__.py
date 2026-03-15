from importlib import import_module
from pkgutil import iter_modules


def _discovered_module_names() -> list[str]:
    module_names = sorted(
        module_info.name
        for module_info in iter_modules(__path__)  # type: ignore[name-defined]
        if not module_info.name.startswith("_")
    )
    duplicates = sorted({name for name in module_names if module_names.count(name) > 1})
    if duplicates:
        names = ", ".join(duplicates)
        raise RuntimeError(f"Duplicate transform module names discovered in {__name__}: {names}")
    return module_names


def _load_discovered_modules() -> None:
    for module_name in _discovered_module_names():
        qualified_name = f"{__name__}.{module_name}"
        try:
            import_module(qualified_name)
        except Exception as exc:
            raise RuntimeError(f"Failed to import discovered transform module: {qualified_name}") from exc


_load_discovered_modules()
