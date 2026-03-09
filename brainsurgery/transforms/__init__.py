from importlib import import_module
from pkgutil import iter_modules


for module_info in iter_modules(__path__):  # type: ignore[name-defined]
    module_name = module_info.name
    if module_name.startswith("_"):
        continue
    import_module(f"{__name__}.{module_name}")
