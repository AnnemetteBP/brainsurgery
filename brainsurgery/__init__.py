from __future__ import annotations

import sys
from importlib import import_module

from . import transforms

_LEGACY_MODULE_ALIASES = {
    "arena": "providers.arena",
    "config": "engine.config",
    "dtypes": "utils.dtypes",
    "execution": "engine.execution",
    "expression": "core.expression",
    "history": "utils.history",
    "interactive": "engine.interactive",
    "mappings": "core.mappings",
    "matching": "core.matching",
    "model": "engine.model",
    "phlora": "core.phlora",
    "plan": "engine.plan",
    "provider_utils": "utils.provider_utils",
    "refs": "core.refs",
    "render": "utils.render",
    "resolver": "utils.resolver",
    "summary": "utils.summary",
    "tensor_checks": "utils.tensor_checks",
    "ternary": "transforms.ternary",
    "transform": "core.transform",
    "transform_types": "core.transform_types",
    "workers": "utils.workers",
}

for legacy_name, target_name in _LEGACY_MODULE_ALIASES.items():
    module = import_module(f"{__name__}.{target_name}")
    sys.modules.setdefault(f"{__name__}.{legacy_name}", module)
    globals().setdefault(legacy_name, module)
