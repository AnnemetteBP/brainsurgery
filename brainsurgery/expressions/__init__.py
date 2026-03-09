from __future__ import annotations

from importlib import import_module
from pkgutil import iter_modules

from ..expression import (
    AssertExpr,
    AssertExprHelp,
    AssertTransformError,
    compile_assert_expr,
    get_assert_expr_help,
    get_assert_expr_names,
    register_assert_expr,
)


for module_info in iter_modules(__path__):  # type: ignore[name-defined]
    module_name = module_info.name
    if module_name.startswith("_"):
        continue
    import_module(f"{__name__}.{module_name}")


__all__ = [
    "AssertExpr",
    "AssertExprHelp",
    "AssertTransformError",
    "compile_assert_expr",
    "get_assert_expr_help",
    "get_assert_expr_names",
    "register_assert_expr",
]
