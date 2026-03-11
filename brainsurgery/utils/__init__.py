from __future__ import annotations

from .workers import *
from .dtypes import *
from .history import *
from .render import *
from .resolver import *
from .summary import *
from .tensor_checks import *
from .provider_utils import *
from . import workers as _workers
from . import dtypes as _dtypes
from . import history as _history
from . import render as _render
from . import resolver as _resolver
from . import summary as _summary
from . import tensor_checks as _tensor_checks
from . import provider_utils as _provider_utils

_TRANSFORMS_ALL = [
    "Docs",
    "UnaryRefs",
    "BinaryRefs",
    "TernaryRefs",
    "DeclarativeUnaryTransform",
    "DeclarativeBinaryTransform",
    "DeclarativeTernaryTransform",
]

try:
    from .transforms import *
    from . import transforms as _transforms
except ImportError:
    _transforms = None


def __getattr__(name: str) -> object:
    if name in _TRANSFORMS_ALL:
        from . import transforms as _lazy_transforms

        return getattr(_lazy_transforms, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    *_workers.__all__,
    *_dtypes.__all__,
    *_history.__all__,
    *_render.__all__,
    *_resolver.__all__,
    *_summary.__all__,
    *_tensor_checks.__all__,
    *_provider_utils.__all__,
]

if _transforms is not None:
    __all__.extend(_transforms.__all__)
else:
    __all__.extend(_TRANSFORMS_ALL)
