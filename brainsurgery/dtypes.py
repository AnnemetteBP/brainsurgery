from __future__ import annotations

import torch

from .transform import TransformError


_DTYPE_ALIASES: dict[str, torch.dtype] = {
    "float16": torch.float16,
    "half": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float32": torch.float32,
    "float": torch.float32,
    "fp32": torch.float32,
    "float64": torch.float64,
    "double": torch.float64,
    "fp64": torch.float64,
    "uint8": torch.uint8,
    "int8": torch.int8,
    "int16": torch.int16,
    "short": torch.int16,
    "int32": torch.int32,
    "int": torch.int32,
    "int64": torch.int64,
    "long": torch.int64,
    "bool": torch.bool,
}


def parse_torch_dtype(
    raw: str,
    *,
    error_type: type[TransformError],
    op_name: str,
    field_name: str,
) -> torch.dtype:
    value = raw.strip().lower()
    try:
        return _DTYPE_ALIASES[value]
    except KeyError as exc:
        allowed = ", ".join(sorted(_DTYPE_ALIASES))
        raise error_type(
            f"{op_name}.{field_name} unsupported dtype {raw!r}; expected one of: {allowed}"
        ) from exc

