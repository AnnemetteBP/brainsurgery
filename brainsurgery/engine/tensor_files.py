from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch

from .. import io as io_api


def infer_tensor_file_format(path: Path) -> Literal["numpy", "safetensors", "torch"]:
    return io_api.infer_tensor_file_format(path)


def load_tensor_from_path(
    path: Path,
    *,
    format: Literal["auto", "numpy", "safetensors", "torch"] = "auto",
) -> torch.Tensor:
    return io_api.load_tensor_from_path(path, format=format)


def save_tensor_to_path(
    tensor_name: str,
    tensor: torch.Tensor,
    path: Path,
    *,
    format: Literal["safetensors", "torch", "numpy"],
) -> None:
    io_api.save_tensor_to_path(
        tensor_name,
        tensor,
        path,
        format=format,
    )
