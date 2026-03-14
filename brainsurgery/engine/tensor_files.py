from pathlib import Path
from typing import Literal

import torch

from ..io import load_tensor_from_path as io_load_tensor_from_path
from ..io import save_tensor_to_path as io_save_tensor_to_path


def load_tensor_from_path(
    path: Path,
    *,
    format: Literal["auto", "numpy", "safetensors", "torch"] = "auto",
) -> torch.Tensor:
    return io_load_tensor_from_path(path, format=format)


def save_tensor_to_path(
    tensor_name: str,
    tensor: torch.Tensor,
    path: Path,
    *,
    format: Literal["safetensors", "torch", "numpy"],
) -> None:
    io_save_tensor_to_path(
        tensor_name,
        tensor,
        path,
        format=format,
    )
