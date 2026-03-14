from pathlib import Path
from typing import Literal

import torch as torch_lib

from .dcp import (
    _detect_layout as detect_torch_distributed_checkpoint_layout,
)
from .dcp import (
    _is_checkpoint_directory as is_torch_distributed_checkpoint_directory,
)
from .dcp import (
    _is_full_tensor_storage_metadata as is_full_torch_distributed_tensor_storage_metadata,
)
from .dcp import (
    _load_state_dict_direct as load_torch_distributed_checkpoint_state_dict_direct,
)
from .dcp import (
    _load_state_dict_via_conversion as load_torch_distributed_checkpoint_state_dict_via_conversion,
)
from .dcp import (
    _resolve_output_directory as resolve_torch_distributed_checkpoint_output_directory,
)
from .dcp import (
    _save_state_dict as save_torch_distributed_checkpoint_state_dict,
)
from .npy import _load_single_tensor as load_numpy_tensor
from .npy import _save_single_tensor as save_numpy_tensor
from .safetensors import _load_single_tensor as load_safetensors_tensor
from .safetensors import _load_state_dict as load_safetensors_state_dict
from .safetensors import _save_single_tensor as save_safetensors_tensor
from .safetensors import _save_state_dict as save_safetensors_state_dict
from .torch import _load_object as load_torch_object
from .torch import _load_single_tensor as load_torch_tensor
from .torch import _load_state_dict as load_torch_state_dict
from .torch import _save_single_tensor as save_torch_tensor
from .torch import _save_state_dict as save_torch_state_dict


TensorFileFormat = Literal["numpy", "safetensors", "torch"]
TensorLoadFormat = Literal["auto", "numpy", "safetensors", "torch"]


def infer_tensor_file_format(path: Path) -> TensorFileFormat:
    suffix = path.suffix.lower()
    if suffix in {".npy", ".npz"}:
        return "numpy"
    if suffix == ".safetensors":
        return "safetensors"
    return "torch"


def load_tensor_from_path(
    path: Path,
    *,
    format: TensorLoadFormat = "auto",
) -> torch_lib.Tensor:
    format_name = infer_tensor_file_format(path) if format == "auto" else format

    if format_name == "numpy":
        return load_numpy_tensor(path)
    if format_name == "safetensors":
        return load_safetensors_tensor(path)
    return load_torch_tensor(path)


def save_tensor_to_path(
    tensor_name: str,
    tensor: torch_lib.Tensor,
    path: Path,
    *,
    format: Literal["safetensors", "torch", "numpy"],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if format == "torch":
        save_torch_tensor(tensor, path)
        return
    if format == "numpy":
        save_numpy_tensor(tensor, path)
        return
    save_safetensors_tensor(tensor_name, tensor, path)


__all__ = [
    "detect_torch_distributed_checkpoint_layout",
    "is_torch_distributed_checkpoint_directory",
    "load_safetensors_state_dict",
    "load_tensor_from_path",
    "load_torch_distributed_checkpoint_state_dict_direct",
    "load_torch_distributed_checkpoint_state_dict_via_conversion",
    "load_torch_state_dict",
    "resolve_torch_distributed_checkpoint_output_directory",
    "save_safetensors_state_dict",
    "save_tensor_to_path",
    "save_torch_distributed_checkpoint_state_dict",
    "save_torch_state_dict",
]
