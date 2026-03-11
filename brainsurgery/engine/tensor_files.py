from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from safetensors.torch import load_file as load_safetensors_file
from safetensors.torch import save_file as save_safetensors_file


def infer_tensor_file_format(path: Path) -> Literal["numpy", "safetensors", "torch"]:
    suffix = path.suffix.lower()
    if suffix in {".npy", ".npz"}:
        return "numpy"
    if suffix == ".safetensors":
        return "safetensors"
    return "torch"


def load_tensor_from_path(
    path: Path,
    *,
    format: Literal["auto", "numpy", "safetensors", "torch"] = "auto",
) -> torch.Tensor:
    format_name = infer_tensor_file_format(path) if format == "auto" else format

    if format_name == "numpy":
        loaded = np.load(path, allow_pickle=False)
        if isinstance(loaded, np.ndarray):
            return torch.from_numpy(loaded)
        if isinstance(loaded, np.lib.npyio.NpzFile):
            keys = list(loaded.keys())
            if len(keys) != 1:
                raise RuntimeError(
                    "loading tensor from .npz requires exactly one array, or use state_dict load"
                )
            return torch.from_numpy(loaded[keys[0]])
        raise RuntimeError(f"unsupported numpy payload at {path}")

    if format_name == "safetensors":
        loaded = load_safetensors_file(str(path), device="cpu")
        if len(loaded) != 1:
            raise RuntimeError(
                "loading tensor from safetensors requires exactly one tensor, or use state_dict load"
            )
        return next(iter(loaded.values()))

    loaded_obj: Any = torch.load(path, map_location="cpu")
    if torch.is_tensor(loaded_obj):
        return loaded_obj
    if isinstance(loaded_obj, np.ndarray):
        return torch.from_numpy(loaded_obj)
    if isinstance(loaded_obj, dict):
        candidate = loaded_obj.get("state_dict", loaded_obj)
        if isinstance(candidate, dict):
            tensors = [
                (k, v) for k, v in candidate.items() if isinstance(k, str) and torch.is_tensor(v)
            ]
            if len(tensors) == 1:
                return tensors[0][1]
            raise RuntimeError(
                "loading tensor from torch checkpoint requires exactly one tensor, or use state_dict load"
            )
    raise RuntimeError(f"unsupported tensor payload at {path}")


def save_tensor_to_path(
    tensor_name: str,
    tensor: torch.Tensor,
    path: Path,
    *,
    format: Literal["safetensors", "torch", "numpy"],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if format == "torch":
        torch.save(tensor, path)
        return
    if format == "numpy":
        np.save(path, tensor.detach().cpu().numpy())
        return
    save_safetensors_file({tensor_name: tensor}, str(path))
