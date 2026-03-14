from pathlib import Path
from typing import Any

import numpy as np
import torch


def _load_object(path: Path) -> Any:
    return torch.load(path, map_location="cpu")


def _save_state_dict(state_dict: dict[str, torch.Tensor], path: Path) -> None:
    torch.save(state_dict, path)


def _load_state_dict(path: Path) -> tuple[dict[str, torch.Tensor], bool]:
    loaded = _load_object(path)
    wrapped = False
    if isinstance(loaded, dict) and "state_dict" in loaded and isinstance(loaded["state_dict"], dict):
        wrapped = True
        loaded = loaded["state_dict"]
    return _validate_state_dict_mapping(loaded, path), wrapped


def _save_single_tensor(tensor: torch.Tensor, path: Path) -> None:
    torch.save(tensor, path)


def _load_single_tensor(path: Path) -> torch.Tensor:
    loaded_obj = _load_object(path)
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


def _validate_state_dict_mapping(loaded: object, path: Path) -> dict[str, torch.Tensor]:
    if not isinstance(loaded, dict):
        raise RuntimeError(f"checkpoint at {path} is not a state_dict mapping")
    if not all(isinstance(k, str) and torch.is_tensor(v) for k, v in loaded.items()):
        raise RuntimeError(f"checkpoint at {path} is not a plain tensor state_dict")
    return dict(loaded)
