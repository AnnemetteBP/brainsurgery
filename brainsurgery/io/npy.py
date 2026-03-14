from pathlib import Path

import numpy as np
import torch


def _load_single_tensor(path: Path) -> torch.Tensor:
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


def _save_single_tensor(tensor: torch.Tensor, path: Path) -> None:
    np.save(path, tensor.detach().cpu().numpy())
