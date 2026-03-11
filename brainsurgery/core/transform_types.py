from __future__ import annotations

from abc import abstractmethod
from collections.abc import MutableMapping
from typing import Any, Protocol

import torch


class TransformError(RuntimeError):
    pass


class StateDictLike(MutableMapping[str, torch.Tensor]):
    @abstractmethod
    def slot(self, key: str) -> Any:
        raise NotImplementedError

    @abstractmethod
    def bind_slot(self, key: str, slot: Any) -> None:
        raise NotImplementedError

    @abstractmethod
    def access_counts(self, key: str) -> dict[str, int]:
        raise NotImplementedError

    @abstractmethod
    def mark_write(self, key: str, count: int = 1) -> None:
        raise NotImplementedError


class StateDictProvider(Protocol):
    def get_state_dict(self, model: str) -> StateDictLike:
        ...


def note_tensor_write(state_dict: object, key: str, count: int = 1) -> None:
    mark_write = getattr(state_dict, "mark_write", None)
    if callable(mark_write):
        mark_write(key, count)
