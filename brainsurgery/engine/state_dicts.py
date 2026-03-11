from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Dict

import torch

from .arena import ProviderError, SegmentedFileBackedArena, TensorSlot
from ..core import StateDictLike


@dataclass
class TensorAccessCounts:
    reads: int = 0
    writes: int = 0


class SlotBackedStateDict(StateDictLike):
    def __init__(self) -> None:
        self._slots: Dict[str, object] = {}
        self._access_counts: Dict[str, TensorAccessCounts] = {}

    def __delitem__(self, key: str) -> None:
        del self._slots[key]
        self._access_counts.pop(key, None)

    def __iter__(self) -> Iterator[str]:
        return iter(self._slots)

    def __len__(self) -> int:
        return len(self._slots)

    def keys(self):
        return self._slots.keys()

    def items(self):
        for key in self._slots:
            yield key, self[key]

    def values(self):
        for key in self._slots:
            yield self[key]

    def access_counts(self, key: str) -> dict[str, int]:
        counts = self._access_counts.get(key)
        if counts is None:
            return {"reads": 0, "writes": 0}
        return {"reads": counts.reads, "writes": counts.writes}

    def mark_write(self, key: str, count: int = 1) -> None:
        if count < 0:
            raise ProviderError("write count increment must be non-negative")
        self._ensure_access_counts(key).writes += count

    def _mark_read(self, key: str, count: int = 1) -> None:
        if count < 0:
            raise ProviderError("read count increment must be non-negative")
        self._ensure_access_counts(key).reads += count

    def _ensure_access_counts(self, key: str) -> TensorAccessCounts:
        if key not in self._slots:
            raise KeyError(key)
        counts = self._access_counts.get(key)
        if counts is None:
            counts = TensorAccessCounts()
            self._access_counts[key] = counts
        return counts


class InMemoryStateDict(SlotBackedStateDict):
    def __getitem__(self, key: str) -> torch.Tensor:
        value = self._slots[key]
        assert isinstance(value, torch.Tensor)
        self._mark_read(key)
        return value

    def __setitem__(self, key: str, value: torch.Tensor) -> None:
        if not torch.is_tensor(value):
            raise ProviderError(f"value for key {key!r} is not a tensor")
        self._slots[key] = value
        self.mark_write(key)

    def slot(self, key: str) -> torch.Tensor:
        value = self._slots[key]
        assert isinstance(value, torch.Tensor)
        return value

    def bind_slot(self, key: str, slot: torch.Tensor) -> None:
        if not torch.is_tensor(slot):
            raise ProviderError(f"slot for key {key!r} is not a tensor")
        self._slots[key] = slot
        self.mark_write(key)


class ArenaStateDict(SlotBackedStateDict):
    def __init__(self, arena: SegmentedFileBackedArena):
        super().__init__()
        self._arena = arena

    def __getitem__(self, key: str) -> torch.Tensor:
        try:
            slot = self._slots[key]
        except KeyError as exc:
            raise KeyError(key) from exc
        assert isinstance(slot, TensorSlot)
        self._mark_read(key)
        return self._arena.tensor_from_slot(slot)

    def __setitem__(self, key: str, value: torch.Tensor) -> None:
        if not torch.is_tensor(value):
            raise ProviderError(f"value for key {key!r} is not a tensor")
        self._slots[key] = self._arena.store_tensor(value)
        self.mark_write(key)

    def slot(self, key: str) -> TensorSlot:
        try:
            return self._slots[key]
        except KeyError as exc:
            raise KeyError(key) from exc

    def bind_slot(self, key: str, slot: TensorSlot) -> None:
        if not isinstance(slot, TensorSlot):
            raise ProviderError(f"slot for key {key!r} is not a TensorSlot")
        self._slots[key] = slot
        self.mark_write(key)
