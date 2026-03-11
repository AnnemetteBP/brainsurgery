from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import torch

from .arena import ProviderError, SegmentedFileBackedArena, TensorSlot
from ..engine.model import (
    load_state_dict_from_path,
    persist_state_dict,
    resolve_output_destination,
    parse_shard_size,
)
from ..engine.plan import SurgeryPlan
from ..engine.output_model import infer_output_model
from ..core import StateDictLike

logger = logging.getLogger("brainsurgery")


@dataclass
class TensorAccessCounts:
    reads: int = 0
    writes: int = 0


# ============================================================
# State-dict implementations
# ============================================================


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
    def __init__(self):
        super().__init__()

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


# ============================================================
# Providers
# ============================================================


class BaseStateDictProvider:
    def __init__(self, model_paths: Dict[str, Path], max_io_workers: int):
        self.model_paths = model_paths
        self.max_io_workers = max_io_workers
        self.state_dicts: Dict[str, StateDictLike] = {}

    def get_state_dict(self, model: str) -> StateDictLike:
        raise NotImplementedError

    def create_state_dict(self) -> StateDictLike:
        raise NotImplementedError

    def list_model_aliases(self) -> set[str]:
        return set(self.model_paths) | set(self.state_dicts)

    def has_model_alias(self, model: str) -> bool:
        return model in self.list_model_aliases()

    def attach_state_dict(self, model: str, state_dict: StateDictLike) -> None:
        self.state_dicts[model] = state_dict
        self.model_paths.pop(model, None)

    def load_state_dict_from_checkpoint_path(self, path: Path) -> StateDictLike:
        state_dict = self.create_state_dict()
        load_state_dict_from_path(path, state_dict, max_io_workers=self.max_io_workers)
        return state_dict

    def load_alias_from_path(self, model: str, path: Path) -> StateDictLike:
        if self.has_model_alias(model):
            raise ProviderError(f"model alias already exists: {model!r}")
        state_dict = self.load_state_dict_from_checkpoint_path(path)
        self.attach_state_dict(model, state_dict)
        return state_dict

    def get_or_create_alias_state_dict(self, model: str) -> StateDictLike:
        if self.has_model_alias(model):
            return self.get_state_dict(model)
        state_dict = self.create_state_dict()
        self.attach_state_dict(model, state_dict)
        return state_dict

    def _get_or_load_state_dict(
        self,
        model: str,
        *,
        loaded_log_message: str,
    ) -> StateDictLike:
        if model in self.state_dicts:
            return self.state_dicts[model]

        if model not in self.model_paths:
            raise ProviderError(f"unknown model alias: {model!r}")

        path = self.model_paths[model]
        logger.info("Opening cranium for brain '%s' at %s", model, path)

        sd = self.load_state_dict_from_checkpoint_path(path)
        self.state_dicts[model] = sd
        logger.info(loaded_log_message, model, len(sd))

        return sd

    def close(self) -> None:
        pass

    def save_output(
        self,
        plan: SurgeryPlan,
        *,
        default_shard_size: str,
        max_io_workers: int,
    ) -> Path:
        if plan.output is None:
            raise ProviderError("save_output requires plan.output")

        output_model = infer_output_model(plan, self)
        state_dict = self.get_state_dict(output_model)

        output_path, output_format, shard_size = resolve_output_destination(
            plan.output,
            default_shard_size=default_shard_size,
        )

        logger.info(
            "Closing incision and preserving brain '%s' to %s (%s)",
            output_model,
            output_path,
            output_format,
        )

        written_path = persist_state_dict(
            dict(state_dict.items()),
            output_path=output_path,
            output_format=output_format,
            shard_size=shard_size,
            sharded_output_root=plan.output.path,
            max_io_workers=max_io_workers,
        )
        if shard_size is None:
            logger.info("Patient stable. Preserved %d tensors at %s", len(state_dict), written_path)
        else:
            logger.info(
                "Patient stable. Wrote %d tensors across sharded safetensors in %s",
                len(state_dict),
                written_path,
            )
        return written_path


class InMemoryStateDictProvider(BaseStateDictProvider):
    def __init__(self, model_paths: Dict[str, Path], max_io_workers: int):
        super().__init__(model_paths, max_io_workers=max_io_workers)

    def get_state_dict(self, model: str) -> InMemoryStateDict:
        state_dict = self._get_or_load_state_dict(
            model,
            loaded_log_message="Brain '%s' exposed: %d tensors laid out on the operating table",
        )
        assert isinstance(state_dict, InMemoryStateDict)
        return state_dict

    def create_state_dict(self) -> InMemoryStateDict:
        return InMemoryStateDict()


class ArenaStateDictProvider(BaseStateDictProvider):
    def __init__(
        self,
        model_paths: Dict[str, Path],
        *,
        arena: SegmentedFileBackedArena,
        max_io_workers: int,
    ):
        super().__init__(model_paths, max_io_workers=max_io_workers)
        self.arena = arena

    def close(self) -> None:
        self.arena.close()

    def get_state_dict(self, model: str) -> ArenaStateDict:
        state_dict = self._get_or_load_state_dict(
            model,
            loaded_log_message=(
                "Brain '%s' transferred to surgical arena: %d tensors laid out on the operating table"
            ),
        )
        assert isinstance(state_dict, ArenaStateDict)
        return state_dict

    def create_state_dict(self) -> ArenaStateDict:
        return ArenaStateDict(self.arena)


def create_state_dict_provider(
    *,
    provider: str,
    model_paths: Dict[str, Path],
    max_io_workers: int,
    arena_root: Path,
    arena_segment_size: str,
) -> BaseStateDictProvider:
    provider_name = provider.strip().lower()

    if provider_name == "inmemory":
        return InMemoryStateDictProvider(
            model_paths,
            max_io_workers=max_io_workers,
        )

    if provider_name == "arena":
        segment_size_bytes = parse_shard_size(arena_segment_size)
        if segment_size_bytes is None:
            raise ProviderError("arena-segment-size must not be 'none'")

        arena = SegmentedFileBackedArena(
            arena_root,
            segment_size_bytes=segment_size_bytes,
        )
        return ArenaStateDictProvider(
            model_paths,
            arena=arena,
            max_io_workers=max_io_workers,
        )

    raise ProviderError("provider must be either 'inmemory' or 'arena'")
