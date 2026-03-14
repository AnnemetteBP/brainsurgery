import logging
from pathlib import Path
from typing import Dict

from .arena import ProviderError, _SegmentedFileBackedArena
from .checkpoint_io import _load_state_dict_from_path, persist_state_dict
from .output_model import _infer_output_model
from .output_paths import parse_shard_size, _resolve_output_destination
from .plan import SurgeryPlan
from .state_dicts import _ArenaStateDict, _InMemoryStateDict
from ..core import StateDictLike


logger = logging.getLogger("brainsurgery")


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
        _load_state_dict_from_path(path, state_dict, max_io_workers=self.max_io_workers)
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

        output_model = _infer_output_model(plan, self)
        state_dict = self.get_state_dict(output_model)

        output_path, output_format, shard_size = _resolve_output_destination(
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
    def get_state_dict(self, model: str) -> _InMemoryStateDict:
        state_dict = self._get_or_load_state_dict(
            model,
            loaded_log_message="Brain '%s' exposed: %d tensors laid out on the operating table",
        )
        assert isinstance(state_dict, _InMemoryStateDict)
        return state_dict

    def create_state_dict(self) -> _InMemoryStateDict:
        return _InMemoryStateDict()


class ArenaStateDictProvider(BaseStateDictProvider):
    def __init__(
        self,
        model_paths: Dict[str, Path],
        *,
        arena: _SegmentedFileBackedArena,
        max_io_workers: int,
    ):
        super().__init__(model_paths, max_io_workers=max_io_workers)
        self.arena = arena

    def close(self) -> None:
        self.arena.close()

    def get_state_dict(self, model: str) -> _ArenaStateDict:
        state_dict = self._get_or_load_state_dict(
            model,
            loaded_log_message=(
                "Brain '%s' transferred to surgical arena: %d tensors laid out on the operating table"
            ),
        )
        assert isinstance(state_dict, _ArenaStateDict)
        return state_dict

    def create_state_dict(self) -> _ArenaStateDict:
        return _ArenaStateDict(self.arena)


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

        arena = _SegmentedFileBackedArena(
            arena_root,
            segment_size_bytes=segment_size_bytes,
        )
        return ArenaStateDictProvider(
            model_paths,
            arena=arena,
            max_io_workers=max_io_workers,
        )

    raise ProviderError("provider must be either 'inmemory' or 'arena'")
