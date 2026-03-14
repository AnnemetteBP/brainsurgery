import json
import logging
from contextlib import nullcontext
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Literal, TypeVar

import torch

from ..core import StateDictLike
from .. import io as io_api
from .output_paths import resolve_sharded_output_directory
from .workers import choose_num_io_workers, run_threadpool_tasks_with_progress


try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    T = TypeVar("T")

    class _TqdmDummy:
        def __init__(self, iterable=None, total=None, **_):
            self.iterable = iterable

        def __iter__(self):
            if self.iterable is None:
                return iter(())
            return iter(self.iterable)

        def update(self, *_):
            pass

        def close(self):
            pass

    def tqdm(iterable=None, **kwargs):  # type: ignore
        return _TqdmDummy(iterable, **kwargs)


logger = logging.getLogger("brainsurgery")


def tensor_nbytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def shard_state_dict(
    state_dict: Dict[str, torch.Tensor],
    max_shard_size: int,
) -> list[dict[str, torch.Tensor]]:
    if max_shard_size <= 0:
        raise RuntimeError("max_shard_size must be positive")

    shards: list[dict[str, torch.Tensor]] = []
    current_shard: dict[str, torch.Tensor] = {}
    current_size = 0

    for key, tensor in state_dict.items():
        size = tensor_nbytes(tensor)

        if size > max_shard_size:
            if current_shard:
                shards.append(current_shard)
                current_shard = {}
                current_size = 0
            shards.append({key: tensor})
            continue

        if current_shard and current_size + size > max_shard_size:
            shards.append(current_shard)
            current_shard = {}
            current_size = 0

        current_shard[key] = tensor
        current_size += size

    if current_shard:
        shards.append(current_shard)

    return shards


def save_sharded_safetensors(
    state_dict: Dict[str, torch.Tensor],
    output_dir: Path,
    max_shard_size: int,
    *,
    max_io_workers: int,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    shards = shard_state_dict(state_dict, max_shard_size)
    total_size = sum(tensor_nbytes(tensor) for tensor in state_dict.values())
    total_shards = len(shards)

    logger.info(
        "Dividing preserved brain into %d safetensor shard(s) with maximum segment size %d bytes",
        total_shards,
        max_shard_size,
    )

    shard_infos: list[tuple[int, str, Path, dict[str, torch.Tensor]]] = []
    weight_map: dict[str, str] = {}

    for shard_index, shard in enumerate(shards, start=1):
        shard_name = f"model-{shard_index:05d}-of-{total_shards:05d}.safetensors"
        shard_path = output_dir / shard_name
        shard_infos.append((shard_index, shard_name, shard_path, shard))
        for key in shard:
            weight_map[key] = shard_name
    if len(weight_map) != len(state_dict):
        raise RuntimeError(
            "internal error while writing sharded safetensors: shard index coverage mismatch"
        )

    num_workers = choose_num_io_workers(total_shards, max_io_workers=max_io_workers)
    logger.info("Dispatching %d orderly thread(s) for preservation", num_workers)

    def _save_one_shard(
        item: tuple[int, str, Path, dict[str, torch.Tensor]],
    ) -> tuple[int, str]:
        shard_index, shard_name, shard_path, shard = item
        _save_safetensors_shard(shard_path, shard)
        return shard_index, shard_name

    def _on_shard_saved(
        item: tuple[int, str, Path, dict[str, torch.Tensor]],
        result: tuple[int, str],
    ) -> None:
        del item
        shard_index, shard_name = result
        logger.debug("Preserved shard %d/%d at %s", shard_index, total_shards, shard_name)

    run_threadpool_tasks_with_progress(
        items=shard_infos,
        worker=_save_one_shard,
        num_workers=num_workers,
        total=total_shards,
        progress_desc="Shard save",
        progress_unit="shard",
        progress_factory=tqdm,
        on_result=_on_shard_saved,
    )

    index = {
        "metadata": {"total_size": total_size},
        "weight_map": weight_map,
    }
    index_path = output_dir / "model.safetensors.index.json"
    index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")
    return index_path


def _save_safetensors_shard(path: Path, shard: dict[str, torch.Tensor]) -> None:
    io_api.save_safetensors_state_dict(shard, path)


def save_state_dict_to_path(
    state_dict: Dict[str, torch.Tensor],
    path: Path,
    *,
    format: Literal["safetensors", "torch", "dcp"],
) -> None:
    if format == "dcp":
        output_dir = resolve_dcp_output_directory(path)
        save_state_dict_to_torch_distributed_checkpoint(state_dict, output_dir)
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    if format == "torch":
        io_api.save_torch_state_dict(state_dict, path)
        return
    io_api.save_safetensors_state_dict(state_dict, path)


def persist_state_dict(
    state_dict: Dict[str, torch.Tensor],
    *,
    output_path: Path,
    output_format: Literal["safetensors", "torch", "dcp"],
    shard_size: int | None,
    sharded_output_root: Path,
    max_io_workers: int,
) -> Path:
    if output_format == "torch":
        save_state_dict_to_path(state_dict, output_path, format="torch")
        return output_path

    if output_format == "dcp":
        save_state_dict_to_path(state_dict, output_path, format="dcp")
        return output_path

    if shard_size is None:
        save_state_dict_to_path(state_dict, output_path, format="safetensors")
        return output_path

    output_dir = resolve_sharded_output_directory(sharded_output_root, output_path)
    return save_sharded_safetensors(
        state_dict,
        output_dir,
        shard_size,
        max_io_workers=max_io_workers,
    )


def resolve_dcp_output_directory(path: Path) -> Path:
    return io_api.resolve_torch_distributed_checkpoint_output_directory(path)


def save_state_dict_to_torch_distributed_checkpoint(
    state_dict: Dict[str, torch.Tensor],
    output_dir: Path,
) -> None:
    io_api.save_torch_distributed_checkpoint_state_dict(state_dict, output_dir)


def load_state_dict_from_path(path: Path, global_state_dict: StateDictLike, *, max_io_workers: int) -> None:
    if not path.exists():
        raise RuntimeError(f"checkpoint path does not exist: {path}")
    if path.is_dir():
        logger.info("CT scan shows a model directory at %s", path)
        return load_state_dict_from_directory(path, global_state_dict, max_io_workers=max_io_workers)
    logger.info("CT scan shows a single checkpoint file at %s", path)
    return load_state_dict_from_file(path, global_state_dict)


def load_state_dict_from_directory(path: Path, global_state_dict: StateDictLike, *, max_io_workers: int) -> None:
    if is_torch_distributed_checkpoint_directory(path):
        logger.info("Detected torch distributed checkpoint directory at %s", path)
        return load_state_dict_from_torch_distributed_checkpoint(path, global_state_dict)

    pt_files = sorted(path.glob("*.pt")) + sorted(path.glob("*.pth")) + sorted(path.glob("*.bin"))
    safetensor_files = sorted(path.glob("*.safetensors"))
    index_file = path / "model.safetensors.index.json"

    if pt_files and safetensor_files:
        raise RuntimeError(
            f"model directory contains both torch and safetensors files; refusing ambiguous load: {path}"
        )

    if safetensor_files:
        if index_file.exists():
            logger.info("Detected safetensors index at %s", index_file)
            files = resolve_safetensor_shards_from_index(index_file, path)
        else:
            logger.info("No safetensors index found; exposing all safetensors shards")
            files = safetensor_files
    else:
        files = pt_files

    if not files:
        raise RuntimeError(f"no supported checkpoint files found in model directory: {path}")

    logger.info("Located %d checkpoint shard(s) in %s", len(files), path)

    num_workers = choose_num_io_workers(len(files), max_io_workers=max_io_workers)
    logger.info("Dispatching %d orderly thread(s) for exposure", num_workers)

    merge_lock = Lock()
    initial_count = len(global_state_dict)
    loaded_counts: list[int] = []

    def _load_one_file(file_path: Path) -> int:
        return load_state_dict_from_file(file_path, global_state_dict, merge_lock)

    def _on_file_loaded(file_path: Path, loaded_count: int) -> None:
        del file_path
        loaded_counts.append(loaded_count)

    run_threadpool_tasks_with_progress(
        items=files,
        worker=_load_one_file,
        num_workers=num_workers,
        total=len(files),
        progress_desc=f"Open {path.name}",
        progress_unit="file",
        progress_factory=tqdm,
        on_result=_on_file_loaded,
    )

    if len(loaded_counts) != len(files):
        raise RuntimeError(
            "internal error while loading checkpoint shards: progress count mismatch"
        )
    if any(count <= 0 for count in loaded_counts):
        raise RuntimeError("checkpoint shard file contained zero tensors")
    loaded_total = sum(loaded_counts)
    final_count = len(global_state_dict)
    if final_count - initial_count != loaded_total:
        raise RuntimeError(
            "internal error while loading checkpoint shards: merged tensor count mismatch"
        )

    logger.info("Cranial assembly complete for %s: %d tensor(s)", path, len(global_state_dict))


def is_torch_distributed_checkpoint_directory(path: Path) -> bool:
    return io_api.is_torch_distributed_checkpoint_directory(path)


def load_state_dict_from_torch_distributed_checkpoint(
    path: Path,
    global_state_dict: StateDictLike,
) -> None:
    layout = detect_torch_distributed_checkpoint_layout(path)
    logger.info("Torch distributed checkpoint layout at %s appears to be: %s", path, layout)
    try:
        loaded = _load_state_dict_from_torch_distributed_checkpoint_direct(path)
    except Exception as direct_exc:
        logger.info(
            "Direct torch distributed checkpoint load failed for %s; falling back to conversion: %s",
            path,
            direct_exc,
        )
        try:
            loaded = _load_state_dict_from_torch_distributed_checkpoint_via_conversion(path)
        except Exception as exc:
            raise RuntimeError(f"failed to load torch distributed checkpoint from {path}") from exc

    loaded_count = _merge_loaded_state_dict(loaded, global_state_dict, path=path)
    if loaded_count <= 0:
        raise RuntimeError("torch distributed checkpoint contained zero tensors")

    logger.info("Cranial assembly complete for %s: %d tensor(s)", path, len(global_state_dict))


def detect_torch_distributed_checkpoint_layout(path: Path) -> Literal["full", "sharded", "mixed", "unknown"]:
    return io_api.detect_torch_distributed_checkpoint_layout(path)


def _is_full_tensor_storage_metadata(entry: Any) -> bool:
    return io_api.is_full_torch_distributed_tensor_storage_metadata(entry)


def _load_state_dict_from_torch_distributed_checkpoint_direct(path: Path) -> Dict[str, torch.Tensor]:
    return io_api.load_torch_distributed_checkpoint_state_dict_direct(path)


def _load_state_dict_from_torch_distributed_checkpoint_via_conversion(path: Path) -> Dict[str, torch.Tensor]:
    loaded, wrapped = io_api.load_torch_distributed_checkpoint_state_dict_via_conversion(path)
    if wrapped:
        logger.info("Detected wrapped state_dict payload while converting DCP in %s", path)
    return loaded


def resolve_safetensor_shards_from_index(index_file: Path, base_dir: Path) -> list[Path]:
    try:
        index_data = json.loads(index_file.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"failed to parse safetensors index: {index_file}") from exc

    weight_map = index_data.get("weight_map")
    if not isinstance(weight_map, dict):
        raise RuntimeError(f"invalid safetensors index: missing weight_map in {index_file}")

    shard_names = sorted(set(weight_map.values()))
    shard_paths: list[Path] = []

    for name in shard_names:
        if not isinstance(name, str) or not name:
            raise RuntimeError(
                f"invalid safetensors index: shard name must be a non-empty string in {index_file}"
            )
        shard_rel = Path(name)
        if shard_rel.is_absolute() or ".." in shard_rel.parts:
            raise RuntimeError(
                f"invalid safetensors index: unsafe shard path {name!r} in {index_file}"
            )
        shard_path = base_dir / name
        if not shard_path.exists():
            raise RuntimeError(
                f"safetensors index references missing shard {name!r} in {base_dir}"
            )
        shard_paths.append(shard_path)

    return shard_paths


def load_state_dict_from_file(
    path: Path,
    global_state_dict: StateDictLike,
    merge_lock: Lock | None = None,
) -> int:
    suffix = path.suffix.lower()
    if suffix == ".safetensors":
        logger.info("Using safetensors instruments on %s", path)
        loaded = io_api.load_safetensors_state_dict(path)
    else:
        logger.info("Using torch instruments on %s", path)
        loaded, wrapped = io_api.load_torch_state_dict(path)
        if wrapped:
            logger.info("Detected wrapped state_dict payload in %s", path)
    return _merge_loaded_state_dict(loaded, global_state_dict, path=path, merge_lock=merge_lock)


def _merge_loaded_state_dict(
    loaded: Dict[str, torch.Tensor],
    global_state_dict: StateDictLike,
    *,
    path: Path,
    merge_lock: Lock | None = None,
) -> int:
    loaded_count = 0
    for key, tensor in loaded.items():
        lock_context = merge_lock if merge_lock is not None else nullcontext()
        with lock_context:
            if key in global_state_dict:
                raise RuntimeError(f"duplicate tensor key {key!r} while loading file {path}")
            global_state_dict[key] = tensor
            loaded_count += 1
    return loaded_count


def validate_state_dict_mapping(loaded: object, path: Path) -> Dict[str, torch.Tensor]:
    if not isinstance(loaded, dict):
        raise RuntimeError(f"checkpoint at {path} is not a state_dict mapping")
    if not all(isinstance(k, str) and torch.is_tensor(v) for k, v in loaded.items()):
        raise RuntimeError(f"checkpoint at {path} is not a plain tensor state_dict")
    return dict(loaded)
