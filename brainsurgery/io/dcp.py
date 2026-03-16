import tempfile
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Literal

import torch


def _is_checkpoint_directory(path: Path) -> bool:
    metadata_file = path / ".metadata"
    has_distcp_shards = any(path.glob("*.distcp"))
    return metadata_file.exists() and has_distcp_shards


def _resolve_output_directory(path: Path) -> Path:
    if path.exists() and not path.is_dir():
        raise RuntimeError("dcp output requires a directory path, not a file")
    if path.suffix != "":
        raise RuntimeError("dcp output requires a directory-style path with no file extension")
    return path


def _save_state_dict(state_dict: dict[str, torch.Tensor], output_dir: Path) -> None:
    try:
        from torch.distributed.checkpoint import save as save_dcp
    except Exception as exc:
        raise RuntimeError(
            "torch distributed checkpoint support requires torch.distributed.checkpoint"
        ) from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    with _suppress_warnings():
        save_dcp(state_dict, checkpoint_id=output_dir, no_dist=True)


def _detect_layout(path: Path) -> Literal["full", "sharded", "mixed", "unknown"]:
    try:
        from torch.distributed.checkpoint import FileSystemReader
        from torch.distributed.checkpoint.metadata import (
            BytesStorageMetadata,
            TensorStorageMetadata,
        )
    except Exception:
        return "unknown"

    try:
        metadata = FileSystemReader(str(path)).read_metadata()
        entries = metadata.state_dict_metadata
    except Exception:
        return "unknown"

    if not isinstance(entries, dict) or not entries:
        return "unknown"

    saw_full = False
    saw_sharded = False

    for entry in entries.values():
        if isinstance(entry, TensorStorageMetadata):
            if _is_full_tensor_storage_metadata(entry):
                saw_full = True
            else:
                saw_sharded = True
        elif isinstance(entry, BytesStorageMetadata):
            saw_sharded = True
        else:
            saw_sharded = True

        if saw_full and saw_sharded:
            return "mixed"

    if saw_sharded:
        return "sharded"
    return "full"


def _is_full_tensor_storage_metadata(entry: Any) -> bool:
    chunks = getattr(entry, "chunks", None)
    size_tuple = tuple(getattr(entry, "size", ()))
    if not isinstance(chunks, list) or len(chunks) != 1:
        return False

    chunk = chunks[0]
    offsets = tuple(getattr(chunk, "offsets", ()))
    sizes = tuple(getattr(chunk, "sizes", ()))
    return offsets == tuple(0 for _ in size_tuple) and sizes == size_tuple


def _load_state_dict_direct(path: Path) -> dict[str, torch.Tensor]:
    from torch.distributed.checkpoint import FileSystemReader
    from torch.distributed.checkpoint import load as load_dcp
    from torch.distributed.checkpoint.metadata import TensorStorageMetadata

    reader = FileSystemReader(str(path))
    metadata = reader.read_metadata()
    entries = metadata.state_dict_metadata
    if not isinstance(entries, dict):
        raise RuntimeError(f"invalid torch distributed checkpoint metadata at {path}")

    loaded: dict[str, torch.Tensor] = {}
    for key, entry in entries.items():
        if not isinstance(key, str):
            raise RuntimeError(
                f"invalid torch distributed checkpoint tensor key in {path}: {key!r}"
            )
        if not isinstance(entry, TensorStorageMetadata):
            raise RuntimeError(f"torch distributed checkpoint contains non-tensor entry: {key!r}")
        properties = getattr(entry, "properties", None)
        dtype = getattr(properties, "dtype", None)
        if not isinstance(dtype, torch.dtype):
            raise RuntimeError(
                f"torch distributed checkpoint tensor entry is missing dtype: {key!r}"
            )
        loaded[key] = torch.empty(tuple(entry.size), dtype=dtype, device="cpu")

    with _suppress_warnings():
        load_dcp(loaded, storage_reader=reader, no_dist=True)

    return _validate_state_dict_mapping(loaded, path)


def _load_state_dict_via_conversion(path: Path) -> tuple[dict[str, torch.Tensor], bool]:
    try:
        from torch.distributed.checkpoint.format_utils import dcp_to_torch_save
    except Exception as exc:
        raise RuntimeError(
            "torch distributed checkpoint support requires torch.distributed.checkpoint.format_utils"
        ) from exc

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp_file:
        tmp_path = Path(tmp_file.name)
    try:
        with _suppress_warnings():
            dcp_to_torch_save(path, tmp_path)
        loaded = torch.load(tmp_path, map_location="cpu")
        wrapped = False
        if (
            isinstance(loaded, dict)
            and "state_dict" in loaded
            and isinstance(loaded["state_dict"], dict)
        ):
            wrapped = True
            loaded = loaded["state_dict"]
        return _validate_state_dict_mapping(loaded, path), wrapped
    finally:
        tmp_path.unlink(missing_ok=True)


@contextmanager
def _suppress_warnings():
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=(
                r"torch\.distributed is disabled, unavailable or uninitialized, "
                r"assuming the intent is to (save|load) in a single process\."
            ),
            category=UserWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message=r"TypedStorage is deprecated\..*",
            category=UserWarning,
        )
        yield


def _validate_state_dict_mapping(loaded: object, path: Path) -> dict[str, torch.Tensor]:
    if not isinstance(loaded, dict):
        raise RuntimeError(f"checkpoint at {path} is not a state_dict mapping")
    if not all(isinstance(k, str) and torch.is_tensor(v) for k, v in loaded.items()):
        raise RuntimeError(f"checkpoint at {path} is not a plain tensor state_dict")
    return dict(loaded)
