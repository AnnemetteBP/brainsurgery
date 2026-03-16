import re
from pathlib import Path
from typing import Literal

from .plan import _OutputSpec


def _resolve_output_destination(
    output: _OutputSpec,
    *,
    default_shard_size: str,
) -> tuple[Path, Literal["safetensors", "torch", "dcp"], int | None]:
    path = output.path
    format_value = output.format
    shard_size = resolve_shard_size(output, default_shard_size=default_shard_size)

    if format_value is not None:
        resolved_path: Path
        resolved_format: Literal["safetensors", "torch", "dcp"]
        if format_value == "safetensors":
            resolved_path, resolved_format = _resolve_explicit_safetensors_destination(path)
        elif format_value == "torch":
            resolved_path, resolved_format = _resolve_explicit_torch_destination(path)
        elif format_value == "dcp":
            resolved_path, resolved_format = _resolve_explicit_dcp_destination(path)
        else:  # pragma: no cover
            raise RuntimeError(f"unsupported explicit output format: {format_value}")
    else:
        suffix = path.suffix.lower()
        if suffix == ".safetensors":
            resolved_path, resolved_format = path, "safetensors"
        elif suffix in {".pt", ".pth", ".bin"}:
            resolved_path, resolved_format = path, "torch"
        elif path.exists() and path.is_dir():
            resolved_path, resolved_format = path / "model.safetensors", "safetensors"
        elif suffix == "":
            resolved_path, resolved_format = path / "model.safetensors", "safetensors"
        else:
            raise RuntimeError(
                f"unsupported output format for {path}; use a directory, .safetensors, .pt, .pth, or .bin, "
                f"or specify output.format explicitly"
            )

    if shard_size is not None and resolved_format != "safetensors":
        raise RuntimeError("output.shard is only supported for safetensors output")

    return resolved_path, resolved_format, shard_size


def _resolve_explicit_safetensors_destination(path: Path) -> tuple[Path, Literal["safetensors"]]:
    if path.exists() and path.is_dir():
        return path / "model.safetensors", "safetensors"
    if path.suffix == "":
        return path / "model.safetensors", "safetensors"
    if path.suffix.lower() != ".safetensors":
        raise RuntimeError(
            f"output.format='safetensors' is incompatible with file path {path}; "
            f"use a directory or a .safetensors file"
        )
    return path, "safetensors"


def _resolve_explicit_torch_destination(path: Path) -> tuple[Path, Literal["torch"]]:
    if path.exists() and path.is_dir():
        raise RuntimeError("output.format='torch' requires a file path, not a directory")
    if path.suffix.lower() not in {".pt", ".pth", ".bin"}:
        raise RuntimeError(
            f"output.format='torch' requires a .pt, .pth, or .bin file path; got {path}"
        )
    return path, "torch"


def _resolve_explicit_dcp_destination(path: Path) -> tuple[Path, Literal["dcp"]]:
    if path.exists() and not path.is_dir():
        raise RuntimeError("output.format='dcp' requires a directory path, not a file")
    if path.suffix != "":
        raise RuntimeError(
            f"output.format='dcp' requires a directory-style path with no file extension; got {path}"
        )
    return path, "dcp"


def resolve_shard_size(output: _OutputSpec, default_shard_size: str) -> int | None:
    raw = output.shard

    if raw is None:
        if _is_directory_style_output(output):
            raw = default_shard_size
        else:
            return None

    return parse_shard_size(raw)


def _is_directory_style_output(output: _OutputSpec) -> bool:
    path = output.path

    if output.format == "torch":
        return False

    if output.format == "dcp":
        return True

    if output.format == "safetensors":
        if path.exists() and path.is_dir():
            return True
        return path.suffix == ""

    if path.exists() and path.is_dir():
        return True

    return path.suffix == ""


def parse_shard_size(raw: str | None) -> int | None:
    if raw is None or raw == "none":
        return None

    if not isinstance(raw, str) or not raw:
        raise RuntimeError("output.shard must be a non-empty string or 'none'")

    match = re.fullmatch(r"(?i)\s*(\d+)\s*(b|kb|mb|gb|tb)\s*", raw)
    if not match:
        raise RuntimeError(
            f"invalid output.shard value {raw!r}; expected values like 'none', '500MB', '5GB'"
        )

    value = int(match.group(1))
    unit = match.group(2).lower()

    multipliers = {
        "b": 1,
        "kb": 1024,
        "mb": 1024**2,
        "gb": 1024**3,
        "tb": 1024**4,
    }

    return value * multipliers[unit]


def _resolve_sharded_output_directory(original_path: Path, resolved_path: Path) -> Path:
    if original_path.exists() and original_path.is_dir():
        return original_path
    if original_path.suffix == "":
        return original_path
    raise RuntimeError(
        "sharded safetensors output requires a directory-style output path, not a single file"
    )
