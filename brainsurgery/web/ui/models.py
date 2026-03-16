from dataclasses import asdict, dataclass
from typing import Any, Literal


def model_to_payload(model: Any) -> dict[str, Any]:
    return asdict(model)


@dataclass(frozen=True)
class RuntimeFlagsPayload:
    dry_run: bool
    preview: bool
    verbose: bool


@dataclass(frozen=True)
class ErrorInfoPayload:
    code: str
    message: str
    endpoint: str
    transform: str | None
    exception_type: str
    location: dict[str, Any] | None = None
    context: dict[str, Any] | None = None


@dataclass(frozen=True)
class ErrorResponsePayload:
    ok: Literal[False]
    error: str
    error_info: ErrorInfoPayload


@dataclass(frozen=True)
class TransformsResponsePayload:
    ok: Literal[True]
    transforms: list[dict[str, Any]]


@dataclass(frozen=True)
class StateResponsePayload:
    ok: Literal[True]
    models: list[dict[str, Any]]
    runtime_flags: RuntimeFlagsPayload


@dataclass(frozen=True)
class ProgressResponsePayload:
    ok: Literal[True]
    progress: dict[str, Any]


@dataclass(frozen=True)
class LoadResponsePayload:
    ok: Literal[True]
    models: list[dict[str, Any]]
    runtime_flags: RuntimeFlagsPayload


@dataclass(frozen=True)
class ApplyTransformResponsePayload:
    ok: Literal[True]
    models: list[dict[str, Any]]
    output: str
    runtime_flags: RuntimeFlagsPayload
    preview_confirmation_required: bool
    preview_transform: str | None = None


@dataclass(frozen=True)
class SaveDownloadResponsePayload:
    ok: Literal[True]
    models: list[dict[str, Any]]
    output: str
    runtime_flags: RuntimeFlagsPayload
    download_filename: str
    download_mime: str
    download_b64: str


@dataclass(frozen=True)
class ModelDumpResponsePayload:
    ok: Literal[True]
    dump: str
    matched_count: int
    total_count: int
