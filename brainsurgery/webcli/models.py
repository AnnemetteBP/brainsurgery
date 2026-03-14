from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class _WebRunResult:
    ok: bool
    logs: list[str]
    output_lines: list[str]
    executed_transforms: list[dict[str, Any]]
    summary_yaml: str | None
    written_path: str | None
    error: str | None = None

