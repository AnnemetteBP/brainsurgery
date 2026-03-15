from dataclasses import dataclass


@dataclass(frozen=True)
class _WebRunResult:
    ok: bool
    logs: list[str]
    output_lines: list[str]
    summary_yaml: str | None
    written_path: str | None
    error: str | None = None
