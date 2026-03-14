import logging
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf
import yaml

from ..cli.interactive import normalize_transform_specs
from ..cli.summary import build_raw_plan
from ..engine import (
    compile_plan,
    create_state_dict_provider,
    execute_transform_pairs,
    get_runtime_flags,
    reset_runtime_flags,
    use_output_emitter,
)
from .models import WebRunResult


logger = logging.getLogger("brainsurgery")


def run_web_plan(
    *,
    plan_yaml: str,
    shard_size: str,
    num_workers: int,
    provider: str,
    arena_root: Path,
    arena_segment_size: str,
    summarize: bool,
    log_level: str,
) -> WebRunResult:
    raw = yaml.safe_load(plan_yaml) if plan_yaml.strip() else {}
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("Plan YAML root must be a mapping.")

    configure_logging(log_level=log_level)
    reset_runtime_flags()

    logs: list[str] = []
    output_lines: list[str] = []
    executed_transforms: list[dict[str, Any]] = []
    summary_yaml: str | None = None
    written_path: str | None = None

    log_handler = _ListLogHandler(logs)
    log_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s"))
    root_logger = logging.getLogger()
    root_logger.addHandler(log_handler)
    logger.addHandler(log_handler)

    state_dict_provider = None
    try:
        surgery_plan = compile_plan(raw)
        state_dict_provider = create_state_dict_provider(
            provider=provider,
            model_paths=surgery_plan.inputs,
            max_io_workers=num_workers,
            arena_root=arena_root,
            arena_segment_size=arena_segment_size,
        )
        configured_pairs = zip(
            normalize_transform_specs(raw.get("transforms")),
            surgery_plan.transforms,
            strict=False,
        )

        with use_output_emitter(output_lines.append):
            _, executed_transforms = execute_transform_pairs(
                configured_pairs,
                state_dict_provider,
                interactive=False,
            )

            if surgery_plan.output is None:
                logger.info("No output configured; execution finished without save.")
            elif get_runtime_flags().dry_run:
                logger.info("Dry-run enabled; skipping output save.")
            else:
                persisted = state_dict_provider.save_output(
                    surgery_plan,
                    default_shard_size=shard_size,
                    max_io_workers=num_workers,
                )
                written_path = str(persisted)
                logger.info("Output saved to %s", written_path)

        if summarize:
            summary_doc = build_raw_plan(
                inputs=raw.get("inputs", []),
                output=raw.get("output"),
                transforms=executed_transforms,
            )
            summary_yaml = OmegaConf.to_yaml(summary_doc)

        return WebRunResult(
            ok=True,
            logs=logs,
            output_lines=output_lines,
            executed_transforms=executed_transforms,
            summary_yaml=summary_yaml,
            written_path=written_path,
        )
    finally:
        if state_dict_provider is not None:
            state_dict_provider.close()
        root_logger.removeHandler(log_handler)
        logger.removeHandler(log_handler)


def configure_logging(*, log_level: str) -> None:
    allowed = {"debug", "info", "warning", "error", "critical"}
    level_name = log_level.strip().lower()
    if level_name not in allowed:
        raise ValueError(f"log_level must be one of: {', '.join(sorted(allowed))}")
    level = getattr(logging, level_name.upper())
    logging.getLogger().setLevel(level)
    logger.setLevel(level)


class _ListLogHandler(logging.Handler):
    def __init__(self, sink: list[str]) -> None:
        super().__init__()
        self._sink = sink

    def emit(self, record: logging.LogRecord) -> None:
        self._sink.append(self.format(record))


__all__ = ["run_web_plan", "configure_logging"]
