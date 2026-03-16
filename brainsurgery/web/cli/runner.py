import importlib
import logging
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from brainsurgery.engine import (
    apply_log_level,
    compile_plan,
    create_state_dict_provider,
    executed_plan_summary_doc,
    get_runtime_flags,
    normalize_raw_plan,
    parse_summary_mode,
    reset_runtime_flags,
    use_output_emitter,
)

from .models import _WebRunResult

logger = logging.getLogger("brainsurgery")
yaml: Any = importlib.import_module("yaml")


def _run_web_plan(
    *,
    plan_yaml: str,
    shard_size: str,
    num_workers: int,
    provider: str,
    arena_root: Path,
    arena_segment_size: str,
    summarize: bool,
    summary_mode: str = "raw",
    log_level: str,
) -> _WebRunResult:
    raw = yaml.safe_load(plan_yaml) if plan_yaml.strip() else {}
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("Plan YAML root must be a mapping.")
    planned_raw = normalize_raw_plan(raw)

    _configure_logging(log_level=log_level)
    reset_runtime_flags()

    logs: list[str] = []
    output_lines: list[str] = []
    summary_yaml: str | None = None
    written_path: str | None = None

    log_handler = _ListLogHandler(logs)
    log_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s"))
    root_logger = logging.getLogger()
    root_logger.addHandler(log_handler)
    logger.addHandler(log_handler)

    state_dict_provider = None
    try:
        surgery_plan = compile_plan(planned_raw)
        state_dict_provider = create_state_dict_provider(
            provider=provider,
            model_paths=surgery_plan.inputs,
            max_io_workers=num_workers,
            arena_root=arena_root,
            arena_segment_size=arena_segment_size,
        )
        with use_output_emitter(output_lines.append):
            surgery_plan.execute_pending(
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
            mode = parse_summary_mode(summary_mode)
            summary_yaml = OmegaConf.to_yaml(executed_plan_summary_doc(surgery_plan, mode=mode))

        return _WebRunResult(
            ok=True,
            logs=logs,
            output_lines=output_lines,
            summary_yaml=summary_yaml,
            written_path=written_path,
        )
    finally:
        if state_dict_provider is not None:
            state_dict_provider.close()
        root_logger.removeHandler(log_handler)
        logger.removeHandler(log_handler)


def _configure_logging(*, log_level: str) -> None:
    apply_log_level(log_level)


class _ListLogHandler(logging.Handler):
    def __init__(self, sink: list[str]) -> None:
        super().__init__()
        self._sink = sink

    def emit(self, record: logging.LogRecord) -> None:
        self._sink.append(self.format(record))
