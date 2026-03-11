from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import typer

from .config import load_cli_config
from .history import configure_history
from .interactive import normalize_transform_specs, prompt_interactive_transform
from .summary import build_raw_plan, write_executed_plan_summary
from ..engine import (
    compile_plan,
    execute_transform_pairs,
    ProviderError,
    create_state_dict_provider,
    list_model_aliases,
)

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("brainsurgery")
_ALLOWED_LOG_LEVELS = {"debug", "info", "warning", "error", "critical"}

app = typer.Typer(help="Brain surgery CLI.")


def configure_logging(log_level: str) -> None:
    level_name = log_level.strip().lower()
    if level_name not in _ALLOWED_LOG_LEVELS:
        raise typer.BadParameter(
            f"log-level must be one of: {', '.join(sorted(_ALLOWED_LOG_LEVELS))}"
        )

    level = getattr(logging, level_name.upper())
    logging.getLogger().setLevel(level)
    logger.setLevel(level)


@app.command()
def run(
    config_items: list[str] = typer.Argument(
        None,
        help=(
            "YAML plan fragments and/or key=value overrides. YAML files are loaded and "
            "deep-merged in order; overrides are applied last. Typically used to define "
            "inputs, transforms, and output."
        ),
    ),
    shard_size: str = typer.Option(
        "5GB",
        help="Default shard size when writing directory outputs (e.g. safetensors shards). Ignored for single-file outputs.",
    ),
    num_workers: int = typer.Option(
        8,
        help="Maximum parallel workers for loading and saving tensors. Higher values improve I/O throughput but increase memory pressure.",
    ),
    provider: str = typer.Option(
        "inmemory",
        help="State-dict backend. 'inmemory' loads tensors into RAM; 'arena' memory-maps tensors to disk for large models.",
    ),
    arena_root: Path = typer.Option(
        Path(".brainsurgery"),
        help="Directory for arena storage when using --provider arena (memory-mapped tensor backing).",
    ),
    arena_segment_size: str = typer.Option(
        "1GB",
        help="Segment size for arena storage (e.g. 1GB). Larger segments reduce fragmentation but use more disk.",
    ),
    interactive: bool = typer.Option(
        False,
        "-i",
        "--interactive",
        help="Run configured transforms, then enter an interactive prompt to execute additional transforms incrementally.",
    ),
    summarize: bool = typer.Option(
        True,
        "-s",
        "--summarize/--no-summarize",
        help="Write a YAML summary of the transforms actually executed (after overrides and interactive edits).",
    ),
    summarize_path: Path | None = typer.Option(
        None,
        help="Destination for the executed-plan summary. Defaults to stdout if not set.",
    ),
    log_level: str = typer.Option(
        "info",
        "--log-level",
        help="Logging verbosity (debug, info, warning, error, critical).",
    ),
) -> None:
    """Load a plan, execute it, and save the rewritten output checkpoint."""
    configure_logging(log_level)
    configure_history()

    raw_plan = load_cli_config(config_items or [])

    logger.info(
        "Scrubbing in. Surgical plan assembled from %d config item(s)",
        len(config_items or []),
    )
    surgery_plan = compile_plan(raw_plan)
    logger.info(
        "Surgical plan ready: %d brain(s) prepped, %d procedure(s) scheduled, preservation %s",
        len(surgery_plan.inputs),
        len(surgery_plan.transforms),
        surgery_plan.output.path if surgery_plan.output else None,
    )

    try:
        state_dict_provider = create_state_dict_provider(
            provider=provider,
            model_paths=surgery_plan.inputs,
            max_io_workers=num_workers,
            arena_root=arena_root,
            arena_segment_size=arena_segment_size,
        )
    except ProviderError as exc:
        raise typer.BadParameter(str(exc)) from exc

    executed_transforms: list[dict[str, Any]] = []
    written_path: str | Path | None = None

    try:
        configured_pairs = zip(
            normalize_transform_specs(raw_plan.get("transforms")),
            surgery_plan.transforms,
            strict=False,
        )
        should_continue, newly_executed = execute_transform_pairs(
            configured_pairs,
            state_dict_provider,
            interactive=False,
        )
        executed_transforms.extend(newly_executed)

        if should_continue and interactive:
            logger.info("Entering interactive mode after configured procedures")

            while True:
                extra_specs = prompt_interactive_transform(state_dict_provider=state_dict_provider)
                if extra_specs is None:
                    logger.info("Interactive session complete")
                    break

                interactive_inputs = raw_plan.get("inputs", [])
                aliases = sorted(list_model_aliases(state_dict_provider))
                if aliases:
                    interactive_inputs = [f"{alias}::/dev/null" for alias in aliases]

                interactive_raw_plan = build_raw_plan(
                    inputs=interactive_inputs,
                    output=raw_plan.get("output"),
                    transforms=extra_specs,
                )

                try:
                    interactive_plan = compile_plan(interactive_raw_plan)
                except Exception as exc:
                    logger.error("Could not compile interactive transform(s): %s", exc)
                    continue

                interactive_pairs = zip(
                    extra_specs,
                    interactive_plan.transforms,
                    strict=False,
                )
                should_continue, newly_executed = execute_transform_pairs(
                    interactive_pairs,
                    state_dict_provider,
                    interactive=True,
                )
                executed_transforms.extend(newly_executed)

                if not should_continue:
                    logger.info("Leaving interactive mode")
                    break

        if surgery_plan.output is None:
            logger.info("No preservation requested; concluding operation without closure")
        else:
            written_path = state_dict_provider.save_output(
                surgery_plan,
                default_shard_size=shard_size,
                max_io_workers=num_workers,
            )
            logger.info("Operation complete. Brain preserved at %s", written_path)

        if summarize:
            write_executed_plan_summary(
                inputs=raw_plan.get("inputs", []),
                output=raw_plan.get("output"),
                transforms=executed_transforms,
                destination=summarize_path,
            )

    finally:
        state_dict_provider.close()


if __name__ == "__main__":
    app()


__all__ = ["app", "configure_logging", "run", "logger"]
