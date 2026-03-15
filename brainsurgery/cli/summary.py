import logging
from pathlib import Path

import typer
from omegaconf import OmegaConf

from ..engine import (
    SurgeryPlan,
    executed_plan_summary_doc,
    executed_plan_summary_yaml,
    parse_summary_mode,
)

logger = logging.getLogger("brainsurgery")


def _write_executed_plan_summary(
    *,
    plan: SurgeryPlan,
    destination: Path | None,
    mode: str = "raw",
) -> None:
    summary_mode = parse_summary_mode(mode)
    if destination is None:
        typer.echo(executed_plan_summary_yaml(plan, mode=summary_mode))
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(
        config=OmegaConf.create(executed_plan_summary_doc(plan, mode=summary_mode)),
        f=str(destination),
    )
    logger.info("Executed plan summary written to %s", destination)
