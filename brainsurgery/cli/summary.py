import logging
from pathlib import Path

import typer
from omegaconf import OmegaConf

from ..engine import SurgeryPlan

logger = logging.getLogger("brainsurgery")


def _write_executed_plan_summary(
    *,
    plan: SurgeryPlan,
    destination: Path | None,
) -> None:
    summary_doc = plan.to_raw_plan(executed_only=True)

    if destination is None:
        typer.echo(OmegaConf.to_yaml(summary_doc))
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config=OmegaConf.create(summary_doc), f=str(destination))
    logger.info("Executed plan summary written to %s", destination)
