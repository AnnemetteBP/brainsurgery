from __future__ import annotations

import sys

import typer

from . import transforms
from .cli import app as cli_app
from .webui import app as webui_app


app = typer.Typer(help="Brain surgery command suite.")
app.add_typer(cli_app, name="cli")
app.add_typer(webui_app, name="webui")


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    top_level = {"cli", "webui", "-h", "--help", "--install-completion", "--show-completion"}
    if args and args[0] in top_level:
        app(args=args, prog_name="brainsurgery")
        return
    app(args=["cli", *args], prog_name="brainsurgery")


__all__ = ["app", "main", "cli_app", "webui_app", "transforms"]
