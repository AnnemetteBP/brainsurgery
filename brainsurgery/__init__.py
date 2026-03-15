import sys

import typer

from . import transforms
from .cli import app as cli_app
from .web.cli import app as webcli_app
from .web.ui import app as webui_app


app = typer.Typer(help="Brain surgery command suite.")
app.add_typer(cli_app, name="cli")
app.add_typer(webcli_app, name="webcli")
app.add_typer(webui_app, name="webui")


_CLI_OPTIONS_WITH_VALUE = {
    "--shard-size",
    "--num-workers",
    "--provider",
    "--arena-root",
    "--arena-segment-size",
    "--summarize-path",
    "--summary-mode",
    "--log-level",
}

_CLI_FLAG_OPTIONS = {
    "-i",
    "--interactive",
    "-s",
    "--summarize",
    "--no-summarize",
}


def _normalize_cli_args(raw_args: list[str]) -> list[str]:
    option_tokens: list[str] = []
    positional_tokens: list[str] = []
    index = 0
    while index < len(raw_args):
        token = raw_args[index]
        if token == "--":
            positional_tokens.extend(raw_args[index:])
            break
        if token in _CLI_OPTIONS_WITH_VALUE:
            option_tokens.append(token)
            if index + 1 < len(raw_args):
                option_tokens.append(raw_args[index + 1])
                index += 2
                continue
        elif any(token.startswith(opt + "=") for opt in _CLI_OPTIONS_WITH_VALUE):
            option_tokens.append(token)
            index += 1
            continue
        elif token in _CLI_FLAG_OPTIONS:
            option_tokens.append(token)
            index += 1
            continue

        positional_tokens.append(token)
        index += 1

    return [*option_tokens, *positional_tokens]


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    top_level = {"cli", "webcli", "webui", "-h", "--help", "--install-completion", "--show-completion"}
    if args and args[0] in top_level:
        if args[0] == "cli":
            app(args=["cli", *_normalize_cli_args(args[1:])], prog_name="brainsurgery")
            return
        app(args=args, prog_name="brainsurgery")
        return
    app(args=["cli", *_normalize_cli_args(args)], prog_name="brainsurgery")


__all__ = ["app", "main", "cli_app", "webcli_app", "webui_app", "transforms"]
