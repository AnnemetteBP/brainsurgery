import logging
import webbrowser

import typer

from ..engine import apply_log_level
from .server import _serve_webui


logger = logging.getLogger("brainsurgery")

app = typer.Typer(help="Brain surgery web UI.")


def configure_logging(log_level: str) -> None:
    try:
        apply_log_level(log_level)
    except ValueError:
        raise typer.BadParameter(
            "log-level must be one of: critical, debug, error, info, warning"
        )


@app.callback(invoke_without_command=True)
def webui(
    host: str = typer.Option(
        "127.0.0.1",
        help="Host interface to bind the web UI server.",
    ),
    port: int = typer.Option(
        8766,
        help="Port for the web UI server.",
    ),
    log_level: str = typer.Option(
        "info",
        "--log-level",
        help="Logging verbosity (debug, info, warning, error, critical).",
    ),
    open_browser: bool = typer.Option(
        True,
        "--open-browser/--no-open-browser",
        help="Open the web UI URL in the default browser on startup.",
    ),
) -> None:
    configure_logging(log_level)
    url_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    url = f"http://{url_host}:{port}"
    logger.info("Launching BrainSurgery web UI on %s", url)
    if open_browser:
        try:
            webbrowser.open(url)
        except Exception as exc:
            logger.warning("Could not open browser automatically: %s", exc)
    _serve_webui(host=host, port=port)


__all__ = ["app", "configure_logging", "webui", "logger"]
