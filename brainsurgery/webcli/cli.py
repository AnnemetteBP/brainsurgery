import logging
import webbrowser

import typer

from .server import serve_webcli


logger = logging.getLogger("brainsurgery")
_ALLOWED_LOG_LEVELS = {"debug", "info", "warning", "error", "critical"}

app = typer.Typer(help="Brain surgery web CLI UI.")


def configure_logging(log_level: str) -> None:
    level_name = log_level.strip().lower()
    if level_name not in _ALLOWED_LOG_LEVELS:
        raise typer.BadParameter(
            f"log-level must be one of: {', '.join(sorted(_ALLOWED_LOG_LEVELS))}"
        )
    level = getattr(logging, level_name.upper())
    logging.getLogger().setLevel(level)
    logger.setLevel(level)


@app.callback(invoke_without_command=True)
def webcli(
    host: str = typer.Option(
        "127.0.0.1",
        help="Host interface to bind the web CLI server.",
    ),
    port: int = typer.Option(
        8765,
        help="Port for the web CLI server.",
    ),
    log_level: str = typer.Option(
        "info",
        "--log-level",
        help="Logging verbosity (debug, info, warning, error, critical).",
    ),
    open_browser: bool = typer.Option(
        True,
        "--open-browser/--no-open-browser",
        help="Open the web CLI URL in the default browser on startup.",
    ),
) -> None:
    configure_logging(log_level)
    url_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    url = f"http://{url_host}:{port}"
    logger.info("Launching BrainSurgery web CLI on %s", url)
    if open_browser:
        try:
            webbrowser.open(url)
        except Exception as exc:
            logger.warning("Could not open browser automatically: %s", exc)
    serve_webcli(host=host, port=port)


__all__ = ["app", "configure_logging", "webcli", "logger"]
