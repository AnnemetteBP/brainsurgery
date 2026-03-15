import logging

from ..http import serve_http
from .handler import _handler_factory


logger = logging.getLogger("brainsurgery")


def _serve_webcli(*, host: str, port: int) -> None:
    serve_http(
        host=host,
        port=port,
        handler_factory=_handler_factory,
        startup_message="Brain surgery web CLI available at http://%s:%d",
        shutdown_message="Shutting down web CLI server",
        logger=logger,
    )
