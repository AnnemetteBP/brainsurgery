from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import logging


def serve_http(
    *,
    host: str,
    port: int,
    handler_factory: Callable[[], type[BaseHTTPRequestHandler]],
    startup_message: str,
    shutdown_message: str,
    logger: logging.Logger | None = None,
    on_close: Callable[[], None] | None = None,
) -> None:
    active_logger = logger or logging.getLogger("brainsurgery")
    server = ThreadingHTTPServer((host, port), handler_factory())
    active_logger.info(startup_message, host, port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        active_logger.info(shutdown_message)
    finally:
        server.server_close()
        if on_close is not None:
            on_close()
