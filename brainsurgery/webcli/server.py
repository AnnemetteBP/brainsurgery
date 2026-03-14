from http.server import ThreadingHTTPServer
import logging

from .handler import handler_factory


logger = logging.getLogger("brainsurgery")


def serve_webcli(*, host: str, port: int) -> None:
    server = ThreadingHTTPServer((host, port), handler_factory())
    logger.info("Brain surgery web CLI available at http://%s:%d", host, port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down web CLI server")
    finally:
        server.server_close()


__all__ = ["serve_webcli"]
