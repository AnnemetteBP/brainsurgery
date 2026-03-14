from __future__ import annotations

from http.server import ThreadingHTTPServer
import logging
from pathlib import Path
import tempfile
import threading

from ..engine import create_state_dict_provider
from .handler import handler_factory
from .session import SessionState


logger = logging.getLogger("brainsurgery")


def serve_webui(*, host: str, port: int) -> None:
    provider = create_state_dict_provider(
        provider="inmemory",
        model_paths={},
        max_io_workers=8,
        arena_root=Path(".brainsurgery"),
        arena_segment_size="1GB",
    )
    session = SessionState(
        provider=provider,
        lock=threading.Lock(),
        upload_root=Path(tempfile.gettempdir()) / "brainsurgery-webui-uploads",
    )
    session.upload_root.mkdir(parents=True, exist_ok=True)

    handler = handler_factory(session)
    server = ThreadingHTTPServer((host, port), handler)
    logger.info("Brain surgery web UI available at http://%s:%d", host, port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down web UI server")
    finally:
        server.server_close()
        provider.close()


__all__ = ["serve_webui"]
