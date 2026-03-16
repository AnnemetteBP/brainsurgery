import logging
import tempfile
import threading
from pathlib import Path

from brainsurgery.engine import create_state_dict_provider

from ..http import serve_http
from .handler import _handler_factory
from .session import _SessionState

logger = logging.getLogger("brainsurgery")


def _serve_webui(*, host: str, port: int) -> None:
    provider = create_state_dict_provider(
        provider="inmemory",
        model_paths={},
        max_io_workers=8,
        arena_root=Path(".brainsurgery"),
        arena_segment_size="1GB",
    )
    session = _SessionState(
        provider=provider,
        lock=threading.Lock(),
        upload_root=Path(tempfile.gettempdir()) / "brainsurgery-webui-uploads",
    )
    session.upload_root.mkdir(parents=True, exist_ok=True)
    serve_http(
        host=host,
        port=port,
        handler_factory=lambda: _handler_factory(session),
        startup_message="Brain surgery web UI available at http://%s:%d",
        shutdown_message="Shutting down web UI server",
        logger=logger,
        on_close=provider.close,
    )
