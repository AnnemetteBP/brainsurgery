import logging
import threading
from http import HTTPStatus
from pathlib import Path
from typing import Any

from brainsurgery.engine import ProviderError

from ..http import JsonRequestHandler, as_int, as_string
from .page import _HTML_PAGE
from .runner import _run_web_plan

logger = logging.getLogger("brainsurgery")
_run_lock = threading.Lock()


def _handler_factory():
    class Handler(JsonRequestHandler):
        request_log_prefix = "webcli"

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/" or self.path.startswith("/?"):
                self._send_html(_HTML_PAGE)
                return
            self.send_error(HTTPStatus.NOT_FOUND, "Not Found")

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/api/run":
                self.send_error(HTTPStatus.NOT_FOUND, "Not Found")
                return
            try:
                payload = self._read_json_body()
                with _run_lock:
                    result = _run_web_plan(
                        plan_yaml=as_string(payload.get("plan_yaml"), "plan_yaml"),
                        shard_size=as_string(payload.get("shard_size", "5GB"), "shard_size"),
                        num_workers=as_int(payload.get("num_workers", 8), "num_workers"),
                        provider=as_string(payload.get("provider", "inmemory"), "provider"),
                        arena_root=Path(
                            as_string(payload.get("arena_root", ".brainsurgery"), "arena_root")
                        ),
                        arena_segment_size=as_string(
                            payload.get("arena_segment_size", "1GB"),
                            "arena_segment_size",
                        ),
                        summarize=bool(payload.get("summarize", True)),
                        summary_mode=as_string(payload.get("summary_mode", "raw"), "summary_mode"),
                        log_level=as_string(payload.get("log_level", "info"), "log_level"),
                    )
            except ProviderError as exc:
                self._send_json(
                    {"ok": False, "error": f"Provider error: {exc}"}, status=HTTPStatus.BAD_REQUEST
                )
                return
            except ValueError as exc:
                self._send_json({"ok": False, "error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                return
            except Exception as exc:
                self._send_json({"ok": False, "error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                return

            self._send_json(
                {
                    "ok": result.ok,
                    "logs": result.logs,
                    "output_lines": result.output_lines,
                    "summary_yaml": result.summary_yaml,
                    "written_path": result.written_path,
                    "error": result.error,
                }
            )

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
            logger.debug("webcli request: " + format, *args)

    return Handler
