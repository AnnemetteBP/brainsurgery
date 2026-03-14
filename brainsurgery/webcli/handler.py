from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
import json
import logging
from pathlib import Path
import threading
from typing import Any

from ..engine import ProviderError
from .page import HTML_PAGE
from .runner import run_web_plan


logger = logging.getLogger("brainsurgery")
_run_lock = threading.Lock()


def handler_factory():
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/" or self.path.startswith("/?"):
                self._send_html(HTML_PAGE)
                return
            self.send_error(HTTPStatus.NOT_FOUND, "Not Found")

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/api/run":
                self.send_error(HTTPStatus.NOT_FOUND, "Not Found")
                return
            try:
                payload = self._read_json_body()
                with _run_lock:
                    result = run_web_plan(
                        plan_yaml=_as_string(payload.get("plan_yaml"), "plan_yaml"),
                        shard_size=_as_string(payload.get("shard_size", "5GB"), "shard_size"),
                        num_workers=_as_int(payload.get("num_workers", 8), "num_workers"),
                        provider=_as_string(payload.get("provider", "inmemory"), "provider"),
                        arena_root=Path(_as_string(payload.get("arena_root", ".brainsurgery"), "arena_root")),
                        arena_segment_size=_as_string(
                            payload.get("arena_segment_size", "1GB"),
                            "arena_segment_size",
                        ),
                        summarize=bool(payload.get("summarize", True)),
                        log_level=_as_string(payload.get("log_level", "info"), "log_level"),
                    )
            except ProviderError as exc:
                self._send_json({"ok": False, "error": f"Provider error: {exc}"}, status=HTTPStatus.BAD_REQUEST)
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
                    "executed_transforms": result.executed_transforms,
                    "summary_yaml": result.summary_yaml,
                    "written_path": result.written_path,
                    "error": result.error,
                }
            )

        def _read_json_body(self) -> dict[str, Any]:
            content_length = self.headers.get("Content-Length")
            if content_length is None:
                raise ValueError("Missing Content-Length.")
            size = int(content_length)
            data = json.loads(self.rfile.read(size).decode("utf-8"))
            if not isinstance(data, dict):
                raise ValueError("Request body must be a JSON object.")
            return data

        def _send_html(self, body: str) -> None:
            encoded = body.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def _send_json(self, payload: dict[str, Any], *, status: HTTPStatus = HTTPStatus.OK) -> None:
            encoded = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
            logger.debug("webcli request: " + format, *args)

    return Handler


def _as_string(value: Any, key: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string.")
    return value


def _as_int(value: Any, key: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be an integer.")
    return value


__all__ = ["handler_factory"]
