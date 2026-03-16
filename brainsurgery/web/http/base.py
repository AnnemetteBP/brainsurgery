import json
import logging
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from typing import Any


def as_string(value: Any, key: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string.")
    return value


def as_int(value: Any, key: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be an integer.")
    return value


class JsonRequestHandler(BaseHTTPRequestHandler):
    request_log_prefix = "web"

    def _cache_headers(self) -> list[tuple[str, str]]:
        return []

    def _read_json_body(self) -> dict[str, Any]:
        content_length = self.headers.get("Content-Length")
        if content_length is None:
            raise ValueError("Missing Content-Length.")
        size = int(content_length)
        body = json.loads(self.rfile.read(size).decode("utf-8"))
        if not isinstance(body, dict):
            raise ValueError("Request body must be a JSON object.")
        return body

    def _send_html(self, body: str) -> None:
        encoded = body.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        for key, value in self._cache_headers():
            self.send_header(key, value)
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _send_json(self, payload: dict[str, Any], *, status: HTTPStatus = HTTPStatus.OK) -> None:
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        for key, value in self._cache_headers():
            self.send_header(key, value)
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        logging.getLogger("brainsurgery").debug(
            self.request_log_prefix + " request: " + format, *args
        )
