from __future__ import annotations

import base64
import json
import logging
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any
import uuid

from ..engine import list_model_aliases
from .backend import (
    apply_load_transform,
    apply_transform,
    default_alias,
    parse_filter_expr,
    render_dump_for_alias,
    require_string,
    serialize_models,
    transform_items,
)
from .page import HTML_PAGE
from .session import SessionState


logger = logging.getLogger("brainsurgery")


def handler_factory(session: SessionState):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/" or self.path.startswith("/?"):
                self._send_html(HTML_PAGE)
                return
            if self.path == "/api/transforms":
                self._send_json({"ok": True, "transforms": transform_items()})
                return
            if self.path == "/api/state":
                with session.lock:
                    self._send_json({"ok": True, "models": serialize_models(session.provider)})
                return
            self.send_error(HTTPStatus.NOT_FOUND, "Not Found")

        def do_POST(self) -> None:  # noqa: N802
            if self.path == "/api/load":
                try:
                    body = self._read_json_body()
                    alias = body.get("alias")
                    if alias is not None and (not isinstance(alias, str) or not alias.strip()):
                        raise ValueError("alias must be a non-empty string when provided.")
                    alias_clean = alias.strip() if isinstance(alias, str) else None

                    filename = require_string(body.get("filename"), "filename")
                    content_b64 = require_string(body.get("content_b64"), "content_b64")
                    raw = base64.b64decode(content_b64, validate=True)
                except Exception as exc:
                    self._send_json({"ok": False, "error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return

                load_path = session.upload_root / f"{uuid.uuid4().hex}_{Path(filename).name}"
                load_path.write_bytes(raw)

                with session.lock:
                    try:
                        chosen_alias = alias_clean or default_alias(session.provider)
                        apply_load_transform(provider=session.provider, path=load_path, alias=chosen_alias)
                        models = serialize_models(session.provider)
                    except Exception as exc:
                        self._send_json({"ok": False, "error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                        return

                self._send_json({"ok": True, "models": models})
                return

            if self.path == "/api/apply_transform":
                try:
                    body = self._read_json_body()
                    transform_name = require_string(body.get("transform"), "transform")
                    payload = body.get("payload")
                    if payload is None:
                        payload = {}
                    if not isinstance(payload, dict):
                        raise ValueError("payload must be an object.")
                except Exception as exc:
                    self._send_json({"ok": False, "error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return

                with session.lock:
                    try:
                        output = apply_transform(
                            provider=session.provider,
                            transform_name=transform_name,
                            payload=payload,
                        )
                        models = serialize_models(session.provider)
                    except Exception as exc:
                        self._send_json({"ok": False, "error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                        return
                self._send_json({"ok": True, "models": models, "output": output})
                return

            if self.path == "/api/model_dump":
                try:
                    body = self._read_json_body()
                    alias = require_string(body.get("alias"), "alias")
                    format_name = require_string(body.get("format"), "format").strip().lower()
                    if format_name not in {"compact", "tree"}:
                        raise ValueError("format must be 'compact' or 'tree'.")
                    verbosity = require_string(body.get("verbosity", "shape"), "verbosity").strip().lower()
                    if verbosity not in {"shape", "stat"}:
                        raise ValueError("verbosity must be 'shape' or 'stat'.")
                    target = parse_filter_expr(body.get("filter"), alias=alias)
                except Exception as exc:
                    self._send_json({"ok": False, "error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return

                with session.lock:
                    try:
                        if alias not in set(list_model_aliases(session.provider)):
                            raise ValueError(f"unknown alias: {alias!r}")
                        dumped, matched_count, total_count = render_dump_for_alias(
                            provider=session.provider,
                            alias=alias,
                            format_name=format_name,
                            verbosity=verbosity,
                            target=target,
                        )
                    except Exception as exc:
                        self._send_json({"ok": False, "error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                        return

                self._send_json(
                    {
                        "ok": True,
                        "dump": dumped,
                        "matched_count": matched_count,
                        "total_count": total_count,
                    }
                )
                return

            self.send_error(HTTPStatus.NOT_FOUND, "Not Found")

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
            self.send_header("Cache-Control", "no-store, max-age=0")
            self.send_header("Pragma", "no-cache")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def _send_json(self, payload: dict[str, Any], *, status: HTTPStatus = HTTPStatus.OK) -> None:
            encoded = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store, max-age=0")
            self.send_header("Pragma", "no-cache")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
            logger.debug("webui2 request: " + format, *args)

    return Handler


__all__ = ["handler_factory"]
