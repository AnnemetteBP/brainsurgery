import base64
from http import HTTPStatus
from pathlib import Path
import tempfile
import time
from typing import Any
import uuid

from omegaconf import OmegaConf

from brainsurgery.core import IteratingTransform, get_transform
from brainsurgery.core.runtime.transform import IterationProgress
from brainsurgery.engine import list_model_aliases
from ..http import JsonRequestHandler
from .backend import (
    _apply_load_transform,
    _apply_transform,
    _default_alias,
    _parse_filter_expr,
    _render_dump_for_alias,
    _render_execution_summary,
    _require_string,
    _serialize_runtime_flags,
    _serialize_models,
    _transform_items,
)
from .page import _HTML_PAGE
from .session import _SessionState


def _now_ms() -> int:
    return int(time.time() * 1000)


def _snapshot_progress(session: _SessionState) -> dict[str, Any]:
    with session.progress_lock:
        return dict(session.progress)


def _begin_progress(session: _SessionState, *, transform: str, iterating: bool) -> None:
    now = _now_ms()
    with session.progress_lock:
        session.progress = {
            "active": True,
            "iterating": iterating,
            "transform": transform,
            "desc": None,
            "unit": "item",
            "completed": 0,
            "total": None,
            "started_at": now,
            "updated_at": now,
            "error": None,
        }


def _update_progress_from_event(session: _SessionState, event: IterationProgress) -> None:
    now = _now_ms()
    with session.progress_lock:
        session.progress["iterating"] = True
        session.progress["desc"] = event.desc
        session.progress["unit"] = event.unit
        session.progress["completed"] = int(event.completed)
        session.progress["total"] = int(event.total) if event.total is not None else None
        session.progress["updated_at"] = now


def _make_progress_callback(session: _SessionState):
    def _callback(event: IterationProgress) -> None:
        _update_progress_from_event(session, event)

    return _callback


def _finish_progress(session: _SessionState, *, error: str | None = None) -> None:
    now = _now_ms()
    with session.progress_lock:
        session.progress["active"] = False
        session.progress["updated_at"] = now
        session.progress["error"] = error


def _handler_factory(session: _SessionState):
    class Handler(JsonRequestHandler):
        request_log_prefix = "webui"

        def _cache_headers(self) -> list[tuple[str, str]]:
            return [
                ("Cache-Control", "no-store, max-age=0"),
                ("Pragma", "no-cache"),
            ]

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/" or self.path.startswith("/?"):
                self._send_html(_HTML_PAGE)
                return
            if self.path == "/api/transforms":
                self._send_json({"ok": True, "transforms": _transform_items()})
                return
            if self.path == "/api/state":
                with session.lock:
                    self._send_json(
                        {
                            "ok": True,
                            "models": _serialize_models(session.provider),
                            "runtime_flags": _serialize_runtime_flags(),
                        }
                    )
                return
            if self.path == "/api/progress":
                self._send_json({"ok": True, "progress": _snapshot_progress(session)})
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
                    server_path_raw = body.get("server_path")
                    server_path = (
                        server_path_raw.strip()
                        if isinstance(server_path_raw, str) and server_path_raw.strip()
                        else None
                    )

                    load_path: Path
                    if server_path is not None:
                        load_path = Path(server_path)
                    else:
                        filename = _require_string(body.get("filename"), "filename")
                        content_b64 = _require_string(body.get("content_b64"), "content_b64")
                        raw = base64.b64decode(content_b64, validate=True)
                        load_path = session.upload_root / f"{uuid.uuid4().hex}_{Path(filename).name}"
                        load_path.write_bytes(raw)
                except Exception as exc:
                    self._send_json({"ok": False, "error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return

                with session.lock:
                    try:
                        chosen_alias = alias_clean or _default_alias(session.provider)
                        _apply_load_transform(
                            provider=session.provider,
                            plan=session.plan,
                            path=load_path,
                            alias=chosen_alias,
                        )
                        session.default_model = chosen_alias
                        models = _serialize_models(session.provider)
                    except Exception as exc:
                        self._send_json({"ok": False, "error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                        return

                self._send_json(
                    {
                        "ok": True,
                        "models": models,
                        "runtime_flags": _serialize_runtime_flags(),
                    }
                )
                return

            if self.path == "/api/_apply_transform":
                try:
                    body = self._read_json_body()
                    transform_name = _require_string(body.get("transform"), "transform")
                    payload = body.get("payload")
                    summary_mode_raw = body.get("summary_mode", "raw")
                    if payload is None:
                        payload = {}
                    if transform_name not in {"help", "assert"} and not isinstance(payload, dict):
                        raise ValueError("payload must be an object.")
                    if not isinstance(summary_mode_raw, str):
                        raise ValueError("summary_mode must be a string when provided.")
                except Exception as exc:
                    self._send_json({"ok": False, "error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return

                try:
                    iterating = isinstance(get_transform(transform_name), IteratingTransform)
                except Exception as exc:
                    self._send_json({"ok": False, "error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return
                _begin_progress(session, transform=transform_name, iterating=iterating)
                progress_callback = _make_progress_callback(session) if iterating else None
                with session.lock:
                    try:
                        if transform_name == "assert" and isinstance(payload, str):
                            text = payload.strip()
                            if not text:
                                raise ValueError("assert payload cannot be empty.")
                            try:
                                parsed = OmegaConf.create(text)
                                payload = OmegaConf.to_container(parsed, resolve=True)
                            except Exception as exc:
                                raise ValueError(f"invalid assert YAML payload: {exc}") from exc
                        output, next_default_model = _apply_transform(
                            provider=session.provider,
                            plan=session.plan,
                            transform_name=transform_name,
                            payload=payload,
                            default_model=session.default_model,
                            progress_callback=progress_callback,
                        )
                        session.default_model = next_default_model
                        if transform_name == "exit":
                            summary = _render_execution_summary(plan=session.plan, mode=summary_mode_raw)
                            output = f"{output}\n\n{summary}".strip() if output else summary
                        models = _serialize_models(session.provider)
                    except Exception as exc:
                        _finish_progress(session, error=str(exc))
                        self._send_json({"ok": False, "error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                        return
                _finish_progress(session, error=None)
                self._send_json(
                    {
                        "ok": True,
                        "models": models,
                        "output": output,
                        "runtime_flags": _serialize_runtime_flags(),
                    }
                )
                return

            if self.path == "/api/save_download":
                try:
                    body = self._read_json_body()
                    payload = body.get("payload")
                    if payload is None:
                        payload = {}
                    if not isinstance(payload, dict):
                        raise ValueError("payload must be an object.")
                except Exception as exc:
                    self._send_json({"ok": False, "error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return

                _begin_progress(session, transform="save", iterating=False)
                with session.lock:
                    try:
                        output, filename, mime, content_b64 = self._run_save_download(payload)
                        models = _serialize_models(session.provider)
                    except Exception as exc:
                        _finish_progress(session, error=str(exc))
                        self._send_json({"ok": False, "error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                        return
                _finish_progress(session, error=None)
                self._send_json(
                    {
                        "ok": True,
                        "models": models,
                        "output": output,
                        "runtime_flags": _serialize_runtime_flags(),
                        "download_filename": filename,
                        "download_mime": mime,
                        "download_b64": content_b64,
                    }
                )
                return

            if self.path == "/api/model_dump":
                try:
                    body = self._read_json_body()
                    alias = _require_string(body.get("alias"), "alias")
                    format_name = _require_string(body.get("format"), "format").strip().lower()
                    if format_name not in {"compact", "tree"}:
                        raise ValueError("format must be 'compact' or 'tree'.")
                    verbosity = _require_string(body.get("verbosity", "shape"), "verbosity").strip().lower()
                    if verbosity not in {"shape", "stat"}:
                        raise ValueError("verbosity must be 'shape' or 'stat'.")
                    target = _parse_filter_expr(body.get("filter"), alias=alias)
                except Exception as exc:
                    self._send_json({"ok": False, "error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return

                with session.lock:
                    try:
                        if alias not in set(list_model_aliases(session.provider)):
                            raise ValueError(f"unknown alias: {alias!r}")
                        dumped, matched_count, total_count = _render_dump_for_alias(
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

        def _run_save_download(self, payload: dict[str, Any]) -> tuple[str, str, str, str]:
            requested_path = _require_string(payload.get("path"), "payload.path")
            requested_name = Path(requested_path).name or "model"
            payload_copy = dict(payload)

            with tempfile.TemporaryDirectory(prefix="brainsurgery-webui-save-") as tmp:
                tmp_root = Path(tmp)
                out_path = tmp_root / requested_name
                payload_copy["path"] = str(out_path)

                output, next_default_model = _apply_transform(
                    provider=session.provider,
                    plan=session.plan,
                    transform_name="save",
                    payload=payload_copy,
                    default_model=session.default_model,
                    record_payload=payload,
                )
                session.default_model = next_default_model

                if out_path.is_dir():
                    raise ValueError(
                        "save download supports files only. "
                        "For directory outputs, use 'save on server path'."
                    )

                if out_path.is_file():
                    raw = out_path.read_bytes()
                    filename = _suggest_download_filename(
                        requested_name=requested_name,
                        out_path=out_path,
                        payload=payload_copy,
                    )
                    mime = "application/octet-stream"
                    return output, filename, mime, base64.b64encode(raw).decode("ascii")

                raise ValueError("save did not produce a file or directory to download.")

    return Handler


def _suggest_download_filename(*, requested_name: str, out_path: Path, payload: dict[str, Any]) -> str:
    if out_path.suffix:
        return out_path.name

    base = requested_name or "model"
    if "." in base:
        return base

    fmt_raw = payload.get("format")
    fmt = fmt_raw.strip().lower() if isinstance(fmt_raw, str) else "safetensors"
    ext = {
        "safetensors": ".safetensors",
        "numpy": ".npy",
        "torch": ".pt",
    }.get(fmt, ".bin")
    return base + ext
