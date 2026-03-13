from __future__ import annotations

from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import base64
import json
import logging
from pathlib import Path
import tempfile
import threading
from typing import Any
import uuid

from ..core import (
    BinaryMappingTransform,
    DestinationPolicy,
    get_transform,
    list_transforms,
    match_expr_names,
    parse_model_expr,
)
from ..engine import (
    create_state_dict_provider,
    list_model_aliases,
    reset_runtime_flags,
    use_output_emitter,
)


logger = logging.getLogger("brainsurgery")


@dataclass
class _SessionState:
    provider: Any
    lock: threading.Lock
    upload_root: Path


def serve_webui2(*, host: str, port: int) -> None:
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
        upload_root=Path(tempfile.gettempdir()) / "brainsurgery-webui2-uploads",
    )
    session.upload_root.mkdir(parents=True, exist_ok=True)

    handler = _handler_factory(session)
    server = ThreadingHTTPServer((host, port), handler)
    logger.info("Brain surgery web UI 2 available at http://%s:%d", host, port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down web UI 2 server")
    finally:
        server.server_close()
        provider.close()


def _handler_factory(session: _SessionState):
    class _Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/" or self.path.startswith("/?"):
                self._send_html(_HTML_PAGE)
                return
            if self.path == "/api/transforms":
                self._send_json({"ok": True, "transforms": _transform_items()})
                return
            if self.path == "/api/state":
                with session.lock:
                    self._send_json({"ok": True, "models": _serialize_models(session.provider)})
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

                    filename = _require_string(body.get("filename"), "filename")
                    content_b64 = _require_string(body.get("content_b64"), "content_b64")
                    raw = base64.b64decode(content_b64, validate=True)
                except Exception as exc:
                    self._send_json({"ok": False, "error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return

                load_path = session.upload_root / f"{uuid.uuid4().hex}_{Path(filename).name}"
                load_path.write_bytes(raw)

                with session.lock:
                    try:
                        chosen_alias = alias_clean or _default_alias(session.provider)
                        _apply_load_transform(provider=session.provider, path=load_path, alias=chosen_alias)
                        models = _serialize_models(session.provider)
                    except Exception as exc:
                        self._send_json({"ok": False, "error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                        return

                self._send_json({"ok": True, "models": models})
                return

            if self.path == "/api/apply_binary_transform":
                try:
                    body = self._read_json_body()
                    transform_name = _require_string(body.get("transform"), "transform")
                    from_ref = _require_committed_ref(body.get("from_ref"), field="from_ref")
                    to_ref = _require_committed_ref(body.get("to_ref"), field="to_ref")
                    extras = body.get("extras")
                    if extras is None:
                        extras = {}
                    if not isinstance(extras, dict):
                        raise ValueError("extras must be an object.")
                except Exception as exc:
                    self._send_json({"ok": False, "error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return

                with session.lock:
                    try:
                        _apply_binary_transform(
                            provider=session.provider,
                            transform_name=transform_name,
                            from_ref=from_ref,
                            to_ref=to_ref,
                            extras=extras,
                        )
                        models = _serialize_models(session.provider)
                    except Exception as exc:
                        self._send_json({"ok": False, "error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                        return
                self._send_json({"ok": True, "models": models})
                return

            if self.path == "/api/model_dump":
                try:
                    body = self._read_json_body()
                    alias = _require_string(body.get("alias"), "alias")
                    format_name = _require_string(body.get("format"), "format").strip().lower()
                    if format_name not in {"compact", "tree"}:
                        raise ValueError("format must be 'compact' or 'tree'.")
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

    return _Handler


def _transform_items() -> list[dict[str, Any]]:
    binary_specs = _binary_transform_specs()
    enabled = {"load", *binary_specs.keys()}
    items: list[dict[str, Any]] = []
    for name in list_transforms():
        spec = binary_specs.get(name)
        items.append(
            {
                "name": name,
                "enabled": name in enabled,
                "binary": bool(spec),
                "extra_keys": spec["extra_keys"] if spec else [],
                "required_extra_keys": spec["required_extra_keys"] if spec else [],
            }
        )
    return items


def _binary_transform_specs() -> dict[str, dict[str, list[str]]]:
    specs: dict[str, dict[str, Any]] = {}
    for name in list_transforms():
        transform = get_transform(name)
        if not isinstance(transform, BinaryMappingTransform):
            continue
        allowed = getattr(transform, "allowed_keys", {"from", "to"})
        required = getattr(transform, "required_keys", {"from", "to"})
        destination_policy = getattr(transform, "destination_policy", DestinationPolicy.ANY)
        extra_keys = sorted(k for k in allowed if k not in {"from", "to"})
        required_extra_keys = sorted(k for k in required if k not in {"from", "to"})
        specs[name] = {
            "extra_keys": extra_keys,
            "required_extra_keys": required_extra_keys,
            "to_must_exist": destination_policy is DestinationPolicy.MUST_EXIST,
        }
    return specs


def _require_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string.")
    return value


def _require_committed_ref(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be an object with alias and filter.")
    alias = _require_string(value.get("alias"), f"{field}.alias")
    raw_filter = value.get("filter")
    if raw_filter is not None and not isinstance(raw_filter, (str, list)):
        raise ValueError(f"{field}.filter must be a string, list, or null.")
    return {"alias": alias, "filter": raw_filter}


def _default_alias(provider: Any) -> str:
    aliases = set(list_model_aliases(provider))
    base = "model"
    if base not in aliases:
        return base
    index = 2
    while True:
        candidate = f"{base}_{index}"
        if candidate not in aliases:
            return candidate
        index += 1


def _parse_filter_expr(raw: Any, *, alias: str) -> str | list[Any]:
    source: Any
    if raw is None:
        source = ".*"
    elif isinstance(raw, str):
        text = raw.strip()
        if not text:
            source = ".*"
        elif text.startswith("["):
            parsed = json.loads(text)
            source = parsed
        else:
            source = text
    elif isinstance(raw, list):
        source = raw
    else:
        raise ValueError("filter must be a string or JSON list.")

    ref = parse_model_expr(source, default_model=alias)
    return ref.expr


def _parse_binary_ref_payload(raw: dict[str, Any]) -> tuple[str | list[Any], str, bool]:
    alias = _require_string(raw.get("alias"), "alias")
    expr = _parse_filter_expr(raw.get("filter"), alias=alias)
    if isinstance(expr, str):
        return f"{alias}::{expr}", alias, False
    return expr, alias, True


def _apply_binary_transform(
    *,
    provider: Any,
    transform_name: str,
    from_ref: dict[str, Any],
    to_ref: dict[str, Any],
    extras: dict[str, Any],
) -> None:
    transform = get_transform(transform_name)
    if not isinstance(transform, BinaryMappingTransform):
        raise ValueError(f"transform {transform_name!r} is not a binary mapping transform.")

    from_payload, from_alias, from_is_list = _parse_binary_ref_payload(from_ref)
    to_payload, to_alias, to_is_list = _parse_binary_ref_payload(to_ref)

    default_model: str | None = None
    if from_is_list or to_is_list:
        aliases = set()
        if from_is_list:
            aliases.add(from_alias)
        if to_is_list:
            aliases.add(to_alias)
        if len(aliases) > 1:
            raise ValueError("Structured path filters for from/to must use the same model alias.")
        default_model = next(iter(aliases))

    payload: dict[str, Any] = {"from": from_payload, "to": to_payload}
    for key, value in extras.items():
        payload[key] = value

    reset_runtime_flags()
    spec = transform.compile(payload, default_model=default_model)
    transform.apply(spec, provider)


def _apply_load_transform(*, provider: Any, path: Path, alias: str) -> None:
    reset_runtime_flags()
    transform = get_transform("load")
    spec = transform.compile(
        {
            "path": str(path),
            "alias": alias,
        },
        default_model=None,
    )
    transform.apply(spec, provider)


def _render_dump_for_alias(
    *,
    provider: Any,
    alias: str,
    format_name: str,
    target: str | list[Any],
) -> tuple[str, int, int]:
    names = provider.get_state_dict(alias).keys()
    matched = match_expr_names(
        expr=target,
        names=names,
        op_name="webui2.dump",
        role="target",
    )
    total_count = len(list(names))
    matched_count = len(matched)
    if not matched:
        return "(no tensors matched filter)", 0, total_count

    dump_transform = get_transform("dump")
    spec = dump_transform.compile(
        {
            "target": target,
            "format": format_name,
            "verbosity": "shape",
        },
        default_model=alias,
    )
    lines: list[str] = []
    with use_output_emitter(lines.append):
        dump_transform.apply(spec, provider)
    return "\n".join(lines), matched_count, total_count


def _serialize_models(provider: Any) -> list[dict[str, Any]]:
    models: list[dict[str, Any]] = []
    for alias in sorted(list_model_aliases(provider)):
        state_dict = provider.get_state_dict(alias)
        dump_compact, matched_count, total_count = _render_dump_for_alias(
            provider=provider,
            alias=alias,
            format_name="compact",
            target=".*",
        )
        dump_tree, _, _ = _render_dump_for_alias(
            provider=provider,
            alias=alias,
            format_name="tree",
            target=".*",
        )
        models.append(
            {
                "alias": alias,
                "tensor_count": len(state_dict),
                "matched_count": matched_count,
                "total_count": total_count,
                "dump_compact": dump_compact,
                "dump_tree": dump_tree,
            }
        )
    return models


_HTML_PAGE = """<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>BrainSurgery WebUI2</title>
  <style>
    :root {
      --ink: #1a2229;
      --muted: #4f5a68;
      --paper: #fcf7ef;
      --panel: #fffdf9;
      --line: #d8c4a4;
      --accent: #d45d1f;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      color: var(--ink);
      font-family: "Avenir Next", "Segoe UI", sans-serif;
      background: radial-gradient(circle at 100% 0%, #ffe5c3, transparent 40%), linear-gradient(120deg, #fff8ed, #f8fff8);
      min-height: 100vh;
      padding: 18px;
    }
    .shell {
      max-width: 1260px;
      margin: 0 auto;
      border: 1px solid var(--line);
      border-radius: 16px;
      background: var(--paper);
      overflow: hidden;
    }
    .head {
      padding: 16px;
      border-bottom: 1px solid var(--line);
      background: linear-gradient(90deg, #ffe7cf, #f4ffef);
    }
    .head h1 {
      margin: 0;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      font-family: "Futura", "Avenir Next", sans-serif;
      font-size: 24px;
    }
    .head p { margin: 6px 0 0 0; color: var(--muted); }
    .main {
      display: grid;
      grid-template-columns: 420px 1fr;
      gap: 12px;
      padding: 12px;
      align-items: start;
    }
    .left-stack {
      display: grid;
      grid-template-rows: minmax(0, 1fr) auto;
      gap: 12px;
      align-content: start;
      height: calc(100vh - 165px);
    }
    .card {
      border: 1px solid var(--line);
      border-radius: 12px;
      background: var(--panel);
      padding: 10px;
    }
    .picker-card {
      display: flex;
      flex-direction: column;
      min-height: 0;
    }
    .title {
      margin: 0 0 8px 0;
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--muted);
    }
    .transform-list {
      display: flex;
      flex-direction: column;
      gap: 6px;
      min-height: 180px;
      overflow: auto;
    }
    #transformSearch {
      margin-bottom: 8px;
    }
    .transform-item {
      border: 1px solid #c8b08b;
      border-radius: 8px;
      padding: 7px 9px;
      font-family: "SFMono-Regular", "Menlo", monospace;
      font-size: 12px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      background: #fff;
      cursor: pointer;
    }
    .transform-item.planned {
      cursor: default;
      opacity: 0.72;
    }
    .transform-item.selected {
      border-color: #d45d1f;
      box-shadow: inset 0 0 0 1px #d45d1f;
      background: #fff5eb;
    }
    .pill {
      font-size: 10px;
      text-transform: uppercase;
      border-radius: 999px;
      padding: 2px 6px;
      border: 1px solid #bdc6d2;
      color: #445063;
      background: #f7fbff;
    }
    .pill.enabled {
      border-color: #b3d1be;
      color: #145b31;
      background: #edfdf2;
    }
    input, button, select {
      width: 100%;
      border-radius: 8px;
      border: 1px solid #c7af8a;
      padding: 8px 10px;
      font-size: 13px;
      margin-bottom: 8px;
      background: #fff;
    }
    button {
      border: 0;
      font-weight: 700;
      color: white;
      cursor: pointer;
      background: linear-gradient(130deg, var(--accent), #eb844b);
    }
    #status { font-size: 12px; color: var(--muted); min-height: 18px; margin-top: 4px; }
    .options-placeholder {
      border: 1px dashed #cab18a;
      border-radius: 10px;
      padding: 12px;
      color: var(--muted);
      background: #fffdf9;
      font-size: 13px;
      line-height: 1.45;
    }
    .hidden { display: none; }
    .models {
      display: grid;
      gap: 10px;
      align-content: start;
    }
    .model-pane {
      border: 1px solid var(--line);
      border-radius: 10px;
      background: #fffefb;
      overflow: hidden;
    }
    .model-head {
      padding: 8px 10px;
      border-bottom: 1px solid var(--line);
      font-weight: 700;
      font-size: 13px;
      display: flex;
      justify-content: space-between;
    }
    .model-controls {
      padding: 8px 10px;
      border-bottom: 1px solid var(--line);
      display: grid;
      grid-template-columns: 100px 1fr auto auto auto;
      gap: 6px;
      align-items: center;
      background: #fffaf2;
    }
    .model-controls input,
    .model-controls select {
      margin: 0;
      padding: 6px 8px;
      font-size: 12px;
    }
    .live-pill {
      font-size: 11px;
      color: #6d5a3f;
      text-transform: uppercase;
    }
    .mini-btn {
      margin: 0;
      width: auto;
      padding: 5px 8px;
      border-radius: 7px;
      font-size: 11px;
      font-weight: 700;
      text-transform: lowercase;
      white-space: nowrap;
    }
    .mini-btn.hidden { display: none; }
    .binary-summary {
      margin: 0 0 8px 0;
      padding: 8px;
      border: 1px solid #dcc8aa;
      border-radius: 8px;
      background: #fffdf8;
      font-family: "SFMono-Regular", "Menlo", monospace;
      font-size: 12px;
      line-height: 1.45;
    }
    .binary-summary .label {
      color: #6d5a3f;
      text-transform: uppercase;
      font-size: 10px;
      letter-spacing: 0.06em;
    }
    .binary-summary .value {
      color: #1f2f3e;
      margin-bottom: 6px;
      word-break: break-word;
    }
    .invalid-field {
      border-color: #b52b2b !important;
      box-shadow: inset 0 0 0 1px #b52b2b;
      background: #fff2f2;
    }
    pre {
      margin: 0;
      padding: 10px;
      overflow: auto;
      max-height: 300px;
      line-height: 1.4;
      font-size: 12px;
      font-family: "SFMono-Regular", "Menlo", monospace;
      background: #fffdfa;
    }
    .empty {
      border: 1px dashed #cab18a;
      border-radius: 10px;
      padding: 16px;
      color: var(--muted);
      background: #fffdf9;
    }
    @media (max-width: 980px) {
      .main { grid-template-columns: 1fr; }
      .left-stack { height: auto; grid-template-rows: auto auto; }
    }
  </style>
</head>
<body>
  <div class=\"shell\">
    <div class=\"head\">
      <h1>BrainSurgery WebUI2 (Experimental)</h1>
      <p>Left: transform picker + options. Right: model panes with compact/tree dumps.</p>
    </div>
    <div class=\"main\">
      <div class=\"left-stack\">
        <div class=\"card picker-card\">
          <h2 class=\"title\">Transforms</h2>
          <input id=\"transformSearch\" placeholder=\"Search transforms (e.g. load)\" />
          <div id=\"transformList\" class=\"transform-list\"></div>
        </div>
        <div class=\"card\">
          <h2 class=\"title\">Transform Options</h2>
          <div id=\"optionsEmpty\" class=\"options-placeholder\">Select a transform from the list above to configure it.</div>
          <div id=\"loadPanel\" class=\"hidden\">
            <h2 class=\"title\">Load</h2>
            <input id=\"aliasInput\" placeholder=\"Alias (optional, defaults to model/model_2/...)\" />
            <input id=\"fileInput\" type=\"file\" />
            <button id=\"loadBtn\">Load Selected File</button>
          </div>
          <div id=\"binaryPanel\" class=\"hidden\">
            <h2 id=\"binaryTitle\" class=\"title\">Transform</h2>
            <div class=\"binary-summary\">
              <div class=\"label\">from</div>
              <div id=\"binaryFromSummary\" class=\"value\">(not committed)</div>
              <div class=\"label\">to</div>
              <div id=\"binaryToSummary\" class=\"value\">(not committed)</div>
            </div>
            <div id=\"binaryExtras\"></div>
            <button id=\"binaryRunBtn\">Run Transform</button>
          </div>
          <div id=\"status\">Ready.</div>
        </div>
      </div>
      <div class=\"card\">
        <h2 class=\"title\">Current Models</h2>
        <div id=\"models\" class=\"models\"></div>
      </div>
    </div>
  </div>
  <script>
    const transformsEl = document.getElementById("transformList");
    const transformSearchEl = document.getElementById("transformSearch");
    const modelsEl = document.getElementById("models");
    const statusEl = document.getElementById("status");
    const fileInput = document.getElementById("fileInput");
    const aliasInput = document.getElementById("aliasInput");
    const loadBtn = document.getElementById("loadBtn");
    const loadPanel = document.getElementById("loadPanel");
    const optionsEmpty = document.getElementById("optionsEmpty");
    const binaryPanel = document.getElementById("binaryPanel");
    const binaryTitle = document.getElementById("binaryTitle");
    const binaryFromSummary = document.getElementById("binaryFromSummary");
    const binaryToSummary = document.getElementById("binaryToSummary");
    const binaryExtras = document.getElementById("binaryExtras");
    const binaryRunBtn = document.getElementById("binaryRunBtn");

    let allTransforms = [{ name: "load", enabled: true, binary: false, extra_keys: [], required_extra_keys: [], to_must_exist: false }];
    let selectedTransform = "load";
    const modelViewState = {};
    const binaryConfigByTransform = {};
    let latestModels = [];

    function setStatus(text) { statusEl.textContent = text; }
    function tensorCountText(shown, total, filterText) {
      return filterText.trim() ? (shown + " out of " + total + " tensors") : (total + " tensors");
    }
    function getTransformMeta(name) {
      return allTransforms.find((t) => t.name === name) || null;
    }
    function isReadyTransform(name) {
      const item = getTransformMeta(name);
      return !!(item && item.enabled);
    }
    function isBinaryTransform(name) {
      const item = getTransformMeta(name);
      return !!(item && item.enabled && item.binary);
    }
    function toMustExist(name) {
      const item = getTransformMeta(name);
      return !!(item && item.enabled && item.binary && item.to_must_exist);
    }
    function getBinaryConfig(name) {
      if (!binaryConfigByTransform[name]) {
        binaryConfigByTransform[name] = {
          from_ref: { alias: "", filter: "" },
          to_ref: { alias: "", filter: "" },
          extras: {}
        };
      }
      return binaryConfigByTransform[name];
    }
    function ensureRefObject(ref) {
      if (!ref || typeof ref !== "object") return { alias: "", filter: "" };
      return {
        alias: typeof ref.alias === "string" ? ref.alias : "",
        filter: typeof ref.filter === "string" ? ref.filter : ""
      };
    }
    function describeCommit(ref) {
      if (!ref || !ref.alias.trim()) return "(not set)";
      const f = (ref.filter || "").trim();
      return ref.alias + "::" + (f ? f : ".*");
    }
    function parseExtraValue(raw) {
      const text = (raw || "").trim();
      if (!text) return undefined;
      try {
        return JSON.parse(text);
      } catch (_err) {
        return text;
      }
    }
    function regexToBackrefTemplate(raw) {
      let count = 0;
      let out = "";
      let escaped = false;
      let inClass = false;
      for (let i = 0; i < raw.length; i += 1) {
        const ch = raw[i];
        if (escaped) {
          out += "\\\\" + ch;
          escaped = false;
          continue;
        }
        if (ch === "\\\\") {
          escaped = true;
          continue;
        }
        if (ch === "[") {
          inClass = true;
          out += ch;
          continue;
        }
        if (ch === "]") {
          inClass = false;
          out += ch;
          continue;
        }
        if (!inClass && ch === "(") {
          const next = raw.slice(i + 1, i + 3);
          if (next === "?:" || next === "?=" || next === "?!" || next === "?<" || next === "?>") {
            out += ch;
            continue;
          }
          count += 1;
          out += "\\\\" + String(count);
          let depth = 1;
          let j = i + 1;
          let innerEscaped = false;
          let innerClass = false;
          for (; j < raw.length; j += 1) {
            const c2 = raw[j];
            if (innerEscaped) {
              innerEscaped = false;
              continue;
            }
            if (c2 === "\\\\") {
              innerEscaped = true;
              continue;
            }
            if (c2 === "[") {
              innerClass = true;
              continue;
            }
            if (c2 === "]") {
              innerClass = false;
              continue;
            }
            if (innerClass) continue;
            if (c2 === "(") depth += 1;
            if (c2 === ")") {
              depth -= 1;
              if (depth === 0) break;
            }
          }
          i = j;
          continue;
        }
        out += ch;
      }
      if (!count) return raw;
      return out;
    }
    function copyFromFilterToToTemplate(raw) {
      const text = (raw || "").trim();
      if (!text) return "";
      if (text.startsWith("[")) return text;
      return regexToBackrefTemplate(text);
    }
    function updateRunButtonLabel() {
      binaryRunBtn.textContent = "Run " + selectedTransform;
    }
    function resetTransformSearch() {
      transformSearchEl.value = "";
      renderTransforms();
    }

    function renderBinaryPanel() {
      if (!isBinaryTransform(selectedTransform)) {
        binaryPanel.classList.add("hidden");
        return;
      }
      const meta = getTransformMeta(selectedTransform);
      const cfg = getBinaryConfig(selectedTransform);
      cfg.from_ref = ensureRefObject(cfg.from_ref);
      cfg.to_ref = ensureRefObject(cfg.to_ref);
      binaryTitle.textContent = selectedTransform;
      binaryFromSummary.textContent = describeCommit(cfg.from_ref);
      binaryToSummary.textContent = describeCommit(cfg.to_ref);
      updateRunButtonLabel();
      binaryExtras.innerHTML = "";

      const fromAliasInput = document.createElement("input");
      fromAliasInput.placeholder = "from alias";
      fromAliasInput.value = cfg.from_ref.alias;
      fromAliasInput.addEventListener("input", () => {
        cfg.from_ref.alias = fromAliasInput.value;
        binaryFromSummary.textContent = describeCommit(cfg.from_ref);
      });
      binaryExtras.appendChild(fromAliasInput);

      const fromFilterInput = document.createElement("input");
      fromFilterInput.placeholder = "from filter (regex or JSON list)";
      fromFilterInput.value = cfg.from_ref.filter;
      fromFilterInput.addEventListener("input", () => {
        cfg.from_ref.filter = fromFilterInput.value;
        binaryFromSummary.textContent = describeCommit(cfg.from_ref);
      });
      binaryExtras.appendChild(fromFilterInput);

      const toAliasInput = document.createElement("input");
      toAliasInput.placeholder = "to alias";
      toAliasInput.value = cfg.to_ref.alias;
      toAliasInput.addEventListener("input", () => {
        cfg.to_ref.alias = toAliasInput.value;
        binaryToSummary.textContent = describeCommit(cfg.to_ref);
      });
      binaryExtras.appendChild(toAliasInput);

      const toFilterInput = document.createElement("input");
      toFilterInput.placeholder = "to filter (regex or JSON list)";
      toFilterInput.value = cfg.to_ref.filter;
      toFilterInput.addEventListener("input", () => {
        cfg.to_ref.filter = toFilterInput.value;
        binaryToSummary.textContent = describeCommit(cfg.to_ref);
      });
      binaryExtras.appendChild(toFilterInput);

      if (!meta.to_must_exist) {
        const copyBtn = document.createElement("button");
        copyBtn.textContent = "Copy from filter to to";
        copyBtn.addEventListener("click", () => {
          const next = copyFromFilterToToTemplate(cfg.from_ref.filter);
          cfg.to_ref.filter = next;
          if (!cfg.to_ref.alias.trim()) {
            cfg.to_ref.alias = cfg.from_ref.alias;
            toAliasInput.value = cfg.to_ref.alias;
          }
          toFilterInput.value = next;
          binaryToSummary.textContent = describeCommit(cfg.to_ref);
          setStatus("Copied from filter into to for " + selectedTransform + ".");
        });
        binaryExtras.appendChild(copyBtn);
      }

      const extraKeys = Array.isArray(meta.extra_keys) ? meta.extra_keys : [];
      const required = new Set(Array.isArray(meta.required_extra_keys) ? meta.required_extra_keys : []);
      for (const key of extraKeys) {
        const input = document.createElement("input");
        input.placeholder = key + (required.has(key) ? " (required)" : " (optional)");
        input.value = cfg.extras[key] == null ? "" : String(cfg.extras[key]);
        input.addEventListener("input", () => {
          cfg.extras[key] = input.value;
        });
        binaryExtras.appendChild(input);
      }
      binaryPanel.classList.remove("hidden");
    }

    function updatePanels() {
      const showLoad = selectedTransform === "load";
      const showBinary = isBinaryTransform(selectedTransform);
      loadPanel.classList.toggle("hidden", !showLoad);
      binaryPanel.classList.toggle("hidden", !showBinary);
      optionsEmpty.classList.toggle("hidden", showLoad || showBinary);
      renderBinaryPanel();
      if (selectedTransform === "load") {
        setStatus("Load is selected. Pick a file to import a model.");
      } else if (!isReadyTransform(selectedTransform)) {
        setStatus("Selected " + selectedTransform + " is planned and not interactive yet.");
      } else if (showBinary) {
        setStatus("Selected " + selectedTransform + ".");
      } else {
        setStatus("Selected " + selectedTransform + ".");
      }
    }

    function renderTransforms() {
      const query = transformSearchEl.value.trim().toLowerCase();
      const items = allTransforms.filter((item) => item.name.toLowerCase().includes(query));
      transformsEl.innerHTML = "";
      if (!items.length) {
        const row = document.createElement("div");
        row.className = "transform-item";
        row.textContent = "No transforms match your search.";
        transformsEl.appendChild(row);
        return;
      }
      for (const item of items) {
        const row = document.createElement("div");
        row.className = "transform-item" + (item.enabled ? "" : " planned") + (selectedTransform === item.name ? " selected" : "");
        const name = document.createElement("span");
        name.textContent = item.name;
        const badge = document.createElement("span");
        badge.className = "pill" + (item.enabled ? " enabled" : "");
        badge.textContent = item.enabled ? "ready" : "planned";
        row.appendChild(name);
        row.appendChild(badge);
        if (item.enabled) {
          row.addEventListener("click", () => {
            selectedTransform = item.name;
            renderTransforms();
            updatePanels();
            renderModels(latestModels);
          });
        }
        transformsEl.appendChild(row);
      }
    }

    function renderModels(models) {
      modelsEl.innerHTML = "";
      if (!models.length) {
        const empty = document.createElement("div");
        empty.className = "empty";
        empty.textContent = "No models loaded yet. Use load to import one.";
        modelsEl.appendChild(empty);
        return;
      }

      for (const model of models) {
        if (!modelViewState[model.alias]) {
          modelViewState[model.alias] = { format: "compact", filter: "", valid: true };
        }
        const state = modelViewState[model.alias];

        const pane = document.createElement("div");
        pane.className = "model-pane";

        const head = document.createElement("div");
        head.className = "model-head";
        const left = document.createElement("span");
        left.textContent = model.alias;
        const right = document.createElement("div");
        right.style.display = "flex";
        right.style.alignItems = "center";
        right.style.gap = "6px";
        const count = document.createElement("span");
        count.textContent = tensorCountText(
          model.matched_count || model.tensor_count,
          model.total_count || model.tensor_count,
          state.filter || ""
        );
        right.appendChild(count);
        head.appendChild(left);
        head.appendChild(right);

        const controls = document.createElement("div");
        controls.className = "model-controls";
        const formatSelect = document.createElement("select");
        formatSelect.innerHTML = "<option value='compact'>compact</option><option value='tree'>tree</option>";
        formatSelect.value = state.format;
        const filterInput = document.createElement("input");
        filterInput.placeholder = "regex or JSON list";
        filterInput.value = state.filter;
        const livePill = document.createElement("span");
        livePill.className = "live-pill";
        livePill.textContent = "live";
        const fromBtn = document.createElement("button");
        fromBtn.className = "mini-btn";
        fromBtn.textContent = "from";
        const toBtn = document.createElement("button");
        toBtn.className = "mini-btn";
        toBtn.textContent = "to";

        const updateCommitButtons = () => {
          const canCommit = isBinaryTransform(selectedTransform) && state.valid;
          fromBtn.classList.toggle("hidden", !canCommit);
          toBtn.classList.toggle("hidden", !(canCommit && toMustExist(selectedTransform)));
        };

        fromBtn.addEventListener("click", () => {
          if (!isBinaryTransform(selectedTransform) || !state.valid) return;
          const meta = getTransformMeta(selectedTransform);
          const cfg = getBinaryConfig(selectedTransform);
          cfg.from_ref = { alias: model.alias, filter: filterInput.value };
          if (!meta.to_must_exist) {
            cfg.to_ref = ensureRefObject(cfg.to_ref);
            if (!cfg.to_ref.alias.trim()) cfg.to_ref.alias = model.alias;
            cfg.to_ref.filter = copyFromFilterToToTemplate(filterInput.value);
          }
          renderBinaryPanel();
          setStatus("Committed from for " + selectedTransform + " from " + model.alias + ".");
        });
        toBtn.addEventListener("click", () => {
          if (!isBinaryTransform(selectedTransform) || !state.valid || !toMustExist(selectedTransform)) return;
          const cfg = getBinaryConfig(selectedTransform);
          cfg.to_ref = { alias: model.alias, filter: filterInput.value };
          renderBinaryPanel();
          setStatus("Committed to for " + selectedTransform + " from " + model.alias + ".");
        });

        const pre = document.createElement("pre");
        const dumps = {
          compact: model.dump_compact || "",
          tree: model.dump_tree || model.dump_compact || ""
        };
        pre.textContent = dumps[formatSelect.value] || "";

        let debounceHandle = null;
        const requestDump = async () => {
          state.format = formatSelect.value;
          state.filter = filterInput.value;
          setStatus("Applying filter for " + model.alias + "...");
          try {
            const response = await fetch("/api/model_dump", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({
                alias: model.alias,
                format: formatSelect.value,
                filter: filterInput.value
              })
            });
            const data = await response.json();
            if (!response.ok || !data.ok) {
              state.valid = false;
              filterInput.classList.add("invalid-field");
              updateCommitButtons();
              setStatus("Dump failed for " + model.alias + ": " + (data.error || "unknown error"));
              return;
            }
            state.valid = true;
            filterInput.classList.remove("invalid-field");
            pre.textContent = data.dump || "";
            count.textContent = tensorCountText(
              data.matched_count || 0,
              data.total_count || 0,
              filterInput.value
            );
            updateCommitButtons();
            setStatus("Updated dump for " + model.alias + ".");
          } catch (err) {
            state.valid = false;
            filterInput.classList.add("invalid-field");
            updateCommitButtons();
            setStatus("Dump failed for " + model.alias + ": " + String(err));
          }
        };

        formatSelect.addEventListener("change", () => requestDump());
        filterInput.addEventListener("input", () => {
          clearTimeout(debounceHandle);
          debounceHandle = setTimeout(requestDump, 220);
        });

        controls.appendChild(formatSelect);
        controls.appendChild(filterInput);
        controls.appendChild(livePill);
        controls.appendChild(fromBtn);
        controls.appendChild(toBtn);
        updateCommitButtons();

        pane.appendChild(head);
        pane.appendChild(controls);
        pane.appendChild(pre);
        modelsEl.appendChild(pane);
      }
    }

    async function refresh() {
      try {
        const [transformsRes, stateRes] = await Promise.all([
          fetch("/api/transforms"),
          fetch("/api/state")
        ]);
        const transformsData = await transformsRes.json();
        const stateData = await stateRes.json();

        if (transformsData.ok && Array.isArray(transformsData.transforms)) {
          allTransforms = transformsData.transforms;
          if (!allTransforms.some((item) => item.name === selectedTransform && item.enabled)) {
            selectedTransform = allTransforms.some((item) => item.name === "load" && item.enabled)
              ? "load"
              : (allTransforms.find((item) => item.enabled)?.name || "load");
          }
        }
        renderTransforms();

        if (stateData.ok) {
          latestModels = stateData.models || [];
        }
        renderModels(latestModels);
        updatePanels();
      } catch (err) {
        setStatus("Refresh failed: " + String(err));
        renderTransforms();
        updatePanels();
      }
    }

    loadBtn.addEventListener("click", async () => {
      if (selectedTransform !== "load") {
        setStatus("Select load first.");
        return;
      }
      const file = fileInput.files && fileInput.files[0];
      if (!file) {
        setStatus("Pick a model file first.");
        return;
      }

      setStatus("Reading file...");
      loadBtn.disabled = true;
      try {
        const payload = { alias: aliasInput.value || null };
        const bytes = new Uint8Array(await file.arrayBuffer());
        let binary = "";
        const chunk = 0x8000;
        for (let i = 0; i < bytes.length; i += chunk) {
          binary += String.fromCharCode(...bytes.subarray(i, i + chunk));
        }
        payload.filename = file.name;
        payload.content_b64 = btoa(binary);

        setStatus("Loading model via transform...");
        const response = await fetch("/api/load", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload)
        });
        const data = await response.json();
        if (!response.ok || !data.ok) {
          setStatus("Load failed: " + (data.error || "unknown error"));
          return;
        }
        latestModels = data.models || [];
        renderModels(latestModels);
        resetTransformSearch();
        setStatus("Load completed successfully.");
      } catch (err) {
        setStatus("Load failed: " + String(err));
      } finally {
        loadBtn.disabled = false;
      }
    });

    binaryRunBtn.addEventListener("click", async () => {
      if (!isBinaryTransform(selectedTransform)) {
        setStatus("Select a READY transform first.");
        return;
      }
      const meta = getTransformMeta(selectedTransform);
      const cfg = getBinaryConfig(selectedTransform);
      cfg.from_ref = ensureRefObject(cfg.from_ref);
      cfg.to_ref = ensureRefObject(cfg.to_ref);
      if (!cfg.from_ref.alias.trim() || !cfg.to_ref.alias.trim()) {
        setStatus("Set both from and to aliases before running " + selectedTransform + ".");
        return;
      }

      const required = new Set(Array.isArray(meta.required_extra_keys) ? meta.required_extra_keys : []);
      const extrasPayload = {};
      for (const key of (Array.isArray(meta.extra_keys) ? meta.extra_keys : [])) {
        const parsed = parseExtraValue(cfg.extras[key] == null ? "" : String(cfg.extras[key]));
        if (parsed === undefined) {
          if (required.has(key)) {
            setStatus("Missing required parameter: " + key);
            return;
          }
          continue;
        }
        extrasPayload[key] = parsed;
      }

      setStatus("Applying " + selectedTransform + "...");
      binaryRunBtn.disabled = true;
      try {
        const response = await fetch("/api/apply_binary_transform", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            transform: selectedTransform,
            from_ref: cfg.from_ref,
            to_ref: cfg.to_ref,
            extras: extrasPayload
          })
        });
        const data = await response.json();
        if (!response.ok || !data.ok) {
          setStatus("Apply failed: " + (data.error || "unknown error"));
          return;
        }
        latestModels = data.models || [];
        renderModels(latestModels);
        resetTransformSearch();
        setStatus("Applied " + selectedTransform + " successfully.");
      } catch (err) {
        setStatus("Apply failed: " + String(err));
      } finally {
        binaryRunBtn.disabled = false;
      }
    });

    transformSearchEl.addEventListener("input", () => renderTransforms());

    transformSearchEl.value = "";
    renderTransforms();
    updatePanels();
    refresh().catch((err) => setStatus("Initial load failed: " + String(err)));
  </script>
</body>
</html>
"""


__all__ = ["serve_webui2"]
