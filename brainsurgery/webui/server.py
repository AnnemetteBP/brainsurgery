from __future__ import annotations

from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
from pathlib import Path
import threading
from typing import Any

from omegaconf import OmegaConf
import yaml

from ..cli.interactive import normalize_transform_specs
from ..cli.summary import build_raw_plan
from ..engine import (
    ProviderError,
    compile_plan,
    create_state_dict_provider,
    execute_transform_pairs,
    get_runtime_flags,
    reset_runtime_flags,
    use_output_emitter,
)


logger = logging.getLogger("brainsurgery")
_run_lock = threading.Lock()


@dataclass(frozen=True)
class WebRunResult:
    ok: bool
    logs: list[str]
    output_lines: list[str]
    executed_transforms: list[dict[str, Any]]
    summary_yaml: str | None
    written_path: str | None
    error: str | None = None


def run_web_plan(
    *,
    plan_yaml: str,
    shard_size: str,
    num_workers: int,
    provider: str,
    arena_root: Path,
    arena_segment_size: str,
    summarize: bool,
    log_level: str,
) -> WebRunResult:
    raw = yaml.safe_load(plan_yaml) if plan_yaml.strip() else {}
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("Plan YAML root must be a mapping.")

    reset_runtime_flags()
    _configure_logging(log_level=log_level)

    logs: list[str] = []
    output_lines: list[str] = []
    executed_transforms: list[dict[str, Any]] = []
    summary_yaml: str | None = None
    written_path: str | None = None

    log_handler = _ListLogHandler(logs)
    log_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s"))
    root_logger = logging.getLogger()
    root_logger.addHandler(log_handler)
    logger.addHandler(log_handler)

    state_dict_provider = None
    try:
        surgery_plan = compile_plan(raw)
        state_dict_provider = create_state_dict_provider(
            provider=provider,
            model_paths=surgery_plan.inputs,
            max_io_workers=num_workers,
            arena_root=arena_root,
            arena_segment_size=arena_segment_size,
        )
        configured_pairs = zip(
            normalize_transform_specs(raw.get("transforms")),
            surgery_plan.transforms,
            strict=False,
        )

        with use_output_emitter(output_lines.append):
            _, executed_transforms = execute_transform_pairs(
                configured_pairs,
                state_dict_provider,
                interactive=False,
            )

            if surgery_plan.output is None:
                logger.info("No output configured; execution finished without save.")
            elif get_runtime_flags().dry_run:
                logger.info("Dry-run enabled; skipping output save.")
            else:
                persisted = state_dict_provider.save_output(
                    surgery_plan,
                    default_shard_size=shard_size,
                    max_io_workers=num_workers,
                )
                written_path = str(persisted)
                logger.info("Output saved to %s", written_path)

        if summarize:
            summary_doc = build_raw_plan(
                inputs=raw.get("inputs", []),
                output=raw.get("output"),
                transforms=executed_transforms,
            )
            summary_yaml = OmegaConf.to_yaml(summary_doc)

        return WebRunResult(
            ok=True,
            logs=logs,
            output_lines=output_lines,
            executed_transforms=executed_transforms,
            summary_yaml=summary_yaml,
            written_path=written_path,
        )
    finally:
        if state_dict_provider is not None:
            state_dict_provider.close()
        root_logger.removeHandler(log_handler)
        logger.removeHandler(log_handler)


def serve_webui(*, host: str, port: int) -> None:
    server = ThreadingHTTPServer((host, port), _handler_factory())
    logger.info("Brain surgery web UI available at http://%s:%d", host, port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down web UI server")
    finally:
        server.server_close()


def _handler_factory():
    class _Handler(BaseHTTPRequestHandler):
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
            logger.debug("webui request: " + format, *args)

    return _Handler


def _configure_logging(*, log_level: str) -> None:
    allowed = {"debug", "info", "warning", "error", "critical"}
    level_name = log_level.strip().lower()
    if level_name not in allowed:
        raise ValueError(f"log_level must be one of: {', '.join(sorted(allowed))}")
    level = getattr(logging, level_name.upper())
    logging.getLogger().setLevel(level)
    logger.setLevel(level)


def _as_string(value: Any, key: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string.")
    return value


def _as_int(value: Any, key: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be an integer.")
    return value


class _ListLogHandler(logging.Handler):
    def __init__(self, sink: list[str]) -> None:
        super().__init__()
        self._sink = sink

    def emit(self, record: logging.LogRecord) -> None:
        self._sink.append(self.format(record))


_HTML_PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>BrainSurgery Web UI</title>
  <style>
    :root {
      --ink: #1a232d;
      --paper: #fff9f0;
      --line: #d4be98;
      --accent: #d7632a;
      --ok: #0a7d3b;
      --err: #a21d24;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: radial-gradient(circle at top right, #ffe7c6, #fffdf8 60%);
      color: var(--ink);
      font-family: "Avenir Next", "Trebuchet MS", sans-serif;
      padding: 20px;
    }
    .shell {
      max-width: 1100px;
      margin: 0 auto;
      background: var(--paper);
      border: 1px solid var(--line);
      border-radius: 16px;
      overflow: hidden;
    }
    .head { padding: 16px 18px; border-bottom: 1px solid var(--line); }
    .head h1 { margin: 0; font-family: "Futura", sans-serif; font-size: 24px; text-transform: uppercase; }
    .main { padding: 14px; display: grid; grid-template-columns: 1.2fr 1fr; gap: 12px; }
    textarea, input, select {
      width: 100%;
      border: 1px solid #c3aa82;
      border-radius: 10px;
      padding: 9px;
      font-size: 14px;
    }
    textarea { min-height: 420px; font-family: "SFMono-Regular", "Menlo", monospace; }
    .card { border: 1px solid var(--line); border-radius: 12px; background: #fffefb; padding: 12px; }
    .row { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-bottom: 8px; }
    button {
      border: 0;
      border-radius: 10px;
      padding: 10px 12px;
      color: white;
      font-weight: 700;
      background: linear-gradient(135deg, var(--accent), #ec8c52);
      cursor: pointer;
    }
    pre {
      margin: 0;
      min-height: 120px;
      max-height: 220px;
      overflow: auto;
      border: 1px solid #d6c3a4;
      border-radius: 10px;
      padding: 8px;
      font-family: "SFMono-Regular", "Menlo", monospace;
      font-size: 12px;
      background: #fffdfa;
      line-height: 1.45;
    }
    .status { font-weight: 700; margin-left: 8px; }
    .ok { color: var(--ok); }
    .err { color: var(--err); }
    @media (max-width: 980px) { .main { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
  <div class="shell">
    <div class="head"><h1>BrainSurgery Web UI</h1></div>
    <div class="main">
      <div class="card">
        <label>Plan YAML</label>
        <textarea id="planYaml">inputs:
  - model::/path/to/input.safetensors
transforms:
  - dump: { target: ".*", format: compact }
output:
  path: /path/to/output.safetensors
</textarea>
      </div>
      <div class="card">
        <div class="row">
          <div><label>Provider</label><select id="provider"><option>inmemory</option><option>arena</option></select></div>
          <div><label>Workers</label><input id="numWorkers" type="number" min="1" value="8"/></div>
        </div>
        <div class="row">
          <div><label>Shard Size</label><input id="shardSize" value="5GB"/></div>
          <div><label>Arena Segment</label><input id="arenaSegmentSize" value="1GB"/></div>
        </div>
        <div class="row">
          <div><label>Arena Root</label><input id="arenaRoot" value=".brainsurgery"/></div>
          <div><label>Log Level</label><select id="logLevel"><option>info</option><option>debug</option><option>warning</option><option>error</option><option>critical</option></select></div>
        </div>
        <div style="display:flex; align-items:center; gap:8px; margin-bottom:8px;">
          <button id="runBtn">Run</button>
          <label style="display:flex; align-items:center; gap:6px;"><input id="summarize" type="checkbox" checked style="width:auto;"/>Summary</label>
          <span id="status" class="status"></span>
        </div>
        <label>Output</label><pre id="outputPane"></pre>
        <label>Logs</label><pre id="logPane"></pre>
        <label>Summary</label><pre id="summaryPane"></pre>
      </div>
    </div>
  </div>
  <script>
    const runBtn = document.getElementById("runBtn");
    const statusEl = document.getElementById("status");
    const outputPane = document.getElementById("outputPane");
    const logPane = document.getElementById("logPane");
    const summaryPane = document.getElementById("summaryPane");
    function status(text, cls) {
      statusEl.textContent = text;
      statusEl.className = "status " + (cls || "");
    }
    runBtn.addEventListener("click", async () => {
      runBtn.disabled = true;
      status("Running...", "");
      outputPane.textContent = "";
      logPane.textContent = "";
      summaryPane.textContent = "";
      const payload = {
        plan_yaml: document.getElementById("planYaml").value,
        provider: document.getElementById("provider").value,
        num_workers: Number(document.getElementById("numWorkers").value),
        shard_size: document.getElementById("shardSize").value,
        arena_root: document.getElementById("arenaRoot").value,
        arena_segment_size: document.getElementById("arenaSegmentSize").value,
        summarize: document.getElementById("summarize").checked,
        log_level: document.getElementById("logLevel").value
      };
      try {
        const response = await fetch("/api/run", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify(payload)
        });
        const data = await response.json();
        if (!response.ok || !data.ok) {
          status("Failed", "err");
          logPane.textContent = data.error || "Unknown error";
          return;
        }
        const out = [];
        if (data.written_path) out.push("Saved output to: " + data.written_path, "");
        out.push(...(data.output_lines || []));
        outputPane.textContent = out.join("\\n");
        logPane.textContent = (data.logs || []).join("\\n");
        summaryPane.textContent = data.summary_yaml || "(summary disabled)";
        status("Completed", "ok");
      } catch (err) {
        status("Failed", "err");
        logPane.textContent = String(err);
      } finally {
        runBtn.disabled = false;
      }
    });
  </script>
</body>
</html>
"""


__all__ = ["WebRunResult", "run_web_plan", "serve_webui"]
