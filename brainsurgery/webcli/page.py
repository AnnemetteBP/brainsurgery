_HTML_PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>BRAINSURGERY WEB CLI</title>
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
    <div class="head"><h1>BRAINSURGERY WEB CLI</h1></div>
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
