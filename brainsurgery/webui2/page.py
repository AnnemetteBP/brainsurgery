from __future__ import annotations

HTML_PAGE = """<!doctype html>
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
    .right-stack {
      display: grid;
      grid-template-rows: minmax(0, 1fr) auto;
      gap: 12px;
      align-content: start;
      height: calc(100vh - 165px);
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
      max-height: 100%;
      overflow: auto;
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
      grid-template-columns: 96px 92px 1fr auto;
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
    .commit-wrap {
      display: flex;
      align-items: center;
      gap: 4px;
      justify-content: flex-end;
      flex-wrap: wrap;
    }
    .secondary-btn {
      color: #4b3a24;
      border: 1px solid #d8c4a4;
      background: linear-gradient(130deg, #fdf2dd, #f5e4c6);
    }
    .secondary-btn:hover {
      background: linear-gradient(130deg, #fbeccb, #f1ddb8);
    }
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
    .panel-row {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
      margin-bottom: 8px;
    }
    .panel-row .title {
      margin: 0;
    }
    #clearOptionsBtn,
    #optionsToggleBtn,
    #resultsToggleBtn,
    #clearResultsBtn {
      margin: 0;
      width: auto;
      padding: 6px 10px;
      font-size: 12px;
    }
    .results-actions {
      display: flex;
      align-items: center;
      gap: 6px;
    }
    #resultsOutput {
      margin: 0;
      border: 1px solid #dcc8aa;
      border-radius: 10px;
      padding: 10px;
      min-height: 120px;
      max-height: 220px;
      overflow: auto;
      white-space: pre-wrap;
      background: #fffdfa;
    }
    @media (max-width: 980px) {
      .main { grid-template-columns: 1fr; }
      .left-stack { height: auto; grid-template-rows: auto auto; }
      .right-stack { height: auto; grid-template-rows: auto auto; }
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
          <div class=\"panel-row\">
            <h2 class=\"title\">Transform Options</h2>
            <div class=\"results-actions\">
              <button id=\"clearOptionsBtn\" class=\"secondary-btn\">Clear</button>
              <button id=\"optionsToggleBtn\" class=\"secondary-btn\">Collapse</button>
            </div>
          </div>
          <div id=\"optionsPanelBody\">
            <div id=\"optionsEmpty\" class=\"options-placeholder hidden\"></div>
            <div id=\"loadPanel\" class=\"hidden\">
              <h2 class=\"title\">Load</h2>
              <input id=\"aliasInput\" placeholder=\"Alias (optional, defaults to model/model_2/...)\" />
              <input id=\"fileInput\" type=\"file\" />
              <button id=\"loadBtn\">Load Selected File</button>
            </div>
            <div id=\"transformPanel\" class=\"hidden\">
              <h2 id=\"transformTitle\" class=\"title\">Transform</h2>
              <div id=\"transformFields\"></div>
              <button id=\"transformRunBtn\">Run Transform</button>
            </div>
            <div id=\"status\">Ready.</div>
          </div>
        </div>
      </div>
      <div class=\"right-stack\">
        <div class=\"card\">
          <h2 class=\"title\">Current Models</h2>
          <div id=\"models\" class=\"models\"></div>
        </div>
        <div class=\"card\">
          <div class=\"panel-row\">
            <h2 class=\"title\">Results</h2>
            <div class=\"results-actions\">
              <button id=\"resultsToggleBtn\" class=\"secondary-btn\">Collapse</button>
              <button id=\"clearResultsBtn\" class=\"secondary-btn\">Clear</button>
            </div>
          </div>
          <div id=\"resultsPanelBody\">
            <pre id=\"resultsOutput\">(no transform output yet)</pre>
          </div>
        </div>
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
    const transformPanel = document.getElementById("transformPanel");
    const transformTitle = document.getElementById("transformTitle");
    const transformFields = document.getElementById("transformFields");
    const transformRunBtn = document.getElementById("transformRunBtn");
    const clearOptionsBtn = document.getElementById("clearOptionsBtn");
    const optionsToggleBtn = document.getElementById("optionsToggleBtn");
    const optionsPanelBody = document.getElementById("optionsPanelBody");
    const resultsToggleBtn = document.getElementById("resultsToggleBtn");
    const clearResultsBtn = document.getElementById("clearResultsBtn");
    const resultsPanelBody = document.getElementById("resultsPanelBody");
    const resultsOutput = document.getElementById("resultsOutput");

    let allTransforms = [{ name: "load", enabled: true, kind: "other", allowed_keys: [], required_keys: [], reference_keys: [], to_must_exist: false }];
    let selectedTransform = "load";
    const modelViewState = {};
    const transformConfigByName = {};
    let latestModels = [];

    function setStatus(text) { statusEl.textContent = text; }
    function appendResultBlock(title, text) {
      const block = "[" + title + "]\\n" + ((text && text.trim()) ? text : "(no output)") + "\\n\\n";
      if (resultsOutput.textContent === "(no transform output yet)") {
        resultsOutput.textContent = "";
      }
      resultsOutput.textContent += block;
      resultsOutput.scrollTop = resultsOutput.scrollHeight;
    }
    function tensorCountText(shown, total, filterText) {
      return filterText.trim() ? (shown + " out of " + total + " tensors") : (total + " tensors");
    }
    function parseFieldValue(raw) {
      const text = (raw || "").trim();
      if (!text) return undefined;
      try { return JSON.parse(text); } catch (_err) { return text; }
    }
    function getTransformMeta(name) {
      return allTransforms.find((t) => t.name === name) || null;
    }
    function isReadyTransform(name) {
      const meta = getTransformMeta(name);
      return !!(meta && meta.enabled);
    }
    function isRunnableTransform(name) {
      return !!name && name !== "load" && isReadyTransform(name);
    }
    function getTransformConfig(name) {
      if (!transformConfigByName[name]) {
        transformConfigByName[name] = { fields: {} };
      }
      return transformConfigByName[name];
    }
    function resetTransformSearch() {
      transformSearchEl.value = "";
      renderTransforms();
    }
    function regexToBackrefTemplate(raw) {
      let count = 0;
      let out = "";
      let escaped = false;
      let inClass = false;
      for (let i = 0; i < raw.length; i += 1) {
        const ch = raw[i];
        if (escaped) { out += "\\\\" + ch; escaped = false; continue; }
        if (ch === "\\\\") { escaped = true; continue; }
        if (ch === "[") { inClass = true; out += ch; continue; }
        if (ch === "]") { inClass = false; out += ch; continue; }
        if (!inClass && ch === "(") {
          const next = raw.slice(i + 1, i + 3);
          if (next === "?:" || next === "?=" || next === "?!" || next === "?<" || next === "?>") { out += ch; continue; }
          count += 1;
          out += "\\\\" + String(count);
          let depth = 1;
          let j = i + 1;
          let esc2 = false;
          let cls2 = false;
          for (; j < raw.length; j += 1) {
            const c2 = raw[j];
            if (esc2) { esc2 = false; continue; }
            if (c2 === "\\\\") { esc2 = true; continue; }
            if (c2 === "[") { cls2 = true; continue; }
            if (c2 === "]") { cls2 = false; continue; }
            if (cls2) continue;
            if (c2 === "(") depth += 1;
            if (c2 === ")") { depth -= 1; if (depth === 0) break; }
          }
          i = j;
          continue;
        }
        out += ch;
      }
      return count ? out : raw;
    }
    function copyFromFilterToToTemplate(raw) {
      const text = (raw || "").trim();
      if (!text) return "";
      if (text.startsWith("[")) return text;
      return regexToBackrefTemplate(text);
    }
    function buildReferenceFromModel(alias, filterText) {
      const expr = (filterText || "").trim() || ".*";
      return alias + "::" + expr;
    }
    function commitRefFromModel(key, alias, filterText) {
      const meta = getTransformMeta(selectedTransform);
      if (!meta) return;
      const cfg = getTransformConfig(selectedTransform);
      cfg.fields[key] = buildReferenceFromModel(alias, filterText);
      if (meta.kind === "binary" && key === "from" && !meta.to_must_exist) {
        const templ = copyFromFilterToToTemplate((filterText || "").trim());
        cfg.fields.to = alias + "::" + (templ || ".*");
      }
      renderTransformPanel();
      setStatus("Committed " + key + " for " + selectedTransform + " from " + alias + ".");
    }

    function renderTransformPanel() {
      if (!isRunnableTransform(selectedTransform)) {
        transformPanel.classList.add("hidden");
        return;
      }
      const meta = getTransformMeta(selectedTransform);
      const cfg = getTransformConfig(selectedTransform);
      const allowed = Array.isArray(meta.allowed_keys) ? meta.allowed_keys : [];
      const required = new Set(Array.isArray(meta.required_keys) ? meta.required_keys : []);
      const refKeys = Array.isArray(meta.reference_keys) ? meta.reference_keys : [];
      const refSet = new Set(refKeys);
      const orderedKeys = [...refKeys, ...allowed.filter((k) => !refSet.has(k))];
      transformTitle.textContent = selectedTransform;
      transformFields.innerHTML = "";
      transformRunBtn.textContent = "Run " + selectedTransform;

      for (const key of orderedKeys) {
        const input = document.createElement("input");
        const suffix = required.has(key) ? "required" : "optional";
        input.placeholder = key + " (" + suffix + ")";
        input.value = cfg.fields[key] == null ? "" : String(cfg.fields[key]);
        input.addEventListener("input", () => { cfg.fields[key] = input.value; });
        transformFields.appendChild(input);
      }

      if (meta.kind === "binary" && !meta.to_must_exist && refSet.has("from") && refSet.has("to")) {
        const copyBtn = document.createElement("button");
        copyBtn.className = "secondary-btn";
        copyBtn.textContent = "Copy from filter to to";
        copyBtn.addEventListener("click", () => {
          const fromRaw = String(cfg.fields.from || "");
          const sep = fromRaw.indexOf("::");
          if (sep < 0) {
            setStatus("Set from as alias::regex first.");
            return;
          }
          const alias = fromRaw.slice(0, sep);
          const expr = fromRaw.slice(sep + 2);
          const templ = copyFromFilterToToTemplate(expr);
          cfg.fields.to = alias + "::" + (templ || ".*");
          renderTransformPanel();
          setStatus("Copied from filter into to for " + selectedTransform + ".");
        });
        transformFields.appendChild(copyBtn);
      }

      transformPanel.classList.remove("hidden");
    }

    function updatePanels() {
      const showLoad = selectedTransform === "load";
      const showTransform = isRunnableTransform(selectedTransform);
      const hasSelection = !!selectedTransform;
      loadPanel.classList.toggle("hidden", !showLoad);
      transformPanel.classList.toggle("hidden", !showTransform);
      optionsEmpty.classList.toggle("hidden", hasSelection);
      renderTransformPanel();
      if (selectedTransform === "load") {
        setStatus("Load is selected. Pick a file to import a model.");
      } else if (!isReadyTransform(selectedTransform)) {
        setStatus("Selected " + selectedTransform + " is planned and not interactive yet.");
      } else if (!hasSelection) {
        setStatus("Ready.");
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
          modelViewState[model.alias] = { format: "compact", verbosity: "shape", filter: "", valid: true };
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
        count.textContent = tensorCountText(model.matched_count || model.tensor_count, model.total_count || model.tensor_count, state.filter || "");
        right.appendChild(count);
        head.appendChild(left);
        head.appendChild(right);

        const controls = document.createElement("div");
        controls.className = "model-controls";
        const formatSelect = document.createElement("select");
        formatSelect.innerHTML = "<option value='compact'>compact</option><option value='tree'>tree</option>";
        formatSelect.value = state.format;
        const verbositySelect = document.createElement("select");
        verbositySelect.innerHTML = "<option value='shape'>shape</option><option value='stat'>stat</option>";
        verbositySelect.value = state.verbosity || "shape";
        const filterInput = document.createElement("input");
        filterInput.placeholder = "regex or JSON list";
        filterInput.value = state.filter;
        const livePill = document.createElement("span");
        livePill.className = "live-pill";
        livePill.textContent = "live";
        const commitWrap = document.createElement("div");
        commitWrap.className = "commit-wrap";

        const renderCommitButtons = () => {
          commitWrap.innerHTML = "";
          const meta = getTransformMeta(selectedTransform);
          const canCommit = !!meta && isRunnableTransform(selectedTransform) && state.valid;
          if (!canCommit) return;
          const keys = Array.isArray(meta.reference_keys) ? meta.reference_keys : [];
          for (const key of keys) {
            if (meta.kind === "binary" && key === "to" && !meta.to_must_exist) continue;
            const btn = document.createElement("button");
            btn.className = "mini-btn";
            btn.textContent = key;
            btn.addEventListener("click", () => commitRefFromModel(key, model.alias, filterInput.value));
            commitWrap.appendChild(btn);
          }
        };

        const pre = document.createElement("pre");
        const dumps = { compact: model.dump_compact || "", tree: model.dump_tree || model.dump_compact || "" };
        pre.textContent = dumps[formatSelect.value] || "";

        let debounceHandle = null;
        const requestDump = async () => {
          state.format = formatSelect.value;
          state.verbosity = verbositySelect.value;
          state.filter = filterInput.value;
          setStatus("Applying filter for " + model.alias + "...");
          try {
            const response = await fetch("/api/model_dump", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({
                alias: model.alias,
                format: formatSelect.value,
                verbosity: verbositySelect.value,
                filter: filterInput.value
              })
            });
            const data = await response.json();
            if (!response.ok || !data.ok) {
              state.valid = false;
              filterInput.classList.add("invalid-field");
              renderCommitButtons();
              setStatus("Dump failed for " + model.alias + ": " + (data.error || "unknown error"));
              return;
            }
            state.valid = true;
            filterInput.classList.remove("invalid-field");
            pre.textContent = data.dump || "";
            count.textContent = tensorCountText(data.matched_count || 0, data.total_count || 0, filterInput.value);
            renderCommitButtons();
            setStatus("Updated dump for " + model.alias + ".");
          } catch (err) {
            state.valid = false;
            filterInput.classList.add("invalid-field");
            renderCommitButtons();
            setStatus("Dump failed for " + model.alias + ": " + String(err));
          }
        };

        formatSelect.addEventListener("change", () => requestDump());
        verbositySelect.addEventListener("change", () => requestDump());
        filterInput.addEventListener("input", () => {
          clearTimeout(debounceHandle);
          debounceHandle = setTimeout(requestDump, 220);
        });

        controls.appendChild(formatSelect);
        controls.appendChild(verbositySelect);
        controls.appendChild(filterInput);
        controls.appendChild(livePill);
        controls.appendChild(commitWrap);
        renderCommitButtons();

        pane.appendChild(head);
        pane.appendChild(controls);
        pane.appendChild(pre);
        modelsEl.appendChild(pane);
      }
    }

    async function refresh() {
      try {
        const [transformsRes, stateRes] = await Promise.all([fetch("/api/transforms"), fetch("/api/state")]);
        const transformsData = await transformsRes.json();
        const stateData = await stateRes.json();
        if (transformsData.ok && Array.isArray(transformsData.transforms)) {
          allTransforms = transformsData.transforms;
          if (selectedTransform && !allTransforms.some((item) => item.name === selectedTransform && item.enabled)) {
            selectedTransform = allTransforms.some((item) => item.name === "load" && item.enabled)
              ? "load"
              : (allTransforms.find((item) => item.enabled)?.name || "");
          }
        }
        renderTransforms();
        if (stateData.ok) latestModels = stateData.models || [];
        renderModels(latestModels);
        updatePanels();
      } catch (err) {
        setStatus("Refresh failed: " + String(err));
        renderTransforms();
        updatePanels();
      }
    }

    loadBtn.addEventListener("click", async () => {
      if (selectedTransform !== "load") { setStatus("Select load first."); return; }
      const file = fileInput.files && fileInput.files[0];
      if (!file) { setStatus("Pick a model file first."); return; }

      setStatus("Reading file...");
      loadBtn.disabled = true;
      try {
        const payload = { alias: aliasInput.value || null };
        const bytes = new Uint8Array(await file.arrayBuffer());
        let binary = "";
        const chunk = 0x8000;
        for (let i = 0; i < bytes.length; i += chunk) binary += String.fromCharCode(...bytes.subarray(i, i + chunk));
        payload.filename = file.name;
        payload.content_b64 = btoa(binary);

        setStatus("Loading model via transform...");
        const response = await fetch("/api/load", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload)
        });
        const data = await response.json();
        if (!response.ok || !data.ok) { setStatus("Load failed: " + (data.error || "unknown error")); return; }
        latestModels = data.models || [];
        renderModels(latestModels);
        resetTransformSearch();
        renderTransforms();
        updatePanels();
        setStatus("Load completed successfully.");
      } catch (err) {
        setStatus("Load failed: " + String(err));
      } finally {
        loadBtn.disabled = false;
      }
    });

    transformRunBtn.addEventListener("click", async () => {
      if (!isRunnableTransform(selectedTransform)) { setStatus("Select a READY transform first."); return; }
      const runTransformName = selectedTransform;
      const meta = getTransformMeta(runTransformName);
      const cfg = getTransformConfig(runTransformName);
      const allowed = Array.isArray(meta.allowed_keys) ? meta.allowed_keys : [];
      const required = new Set(Array.isArray(meta.required_keys) ? meta.required_keys : []);
      const payload = {};

      for (const key of allowed) {
        const parsed = parseFieldValue(cfg.fields[key] == null ? "" : String(cfg.fields[key]));
        if (parsed === undefined) {
          if (required.has(key)) { setStatus("Missing required parameter: " + key); return; }
          continue;
        }
        payload[key] = parsed;
      }

      setStatus("Applying " + runTransformName + "...");
      transformRunBtn.disabled = true;
      try {
        const response = await fetch("/api/apply_transform", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ transform: runTransformName, payload: payload })
        });
        const data = await response.json();
        if (!response.ok || !data.ok) { setStatus("Apply failed: " + (data.error || "unknown error")); return; }
        latestModels = data.models || [];
        renderModels(latestModels);
        resetTransformSearch();
        renderTransforms();
        updatePanels();
        setStatus("Applied " + runTransformName + " successfully.");
        appendResultBlock(runTransformName, data.output || "");
      } catch (err) {
        setStatus("Apply failed: " + String(err));
      } finally {
        transformRunBtn.disabled = false;
      }
    });

    clearResultsBtn.addEventListener("click", () => {
      resultsOutput.textContent = "(no transform output yet)";
      setStatus("Results cleared.");
    });
    resultsToggleBtn.addEventListener("click", () => {
      const collapsing = !resultsPanelBody.classList.contains("hidden");
      resultsPanelBody.classList.toggle("hidden", collapsing);
      resultsToggleBtn.textContent = collapsing ? "Expand" : "Collapse";
    });
    optionsToggleBtn.addEventListener("click", () => {
      const collapsing = !optionsPanelBody.classList.contains("hidden");
      optionsPanelBody.classList.toggle("hidden", collapsing);
      optionsToggleBtn.textContent = collapsing ? "Expand" : "Collapse";
    });
    clearOptionsBtn.addEventListener("click", () => {
      if (selectedTransform === "load") {
        aliasInput.value = "";
        fileInput.value = "";
        setStatus("Cleared load options.");
        return;
      }
      if (isRunnableTransform(selectedTransform)) {
        const cfg = getTransformConfig(selectedTransform);
        cfg.fields = {};
        renderTransformPanel();
        setStatus("Cleared " + selectedTransform + " options.");
        return;
      }
      setStatus("Nothing to clear.");
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

__all__ = ["HTML_PAGE"]
