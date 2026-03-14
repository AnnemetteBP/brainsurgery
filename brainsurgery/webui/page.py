from __future__ import annotations

HTML_PAGE = """<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>BrainSurgery WebUI</title>
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
    .focus-scope {
      position: relative;
      transition: box-shadow 0.12s ease, border-color 0.12s ease;
    }
    .focus-scope.focused {
      border-color: #d45d1f;
      box-shadow: 0 0 0 2px rgba(212, 93, 31, 0.3);
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
    .transform-item:focus-visible {
      outline: none;
      border-color: #d45d1f;
      box-shadow: inset 0 0 0 1px #d45d1f;
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
    input, button, select, textarea {
      width: 100%;
      border-radius: 8px;
      border: 1px solid #c7af8a;
      padding: 8px 10px;
      font-size: 13px;
      margin-bottom: 8px;
      background: #fff;
    }
    input:focus-visible,
    button:focus-visible,
    select:focus-visible,
    textarea:focus-visible {
      outline: none;
      border-color: #d45d1f;
      box-shadow: inset 0 0 0 1px #d45d1f;
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
      min-height: 0;
      overflow: auto;
    }
    #modelsPanel {
      display: flex;
      flex-direction: column;
      min-height: 0;
      max-height: calc(100vh - 255px);
      overflow: hidden;
    }
    #models {
      flex: 1 1 auto;
      min-height: 0;
      overflow: auto;
    }
    .model-pane {
      border: 1px solid var(--line);
      border-radius: 10px;
      background: #fffefb;
      overflow: visible;
    }
    .model-head {
      padding: 6px 8px;
      border-bottom: 1px solid var(--line);
      font-weight: 700;
      font-size: 12px;
      display: flex;
      justify-content: space-between;
    }
    .model-head .toggle-dump-btn {
      margin: 0;
      width: auto;
      padding: 4px 8px;
      font-size: 11px;
    }
    .model-controls {
      padding: 6px 8px;
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
      padding: 8px;
      overflow-x: auto;
      overflow-y: visible;
      max-height: none;
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
    #copyResultsBtn,
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
    .checkbox-row {
      display: flex;
      align-items: center;
      gap: 8px;
      margin-bottom: 8px;
    }
    .checkbox-row input[type="checkbox"] {
      width: auto;
      margin: 0;
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
      #modelsPanel {
        max-height: none;
      }
    }
  </style>
</head>
<body>
  <div class=\"shell\">
    <div class=\"head\">
      <h1>BrainSurgery WebUI</h1>
    </div>
    <div class=\"main\">
      <div class=\"left-stack\">
        <div id=\"transformsPanel\" class=\"card picker-card focus-scope\" tabindex=\"0\">
          <h2 class=\"title\">Transforms</h2>
          <input id=\"transformSearch\" placeholder=\"Search transforms (e.g. load)\" />
          <div id=\"transformList\" class=\"transform-list\"></div>
        </div>
        <div id=\"optionsPanel\" class=\"card focus-scope\" tabindex=\"0\">
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
              <input id=\"serverPathInput\" placeholder=\"Server path (alternative to upload)\" />
              <input id=\"fileInput\" type=\"file\" />
              <button id=\"loadBtn\">Load Model</button>
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
        <div id=\"modelsPanel\" class=\"card focus-scope\" tabindex=\"0\">
          <h2 class=\"title\">Current Models</h2>
          <div id=\"models\" class=\"models\"></div>
        </div>
        <div id=\"resultsPanel\" class=\"card focus-scope\" tabindex=\"0\">
          <div class=\"panel-row\">
            <h2 class=\"title\">Results</h2>
            <div class=\"results-actions\">
              <button id=\"copyResultsBtn\" class=\"secondary-btn\">Copy</button>
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
    const serverPathInput = document.getElementById("serverPathInput");
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
    const optionsPanel = document.getElementById("optionsPanel");
    const transformsPanel = document.getElementById("transformsPanel");
    const modelsPanel = document.getElementById("modelsPanel");
    const resultsPanel = document.getElementById("resultsPanel");
    const resultsToggleBtn = document.getElementById("resultsToggleBtn");
    const copyResultsBtn = document.getElementById("copyResultsBtn");
    const clearResultsBtn = document.getElementById("clearResultsBtn");
    const resultsPanelBody = document.getElementById("resultsPanelBody");
    const resultsOutput = document.getElementById("resultsOutput");

    let allTransforms = [{ name: "load", enabled: true, kind: "other", allowed_keys: [], required_keys: [], reference_keys: [], to_must_exist: false }];
    let selectedTransform = "load";
    const modelViewState = {};
    const transformConfigByName = {};
    let latestModels = [];
    let latestRuntimeFlags = { dry_run: false, verbose: false };
    const panelScopes = [transformsPanel, optionsPanel, modelsPanel, resultsPanel];

    function setStatus(text) { statusEl.textContent = text; }
    function setFocusedPanel(panel) {
      for (const scope of panelScopes) {
        if (!scope) continue;
        scope.classList.toggle("focused", scope === panel);
      }
    }
    function getFocusableInPanel(panel) {
      if (!panel) return [];
      const all = panel.querySelectorAll("input, select, textarea, button, [tabindex]:not([tabindex='-1'])");
      const out = [];
      for (const el of all) {
        if (el.disabled) continue;
        if (el.offsetParent === null) continue;
        out.push(el);
      }
      return out;
    }
    function runCurrentTransformFromKeyboard() {
      if (selectedTransform === "load") {
        loadBtn.click();
        return;
      }
      if (isRunnableTransform(selectedTransform)) {
        transformRunBtn.click();
      }
    }
    function focusPanelRelative(current, delta) {
      const count = panelScopes.length;
      if (!count) return;
      const idx = panelScopes.indexOf(current);
      const nextIdx = ((idx >= 0 ? idx : 0) + delta + count) % count;
      const target = panelScopes[nextIdx];
      if (!target) return;
      const focusable = getFocusableInPanel(target);
      if (focusable.length) {
        focusable[0].focus();
      } else {
        target.focus();
      }
      setFocusedPanel(target);
    }
    function appendResultBlock(title, text) {
      const block = "[" + title + "]\\n" + ((text && text.trim()) ? text : "(no output)") + "\\n\\n";
      if (resultsOutput.textContent === "(no transform output yet)") {
        resultsOutput.textContent = "";
      }
      resultsOutput.textContent += block;
      resultsOutput.scrollTop = resultsOutput.scrollHeight;
    }
    async function copyTextToClipboard(text) {
      try {
        await navigator.clipboard.writeText(text);
        return true;
      } catch (_err) {
        const ta = document.createElement("textarea");
        ta.value = text;
        document.body.appendChild(ta);
        ta.select();
        const ok = document.execCommand("copy");
        document.body.removeChild(ta);
        return ok;
      }
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
        transformConfigByName[name] = { fields: {}, save_mode: "server", save_download_format: "safetensors" };
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
      const booleanKeys = new Set(Array.isArray(meta.boolean_keys) ? meta.boolean_keys : []);
      const orderedKeys = [...refKeys, ...allowed.filter((k) => !refSet.has(k))];
      transformTitle.textContent = selectedTransform;
      transformFields.innerHTML = "";
      transformRunBtn.textContent = "Run " + selectedTransform;

      if (selectedTransform === "set") {
        const current = document.createElement("div");
        current.className = "binary-summary";
        const line = document.createElement("div");
        line.className = "value";
        line.style.marginBottom = "0";
        line.textContent =
          "dry-run=" + String(Boolean(latestRuntimeFlags.dry_run)) +
          ", verbose=" + String(Boolean(latestRuntimeFlags.verbose));
        current.appendChild(line);
        transformFields.appendChild(current);
      }

      for (const key of orderedKeys) {
        if (selectedTransform === "dump" && key === "format") {
          const fmtSelect = document.createElement("select");
          fmtSelect.innerHTML =
            "<option value='compact'>format: compact</option>" +
            "<option value='tree'>format: tree</option>" +
            "<option value='json'>format: json</option>";
          fmtSelect.value = cfg.fields.format == null ? "compact" : String(cfg.fields.format).toLowerCase();
          if (!["compact", "tree", "json"].includes(fmtSelect.value)) fmtSelect.value = "compact";
          fmtSelect.addEventListener("change", () => { cfg.fields.format = fmtSelect.value; });
          transformFields.appendChild(fmtSelect);
          continue;
        }
        if (selectedTransform === "dump" && key === "verbosity") {
          const verbositySelect = document.createElement("select");
          verbositySelect.innerHTML =
            "<option value='shape'>verbosity: shape</option>" +
            "<option value='stat'>verbosity: stat</option>" +
            "<option value='full'>verbosity: full</option>";
          verbositySelect.value = cfg.fields.verbosity == null ? "shape" : String(cfg.fields.verbosity).toLowerCase();
          if (!["shape", "stat", "full"].includes(verbositySelect.value)) verbositySelect.value = "shape";
          verbositySelect.addEventListener("change", () => { cfg.fields.verbosity = verbositySelect.value; });
          transformFields.appendChild(verbositySelect);
          continue;
        }
        if (booleanKeys.has(key)) {
          const boolLabel = document.createElement("label");
          boolLabel.className = "checkbox-row";
          const boolInput = document.createElement("input");
          boolInput.type = "checkbox";
          const current = cfg.fields[key];
          if (typeof current === "boolean") {
            boolInput.checked = current;
            boolInput.indeterminate = false;
          } else {
            boolInput.checked = false;
            boolInput.indeterminate = true;
          }
          boolInput.addEventListener("change", () => {
            boolInput.indeterminate = false;
            cfg.fields[key] = boolInput.checked;
          });
          const boolText = document.createElement("span");
          const suffix = required.has(key) ? "required" : "optional";
          boolText.textContent = key + " (" + suffix + ")";
          boolLabel.appendChild(boolInput);
          boolLabel.appendChild(boolText);
          transformFields.appendChild(boolLabel);
          continue;
        }
        const input = document.createElement("input");
        const suffix = required.has(key) ? "required" : "optional";
        input.placeholder = key + " (" + suffix + ")";
        input.value = cfg.fields[key] == null ? "" : String(cfg.fields[key]);
        input.addEventListener("input", () => { cfg.fields[key] = input.value; });
        transformFields.appendChild(input);
      }

      if (selectedTransform === "help") {
        const commandSelect = document.createElement("select");
        const helpCommands = Array.isArray(meta.help_commands) ? meta.help_commands : [];
        commandSelect.innerHTML =
          "<option value=''>command: all commands</option>" +
          helpCommands.map((name) => "<option value='" + name + "'>command: " + name + "</option>").join("");
        commandSelect.value = cfg.fields.help_command == null ? "" : String(cfg.fields.help_command);
        if (!["", ...helpCommands].includes(commandSelect.value)) commandSelect.value = "";
        commandSelect.addEventListener("change", () => {
          cfg.fields.help_command = commandSelect.value;
          if (commandSelect.value !== "assert") cfg.fields.help_subcommand = "";
          renderTransformPanel();
        });
        transformFields.appendChild(commandSelect);

        const subcommandSelect = document.createElement("select");
        const subcommandsByCommand =
          (meta.help_subcommands && typeof meta.help_subcommands === "object")
            ? meta.help_subcommands
            : {};
        const selectedCommand = String(cfg.fields.help_command || "");
        const commandSubcommands = selectedCommand
          ? (Array.isArray(subcommandsByCommand[selectedCommand]) ? subcommandsByCommand[selectedCommand] : [])
          : [];
        subcommandSelect.innerHTML =
          "<option value=''>subcommand: none</option>" +
          commandSubcommands.map((name) => "<option value='" + name + "'>subcommand: " + name + "</option>").join("");
        subcommandSelect.value = cfg.fields.help_subcommand == null ? "" : String(cfg.fields.help_subcommand);
        if (!["", ...commandSubcommands].includes(subcommandSelect.value)) subcommandSelect.value = "";
        subcommandSelect.disabled = !selectedCommand || commandSubcommands.length === 0;
        subcommandSelect.addEventListener("change", () => { cfg.fields.help_subcommand = subcommandSelect.value; });
        transformFields.appendChild(subcommandSelect);
      }

      if (selectedTransform === "assert") {
        const yamlInput = document.createElement("textarea");
        yamlInput.rows = 9;
        yamlInput.placeholder =
          "YAML assert expression\\n\\n" +
          "equal:\\n" +
          "  left: model::a.weight\\n" +
          "  right: model::b.weight\\n\\n" +
          "Nested example:\\n" +
          "all:\\n" +
          "  - exists: model::.*weight\\n" +
          "  - not:\\n" +
          "      equal:\\n" +
          "        left: model::a.weight\\n" +
          "        right: model::b.weight";
        yamlInput.value = cfg.fields.assert_yaml == null ? "" : String(cfg.fields.assert_yaml);
        yamlInput.addEventListener("input", () => { cfg.fields.assert_yaml = yamlInput.value; });
        transformFields.appendChild(yamlInput);
      }

      if (selectedTransform === "exit") {
        const copyLabel = document.createElement("label");
        copyLabel.style.display = "flex";
        copyLabel.style.alignItems = "center";
        copyLabel.style.gap = "8px";
        copyLabel.style.marginBottom = "8px";
        const copyCheckbox = document.createElement("input");
        copyCheckbox.type = "checkbox";
        copyCheckbox.style.width = "auto";
        copyCheckbox.style.margin = "0";
        copyCheckbox.checked = Boolean(cfg.fields.exit_auto_copy);
        copyCheckbox.addEventListener("change", () => { cfg.fields.exit_auto_copy = copyCheckbox.checked; });
        const copyText = document.createElement("span");
        copyText.textContent = "Copy plan to clipboard";
        copyLabel.appendChild(copyCheckbox);
        copyLabel.appendChild(copyText);
        transformFields.appendChild(copyLabel);
      }

      if (selectedTransform === "save") {
        const modeSelect = document.createElement("select");
        modeSelect.innerHTML = "<option value='server'>save on server path</option><option value='download'>download to browser</option>";
        modeSelect.value = cfg.save_mode || "server";
        modeSelect.addEventListener("change", () => {
          cfg.save_mode = modeSelect.value;
          renderTransformPanel();
        });
        transformFields.appendChild(modeSelect);

        if ((cfg.save_mode || "server") === "download") {
          const fmtSelect = document.createElement("select");
          fmtSelect.innerHTML = "<option value='safetensors'>download format: safetensors</option><option value='numpy'>download format: numpy</option><option value='torch'>download format: pytorch</option>";
          fmtSelect.value = cfg.save_download_format || "safetensors";
          fmtSelect.addEventListener("change", () => {
            cfg.save_download_format = fmtSelect.value;
          });
          transformFields.appendChild(fmtSelect);
        }
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
        row.tabIndex = item.enabled ? 0 : -1;
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
          row.addEventListener("keydown", (event) => {
            if (event.key !== "Enter" && event.key !== " ") return;
            event.preventDefault();
            row.click();
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
          modelViewState[model.alias] = {
            format: "compact",
            verbosity: "shape",
            filter: "",
            valid: true,
            dump_collapsed: false
          };
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
        const dumpToggleBtn = document.createElement("button");
        dumpToggleBtn.className = "secondary-btn toggle-dump-btn";
        dumpToggleBtn.textContent = state.dump_collapsed ? "Show Dump" : "Hide Dump";
        right.appendChild(count);
        right.appendChild(dumpToggleBtn);
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
        pre.classList.toggle("hidden", !!state.dump_collapsed);

        dumpToggleBtn.addEventListener("click", () => {
          state.dump_collapsed = !state.dump_collapsed;
          pre.classList.toggle("hidden", !!state.dump_collapsed);
          dumpToggleBtn.textContent = state.dump_collapsed ? "Show Dump" : "Hide Dump";
        });

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
        if (stateData.ok) {
          latestModels = stateData.models || [];
          if (stateData.runtime_flags && typeof stateData.runtime_flags === "object") {
            latestRuntimeFlags = {
              dry_run: Boolean(stateData.runtime_flags.dry_run),
              verbose: Boolean(stateData.runtime_flags.verbose),
            };
          }
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
      if (selectedTransform !== "load") { setStatus("Select load first."); return; }
      const serverPath = (serverPathInput.value || "").trim();
      const file = fileInput.files && fileInput.files[0];
      if (!serverPath && !file) { setStatus("Provide a server path or pick a model file first."); return; }

      loadBtn.disabled = true;
      try {
        const payload = { alias: aliasInput.value || null };
        if (serverPath) {
          payload.server_path = serverPath;
        } else {
          setStatus("Reading file...");
          const bytes = new Uint8Array(await file.arrayBuffer());
          let binary = "";
          const chunk = 0x8000;
          for (let i = 0; i < bytes.length; i += chunk) binary += String.fromCharCode(...bytes.subarray(i, i + chunk));
          payload.filename = file.name;
          payload.content_b64 = btoa(binary);
        }

        setStatus("Loading model via transform...");
        const response = await fetch("/api/load", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload)
        });
        const data = await response.json();
        if (!response.ok || !data.ok) { setStatus("Load failed: " + (data.error || "unknown error")); return; }
        latestModels = data.models || [];
        if (data.runtime_flags && typeof data.runtime_flags === "object") {
          latestRuntimeFlags = {
            dry_run: Boolean(data.runtime_flags.dry_run),
            verbose: Boolean(data.runtime_flags.verbose),
          };
        }
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
      let payload = {};

      for (const key of allowed) {
        const rawValue = cfg.fields[key];
        if (typeof rawValue === "boolean") {
          payload[key] = rawValue;
          continue;
        }
        const parsed = parseFieldValue(rawValue == null ? "" : String(rawValue));
        if (parsed === undefined) {
          if (required.has(key)) { setStatus("Missing required parameter: " + key); return; }
          continue;
        }
        payload[key] = parsed;
      }

      if (runTransformName === "help") {
        const command = String(cfg.fields.help_command || "").trim();
        const subcommand = String(cfg.fields.help_subcommand || "").trim();
        if (!command && subcommand) {
          setStatus("Set command before subcommand for help.");
          return;
        }
        if (!command) {
          payload = {};
        } else if (!subcommand) {
          payload = command;
        } else {
          payload = { [command]: subcommand };
        }
      }

      if (runTransformName === "assert") {
        const yamlExpr = String(cfg.fields.assert_yaml || "").trim();
        if (!yamlExpr) {
          setStatus("Provide an assert YAML expression.");
          return;
        }
        payload = yamlExpr;
      }

      if (runTransformName === "save" && (cfg.save_mode || "server") === "download") {
        payload.format = cfg.save_download_format || "safetensors";
      }

      setStatus("Applying " + runTransformName + "...");
      transformRunBtn.disabled = true;
      try {
        const isSaveDownload = runTransformName === "save" && (cfg.save_mode || "server") === "download";
        const response = await fetch(
          isSaveDownload ? "/api/save_download" : "/api/apply_transform",
          {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(
              isSaveDownload
                ? { payload: payload }
                : { transform: runTransformName, payload: payload }
            )
          }
        );
        const data = await response.json();
        if (!response.ok || !data.ok) { setStatus("Apply failed: " + (data.error || "unknown error")); return; }
        latestModels = data.models || [];
        if (data.runtime_flags && typeof data.runtime_flags === "object") {
          latestRuntimeFlags = {
            dry_run: Boolean(data.runtime_flags.dry_run),
            verbose: Boolean(data.runtime_flags.verbose),
          };
        }
        renderModels(latestModels);
        resetTransformSearch();
        renderTransforms();
        updatePanels();
        if (runTransformName === "exit") {
          const exitText = (data.output && data.output.trim()) ? data.output : "(no output)";
          appendResultBlock(runTransformName, exitText);
          if (cfg.fields.exit_auto_copy && exitText !== "(no output)") {
            const copied = await copyTextToClipboard(exitText);
            setStatus(
              copied
                ? "Applied exit successfully. Copied plan to clipboard."
                : "Applied exit successfully. Could not copy automatically."
            );
          } else {
            setStatus("Applied exit successfully.");
          }
        } else {
          setStatus("Applied " + runTransformName + " successfully.");
          appendResultBlock(runTransformName, data.output || "");
        }
        if (isSaveDownload) {
          const raw = atob(data.download_b64 || "");
          const bytes = new Uint8Array(raw.length);
          for (let i = 0; i < raw.length; i += 1) bytes[i] = raw.charCodeAt(i);
          const blob = new Blob([bytes], { type: data.download_mime || "application/octet-stream" });
          const url = URL.createObjectURL(blob);
          const a = document.createElement("a");
          a.href = url;
          a.download = data.download_filename || "brainsurgery-save.bin";
          a.click();
          URL.revokeObjectURL(url);
        }
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
    copyResultsBtn.addEventListener("click", async () => {
      const text = resultsOutput.textContent || "";
      if (!text.trim() || text === "(no transform output yet)") {
        setStatus("Nothing to copy.");
        return;
      }
      const copied = await copyTextToClipboard(text);
      setStatus(copied ? "Copied results to clipboard." : "Could not copy results.");
    });
    for (const panel of panelScopes) {
      if (!panel) continue;
      panel.addEventListener("focusin", () => setFocusedPanel(panel));
      panel.addEventListener("mousedown", () => setFocusedPanel(panel));
      panel.addEventListener("keydown", (event) => {
        if (panel === transformsPanel && (event.key === "ArrowDown" || event.key === "ArrowUp")) {
          const rows = Array.from(
            transformsEl.querySelectorAll(".transform-item:not(.planned)")
          ).filter((row) => row instanceof HTMLElement);
          if (rows.length) {
            event.preventDefault();
            const currentIndex = rows.indexOf(document.activeElement);
            const delta = event.key === "ArrowDown" ? 1 : -1;
            const nextIndex = (currentIndex + delta + rows.length) % rows.length;
            rows[nextIndex].focus();
            rows[nextIndex].click();
          }
          return;
        }
        if (event.key === "Tab") {
          if (event.shiftKey) {
            event.preventDefault();
            focusPanelRelative(panel, -1);
            return;
          }
          const focusable = getFocusableInPanel(panel);
          if (!focusable.length) return;
          const idx = focusable.indexOf(document.activeElement);
          if (idx === -1 || idx >= focusable.length - 1) {
            event.preventDefault();
            focusable[0].focus();
          }
          return;
        }
        if (panel === optionsPanel && event.key === "Enter" && event.shiftKey) {
          const target = event.target;
          if (!target) return;
          event.preventDefault();
          runCurrentTransformFromKeyboard();
        }
      });
    }
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
        serverPathInput.value = "";
        fileInput.value = "";
        setStatus("Cleared load options.");
        return;
      }
      if (isRunnableTransform(selectedTransform)) {
        const cfg = getTransformConfig(selectedTransform);
        cfg.fields = {};
        cfg.save_mode = "server";
        cfg.save_download_format = "safetensors";
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
