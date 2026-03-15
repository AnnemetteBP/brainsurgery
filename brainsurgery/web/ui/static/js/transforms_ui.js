function createTransformsUI({
  appState,
  transformsEl,
  transformSearchEl,
  loadPanel,
  optionsEmpty,
  transformPanel,
  transformTitle,
  transformFields,
  transformRunBtn,
  setStatus,
  stopProgress,
  buildReferenceFromModel,
  copyFromFilterToToTemplate,
}) {
  let onSelectionChanged = () => {};

  function setOnSelectionChanged(callback) {
    onSelectionChanged = callback;
  }

  function getTransformMeta(name) {
    return appState.allTransforms.find((t) => t.name === name) || null;
  }

  function isReadyTransform(name) {
    const meta = getTransformMeta(name);
    return !!(meta && meta.enabled);
  }

  function isRunnableTransform(name) {
    return !!name && name !== "load" && isReadyTransform(name);
  }

  function getIsIteratingTransform(name) {
    const meta = getTransformMeta(name);
    return !!(meta && meta.iterating);
  }

  function getTransformConfig(name) {
    if (!appState.transformConfigByName[name]) {
      appState.transformConfigByName[name] = { fields: {}, save_mode: "server", save_download_format: "safetensors" };
    }
    return appState.transformConfigByName[name];
  }

  function resetTransformSearch() {
    transformSearchEl.value = "";
    renderTransforms();
  }

  function commitRefFromModel(key, alias, filterText) {
    const meta = getTransformMeta(appState.selectedTransform);
    if (!meta) return;
    const cfg = getTransformConfig(appState.selectedTransform);
    cfg.fields[key] = buildReferenceFromModel(alias, filterText);
    if (meta.kind === "binary" && key === "from" && !meta.to_must_exist) {
      const templ = copyFromFilterToToTemplate((filterText || "").trim());
      cfg.fields.to = alias + "::" + (templ || ".*");
    }
    renderTransformPanel();
    setStatus("Committed " + key + " for " + appState.selectedTransform + " from " + alias + ".");
  }

  function renderTransformPanel() {
    if (!isRunnableTransform(appState.selectedTransform)) {
      transformPanel.classList.add("hidden");
      return;
    }
    const meta = getTransformMeta(appState.selectedTransform);
    const cfg = getTransformConfig(appState.selectedTransform);
    const allowed = Array.isArray(meta.allowed_keys) ? meta.allowed_keys : [];
    const required = new Set(Array.isArray(meta.required_keys) ? meta.required_keys : []);
    const refKeys = Array.isArray(meta.reference_keys) ? meta.reference_keys : [];
    const refSet = new Set(refKeys);
    const booleanKeys = new Set(Array.isArray(meta.boolean_keys) ? meta.boolean_keys : []);
    const orderedKeys = [...refKeys, ...allowed.filter((k) => !refSet.has(k))];
    transformTitle.textContent = appState.selectedTransform;
    transformFields.innerHTML = "";
    transformRunBtn.textContent = "Run " + appState.selectedTransform;

    if (appState.selectedTransform === "prefixes") {
      const modeSelect = document.createElement("select");
      modeSelect.innerHTML =
        "<option value='list'>mode: list aliases</option>" +
        "<option value='add'>mode: add alias</option>" +
        "<option value='remove'>mode: remove alias</option>" +
        "<option value='rename'>mode: rename alias</option>";
      const rawMode = String(cfg.fields.mode == null ? "list" : cfg.fields.mode).toLowerCase();
      modeSelect.value = ["list", "add", "remove", "rename"].includes(rawMode) ? rawMode : "list";
      cfg.fields.mode = modeSelect.value;
      modeSelect.addEventListener("change", () => {
        cfg.fields.mode = modeSelect.value;
        if (modeSelect.value === "list") {
          delete cfg.fields.alias;
          delete cfg.fields.from;
          delete cfg.fields.to;
        } else if (modeSelect.value === "add" || modeSelect.value === "remove") {
          delete cfg.fields.from;
          delete cfg.fields.to;
        } else if (modeSelect.value === "rename") {
          delete cfg.fields.alias;
        }
        renderTransformPanel();
      });
      transformFields.appendChild(modeSelect);

      if (modeSelect.value === "add" || modeSelect.value === "remove") {
        const aliasInput = document.createElement("input");
        aliasInput.placeholder = "alias (required)";
        aliasInput.value = cfg.fields.alias == null ? "" : String(cfg.fields.alias);
        aliasInput.addEventListener("input", () => { cfg.fields.alias = aliasInput.value; });
        transformFields.appendChild(aliasInput);
      } else if (modeSelect.value === "rename") {
        const fromInput = document.createElement("input");
        fromInput.placeholder = "from (required)";
        fromInput.value = cfg.fields.from == null ? "" : String(cfg.fields.from);
        fromInput.addEventListener("input", () => { cfg.fields.from = fromInput.value; });
        transformFields.appendChild(fromInput);

        const toInput = document.createElement("input");
        toInput.placeholder = "to (required)";
        toInput.value = cfg.fields.to == null ? "" : String(cfg.fields.to);
        toInput.addEventListener("input", () => { cfg.fields.to = toInput.value; });
        transformFields.appendChild(toInput);
      }

      transformPanel.classList.remove("hidden");
      return;
    }

    if (appState.selectedTransform === "set") {
      const current = document.createElement("div");
      current.className = "binary-summary";
      const line = document.createElement("div");
      line.className = "value";
      line.style.marginBottom = "0";
      line.textContent =
        "dry-run=" + String(Boolean(appState.latestRuntimeFlags.dry_run)) +
        ", verbose=" + String(Boolean(appState.latestRuntimeFlags.verbose));
      current.appendChild(line);
      transformFields.appendChild(current);
    }

    for (const key of orderedKeys) {
      if (appState.selectedTransform === "dump" && key === "format") {
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
      if (appState.selectedTransform === "dump" && key === "verbosity") {
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

    if (appState.selectedTransform === "help") {
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

    if (appState.selectedTransform === "assert") {
      const yamlInput = document.createElement("textarea");
      yamlInput.rows = 9;
      yamlInput.placeholder =
        "YAML assert expression\n\n" +
        "equal:\n" +
        "  left: model::a.weight\n" +
        "  right: model::b.weight\n\n" +
        "Nested example:\n" +
        "all:\n" +
        "  - exists: model::.*weight\n" +
        "  - not:\n" +
        "      equal:\n" +
        "        left: model::a.weight\n" +
        "        right: model::b.weight";
      yamlInput.value = cfg.fields.assert_yaml == null ? "" : String(cfg.fields.assert_yaml);
      yamlInput.addEventListener("input", () => { cfg.fields.assert_yaml = yamlInput.value; });
      transformFields.appendChild(yamlInput);
    }

    if (appState.selectedTransform === "exit") {
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

      const summaryModeSelect = document.createElement("select");
      summaryModeSelect.innerHTML =
        "<option value='raw'>summary mode: raw</option>" +
        "<option value='resolve'>summary mode: resolve</option>";
      summaryModeSelect.value = cfg.fields.exit_summary_mode == null ? "raw" : String(cfg.fields.exit_summary_mode).toLowerCase();
      if (!["raw", "resolve"].includes(summaryModeSelect.value)) summaryModeSelect.value = "raw";
      summaryModeSelect.addEventListener("change", () => { cfg.fields.exit_summary_mode = summaryModeSelect.value; });
      transformFields.appendChild(summaryModeSelect);
    }

    if (appState.selectedTransform === "save") {
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
        setStatus("Copied from filter into to for " + appState.selectedTransform + ".");
      });
      transformFields.appendChild(copyBtn);
    }

    transformPanel.classList.remove("hidden");
  }

  function updatePanels() {
    const showLoad = appState.selectedTransform === "load";
    const showTransform = isRunnableTransform(appState.selectedTransform);
    const hasSelection = !!appState.selectedTransform;
    loadPanel.classList.toggle("hidden", !showLoad);
    transformPanel.classList.toggle("hidden", !showTransform);
    optionsEmpty.classList.toggle("hidden", hasSelection);
    renderTransformPanel();
    if (appState.selectedTransform === "load") {
      stopProgress();
      setStatus("Load is selected. Pick a file to import a model.");
    } else if (!isReadyTransform(appState.selectedTransform)) {
      stopProgress();
      setStatus("Selected " + appState.selectedTransform + " is planned and not interactive yet.");
    } else if (!hasSelection) {
      stopProgress();
      setStatus("Ready.");
    } else {
      setStatus("Selected " + appState.selectedTransform + ".");
    }
  }

  function renderTransforms() {
    const query = transformSearchEl.value.trim().toLowerCase();
    const items = appState.allTransforms.filter((item) => item.name.toLowerCase().includes(query));
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
      row.className = "transform-item" + (item.enabled ? "" : " planned") + (appState.selectedTransform === item.name ? " selected" : "");
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
          appState.selectedTransform = item.name;
          renderTransforms();
          updatePanels();
          onSelectionChanged();
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

  return {
    commitRefFromModel,
    getIsIteratingTransform,
    getTransformConfig,
    getTransformMeta,
    isRunnableTransform,
    renderTransforms,
    renderTransformPanel,
    resetTransformSearch,
    setOnSelectionChanged,
    updatePanels,
  };
}

export { createTransformsUI };
