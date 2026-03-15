function createModelsRenderer({
  modelsEl,
  modelViewState,
  getSelectedTransform,
  getTransformMeta,
  isRunnableTransform,
  commitRefFromModel,
  setStatus,
  tensorCountText,
}) {
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
          dump_collapsed: false,
          lastDumpText: "",
          lastDumpSignature: "",
          lastMatchedCount: null,
          lastTotalCount: null,
          sourceTensorCount: null,
        };
      }
      const state = modelViewState[model.alias];
      if (state.sourceTensorCount !== model.tensor_count) {
        state.sourceTensorCount = model.tensor_count;
        state.lastDumpText = "";
        state.lastDumpSignature = "";
        state.lastMatchedCount = null;
        state.lastTotalCount = null;
      }
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
      const currentSignature = state.format + "|" + String(state.verbosity || "shape") + "|" + state.filter;
      const hasCachedForCurrentView = state.lastDumpSignature === currentSignature && !!state.lastDumpText;
      const count = document.createElement("span");
      if (
        hasCachedForCurrentView &&
        typeof state.lastMatchedCount === "number" &&
        typeof state.lastTotalCount === "number"
      ) {
        count.textContent = tensorCountText(state.lastMatchedCount, state.lastTotalCount, state.filter || "");
      } else {
        count.textContent = tensorCountText(model.matched_count || model.tensor_count, model.total_count || model.tensor_count, state.filter || "");
      }
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
        const selectedTransform = getSelectedTransform();
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
      pre.textContent = hasCachedForCurrentView ? state.lastDumpText : "";
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
              filter: filterInput.value,
            }),
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
          state.lastDumpText = pre.textContent;
          state.lastDumpSignature = state.format + "|" + String(state.verbosity || "shape") + "|" + state.filter;
          state.lastMatchedCount = data.matched_count || 0;
          state.lastTotalCount = data.total_count || 0;
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

      if (!pre.textContent) {
        pre.textContent = "(loading dump...)";
        requestDump();
      }

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

  return { renderModels };
}

export { createModelsRenderer };
