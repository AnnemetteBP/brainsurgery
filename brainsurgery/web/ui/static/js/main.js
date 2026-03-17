import {
  aliasInput,
  assertErrorCloseBtn,
  assertErrorDetails,
  assertErrorMessage,
  assertErrorModal,
  clearOptionsBtn,
  clearResultsBtn,
  copyResultsBtn,
  diffFilterInput,
  diffLeftAlias,
  diffRefreshBtn,
  diffRightAlias,
  diffSummary,
  diffTopRows,
  fileInput,
  insightsPanel,
  iteratingProgressEl,
  loadBtn,
  loadPanel,
  modelsEl,
  modelsPanel,
  optionsEmpty,
  optionsPanel,
  optionsPanelBody,
  optionsToggleBtn,
  panelScopes,
  previewConfirmDetails,
  previewConfirmMessage,
  previewConfirmModal,
  previewGoBtn,
  previewNoGoBtn,
  previewImpactRows,
  resultsOutput,
  resultsPanel,
  resultsPanelBody,
  resultsToggleBtn,
  serverPathInput,
  statusEl,
  transformFields,
  transformPanel,
  transformRunBtn,
  transformSearchEl,
  transformsEl,
  transformsPanel,
  transformTitle,
} from "./dom.js";
import { createActions } from "./actions.js";
import { bindPanelInteractions } from "./keyboard.js";
import { createModelsRenderer } from "./models.js";
import { createIteratingProgressController } from "./progress.js";
import { appState } from "./state.js";
import { createTransformsUI } from "./transforms_ui.js";
import {
  buildReferenceFromModel,
  copyFromFilterToToTemplate,
  copyTextToClipboard,
  parseFieldValue,
  tensorCountText,
} from "./utils.js";

const progressController = createIteratingProgressController({ iteratingProgressEl });

function setStatus(text) {
  statusEl.textContent = text;
}

function hideAssertFailureModal() {
  assertErrorModal.classList.add("hidden");
}

function showAssertFailureModal(data) {
  const info = (data && typeof data === "object" && data.error_info && typeof data.error_info === "object")
    ? data.error_info
    : {};
  const context = (info.context && typeof info.context === "object") ? info.context : {};
  const message = typeof info.message === "string" && info.message.trim()
    ? info.message.trim()
    : (typeof data.error === "string" ? data.error : "Assertion failed.");

  const detailLines = [];
  if (typeof context.expression === "string" && context.expression) {
    detailLines.push("expression: " + context.expression);
  }
  if (Array.isArray(context.expression_keys) && context.expression_keys.length) {
    detailLines.push("keys: " + context.expression_keys.join(", "));
  }
  if (typeof info.endpoint === "string" && info.endpoint) {
    detailLines.push("endpoint: " + info.endpoint);
  }
  if (typeof context.raw_payload === "string" && context.raw_payload) {
    detailLines.push("");
    detailLines.push("payload:");
    detailLines.push(context.raw_payload);
  }

  assertErrorMessage.textContent = message;
  assertErrorDetails.textContent = detailLines.join("\n");
  assertErrorModal.classList.remove("hidden");
}

let previewOnGo = null;
let previewOnNoGo = null;

function hidePreviewConfirmModal() {
  previewConfirmModal.classList.add("hidden");
  previewOnGo = null;
  previewOnNoGo = null;
}

function showPreviewConfirmModal({ transformName, previewOutput, onGo, onNoGo }) {
  previewConfirmMessage.textContent = "Preview indicates tensor-impacting changes. Proceed?";
  previewConfirmDetails.textContent = (previewOutput && String(previewOutput).trim())
    ? String(previewOutput)
    : ("preview 1/1 " + transformName + ": no preview output");
  previewOnGo = typeof onGo === "function" ? onGo : null;
  previewOnNoGo = typeof onNoGo === "function" ? onNoGo : null;
  previewConfirmModal.classList.remove("hidden");
}

function setFocusedPanel(panel) {
  for (const scope of panelScopes) {
    if (!scope) continue;
    scope.classList.toggle("focused", scope === panel);
  }
}

function runCurrentTransformFromKeyboard(isRunnableTransform) {
  if (appState.selectedTransform === "load") {
    loadBtn.click();
    return;
  }
  if (isRunnableTransform(appState.selectedTransform)) {
    transformRunBtn.click();
  }
}

function appendResultBlock(title, text) {
  const block = "[" + title + "]\n" + ((text && text.trim()) ? text : "(no output)") + "\n\n";
  if (resultsOutput.textContent === "(no transform output yet)") {
    resultsOutput.textContent = "";
  }
  resultsOutput.textContent += block;
  resultsOutput.scrollTop = resultsOutput.scrollHeight;
}

function _syncDiffAliasSelects(models) {
  if (!diffLeftAlias || !diffRightAlias) return;
  const aliases = Array.isArray(models) ? models.map((item) => String(item.alias || "")).filter(Boolean) : [];
  const leftCurrent = diffLeftAlias.value;
  const rightCurrent = diffRightAlias.value;
  diffLeftAlias.innerHTML = "";
  diffRightAlias.innerHTML = "";
  for (const alias of aliases) {
    const leftOpt = document.createElement("option");
    leftOpt.value = alias;
    leftOpt.textContent = "left: " + alias;
    diffLeftAlias.appendChild(leftOpt);
    const rightOpt = document.createElement("option");
    rightOpt.value = alias;
    rightOpt.textContent = "right: " + alias;
    diffRightAlias.appendChild(rightOpt);
  }
  if (aliases.includes(leftCurrent)) diffLeftAlias.value = leftCurrent;
  if (aliases.includes(rightCurrent)) diffRightAlias.value = rightCurrent;
  if (!diffLeftAlias.value && aliases[0]) diffLeftAlias.value = aliases[0];
  if (!diffRightAlias.value && aliases[1]) diffRightAlias.value = aliases[1];
  if (!diffRightAlias.value && aliases[0]) diffRightAlias.value = aliases[0];
}

function _renderPreviewImpactHistory() {
  if (!previewImpactRows) return;
  previewImpactRows.innerHTML = "";
  const insights = appState.latestInsights || {};
  const history = Array.isArray(insights.preview_history) ? insights.preview_history : [];
  if (!history.length) {
    const empty = document.createElement("div");
    empty.className = "empty";
    empty.textContent = "No preview activity yet.";
    previewImpactRows.appendChild(empty);
    return;
  }
  const recent = history.slice(-8).reverse();
  for (const item of recent) {
    const changed = Number(item.changed_count || 0);
    const created = Number(item.created_count || 0);
    const deleted = Number(item.deleted_count || 0);
    const total = Math.max(1, changed + created + deleted);
    const row = document.createElement("div");
    row.className = "insight-row";
    const label = document.createElement("div");
    label.className = "insight-row-label";
    label.textContent = String(item.transform || "transform") + " · c:" + changed + " n:" + created + " d:" + deleted;
    const meter = document.createElement("div");
    meter.className = "stack-meter";
    const changedBar = document.createElement("span");
    changedBar.className = "stack-fill changed";
    changedBar.style.width = Math.round((changed / total) * 100) + "%";
    const createdBar = document.createElement("span");
    createdBar.className = "stack-fill created";
    createdBar.style.width = Math.round((created / total) * 100) + "%";
    const deletedBar = document.createElement("span");
    deletedBar.className = "stack-fill deleted";
    deletedBar.style.width = Math.round((deleted / total) * 100) + "%";
    meter.appendChild(changedBar);
    meter.appendChild(createdBar);
    meter.appendChild(deletedBar);
    row.appendChild(label);
    row.appendChild(meter);
    previewImpactRows.appendChild(row);
  }
}

function _renderDiffPayload(diff) {
  if (!diffSummary || !diffTopRows) return;
  diffTopRows.innerHTML = "";
  if (!diff || typeof diff !== "object") {
    diffSummary.textContent = "No diff loaded.";
    return;
  }
  const summary = (diff.summary && typeof diff.summary === "object") ? diff.summary : {};
  diffSummary.textContent =
    "shared=" + Number(summary.shared_count || 0) +
    " changed=" + Number(summary.changed || 0) +
    " unchanged=" + Number(summary.unchanged || 0) +
    " missing_left=" + Number(summary.missing_on_left || 0) +
    " missing_right=" + Number(summary.missing_on_right || 0);
  const top = Array.isArray(diff.top_differences) ? diff.top_differences : [];
  if (!top.length) {
    const empty = document.createElement("div");
    empty.className = "empty";
    empty.textContent = "No differing tensors in current selection.";
    diffTopRows.appendChild(empty);
    return;
  }
  const maxDiff = top.reduce((acc, item) => {
    const value = Number(item.max_abs_diff || 0);
    return Math.max(acc, Number.isFinite(value) ? value : 0);
  }, 0);
  for (const item of top.slice(0, 12)) {
    const row = document.createElement("div");
    row.className = "insight-row";
    const label = document.createElement("div");
    label.className = "insight-row-label";
    const diffValue = item.max_abs_diff == null ? "n/a" : Number(item.max_abs_diff).toPrecision(4);
    label.textContent = String(item.name || "(unknown)") + " · " + String(item.kind || "values") + " · " + diffValue;
    row.appendChild(label);
    if (item.max_abs_diff != null) {
      const meter = document.createElement("div");
      meter.className = "meter";
      const fill = document.createElement("div");
      fill.className = "meter-fill";
      const width = maxDiff > 0 ? Math.max(2, Math.round((Number(item.max_abs_diff) / maxDiff) * 100)) : 2;
      fill.style.width = width + "%";
      meter.appendChild(fill);
      row.appendChild(meter);
    }
    diffTopRows.appendChild(row);
  }
}

async function refreshModelDiff() {
  if (!diffLeftAlias || !diffRightAlias || !diffSummary) return;
  const leftAlias = String(diffLeftAlias.value || "").trim();
  const rightAlias = String(diffRightAlias.value || "").trim();
  if (!leftAlias || !rightAlias) {
    diffSummary.textContent = "Load at least two model aliases to diff.";
    if (diffTopRows) diffTopRows.innerHTML = "";
    return;
  }
  diffSummary.textContent = "Loading diff...";
  if (diffTopRows) diffTopRows.innerHTML = "";
  try {
    const filter = String((diffFilterInput && diffFilterInput.value) || "").trim();
    const response = await fetch("/api/model_diff", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        left_alias: leftAlias,
        right_alias: rightAlias,
        left_filter: filter || ".*",
        right_filter: filter || ".*",
      }),
    });
    const data = await response.json();
    if (!response.ok || !data.ok) {
      diffSummary.textContent = "Diff failed: " + String(data.error || "unknown error");
      return;
    }
    _renderDiffPayload(data.diff);
  } catch (err) {
    diffSummary.textContent = "Diff failed: " + String(err);
  }
}

function renderInsights() {
  _syncDiffAliasSelects(appState.latestModels);
  _renderPreviewImpactHistory();
  if (Array.isArray(appState.latestModels) && appState.latestModels.length >= 2) {
    refreshModelDiff();
  } else if (diffSummary) {
    diffSummary.textContent = "Load at least two model aliases to diff.";
    if (diffTopRows) diffTopRows.innerHTML = "";
  }
}

const transformsUI = createTransformsUI({
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
  stopProgress: () => progressController.stop(),
  buildReferenceFromModel,
  copyFromFilterToToTemplate,
});

const { renderModels } = createModelsRenderer({
  modelsEl,
  modelViewState: appState.modelViewState,
  getSelectedTransform: () => appState.selectedTransform,
  getTransformMeta: transformsUI.getTransformMeta,
  isRunnableTransform: transformsUI.isRunnableTransform,
  commitRefFromModel: transformsUI.commitRefFromModel,
  setStatus,
  tensorCountText,
});

transformsUI.setOnSelectionChanged(() => {
  renderModels(appState.latestModels);
});

const actions = createActions({
  appState,
  setStatus,
  progressController,
  getIsIteratingTransform: transformsUI.getIsIteratingTransform,
  getTransformMeta: transformsUI.getTransformMeta,
  getTransformConfig: transformsUI.getTransformConfig,
  isRunnableTransform: transformsUI.isRunnableTransform,
  resetTransformSearch: transformsUI.resetTransformSearch,
  renderTransforms: transformsUI.renderTransforms,
  updatePanels: transformsUI.updatePanels,
  renderModels,
  renderInsights,
  appendResultBlock,
  copyTextToClipboard,
  parseFieldValue,
  showAssertFailure: showAssertFailureModal,
  showPreviewConfirm: showPreviewConfirmModal,
  loadBtn,
  aliasInput,
  serverPathInput,
  fileInput,
  transformRunBtn,
});

actions.bindLoad();
actions.bindRun();

if (diffRefreshBtn) {
  diffRefreshBtn.addEventListener("click", () => {
    refreshModelDiff();
  });
}
if (diffLeftAlias) diffLeftAlias.addEventListener("change", () => refreshModelDiff());
if (diffRightAlias) diffRightAlias.addEventListener("change", () => refreshModelDiff());
if (diffFilterInput) {
  let diffDebounce = null;
  diffFilterInput.addEventListener("input", () => {
    clearTimeout(diffDebounce);
    diffDebounce = setTimeout(() => refreshModelDiff(), 260);
  });
}

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

bindPanelInteractions({
  panelScopes,
  transformsPanel,
  optionsPanel,
  transformsEl,
  setFocusedPanel,
  runCurrentTransformFromKeyboard: () => runCurrentTransformFromKeyboard(transformsUI.isRunnableTransform),
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
  if (appState.selectedTransform === "load") {
    aliasInput.value = "";
    serverPathInput.value = "";
    fileInput.value = "";
    setStatus("Cleared load options.");
    return;
  }
  if (transformsUI.isRunnableTransform(appState.selectedTransform)) {
    const cfg = transformsUI.getTransformConfig(appState.selectedTransform);
    cfg.fields = {};
    cfg.save_mode = "server";
    cfg.save_download_format = "safetensors";
    transformsUI.renderTransformPanel();
    setStatus("Cleared " + appState.selectedTransform + " options.");
    return;
  }
  setStatus("Nothing to clear.");
});

transformSearchEl.addEventListener("input", () => transformsUI.renderTransforms());

transformSearchEl.value = "";
transformsUI.renderTransforms();
transformsUI.updatePanels();
actions.refresh().catch((err) => setStatus("Initial load failed: " + String(err)));
if (insightsPanel) {
  renderInsights();
}

assertErrorCloseBtn.addEventListener("click", hideAssertFailureModal);
assertErrorModal.addEventListener("click", (event) => {
  if (event.target === assertErrorModal) {
    hideAssertFailureModal();
  }
});
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && !assertErrorModal.classList.contains("hidden")) {
    hideAssertFailureModal();
    return;
  }
  if (event.key === "Escape" && !previewConfirmModal.classList.contains("hidden")) {
    if (typeof previewOnNoGo === "function") {
      const noGo = previewOnNoGo;
      hidePreviewConfirmModal();
      noGo();
      return;
    }
    hidePreviewConfirmModal();
  }
});

previewGoBtn.addEventListener("click", () => {
  if (typeof previewOnGo !== "function") return;
  const go = previewOnGo;
  hidePreviewConfirmModal();
  go();
});

previewNoGoBtn.addEventListener("click", () => {
  if (typeof previewOnNoGo !== "function") {
    hidePreviewConfirmModal();
    return;
  }
  const noGo = previewOnNoGo;
  hidePreviewConfirmModal();
  noGo();
});
