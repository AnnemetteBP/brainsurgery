import {
  aliasInput,
  clearOptionsBtn,
  clearResultsBtn,
  copyResultsBtn,
  fileInput,
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
  appendResultBlock,
  copyTextToClipboard,
  parseFieldValue,
  loadBtn,
  aliasInput,
  serverPathInput,
  fileInput,
  transformRunBtn,
});

actions.bindLoad();
actions.bindRun();

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
