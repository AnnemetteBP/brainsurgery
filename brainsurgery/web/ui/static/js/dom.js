const transformsEl = document.getElementById("transformList");
const transformSearchEl = document.getElementById("transformSearch");
const modelsEl = document.getElementById("models");
const statusEl = document.getElementById("status");
const iteratingProgressEl = document.getElementById("iteratingProgress");
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

const panelScopes = [transformsPanel, optionsPanel, modelsPanel, resultsPanel];

export {
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
};
