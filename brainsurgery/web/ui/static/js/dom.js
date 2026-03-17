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
const insightsPanel = document.getElementById("insightsPanel");
const previewImpactRows = document.getElementById("previewImpactRows");
const diffLeftAlias = document.getElementById("diffLeftAlias");
const diffRightAlias = document.getElementById("diffRightAlias");
const diffFilterInput = document.getElementById("diffFilterInput");
const diffRefreshBtn = document.getElementById("diffRefreshBtn");
const diffSummary = document.getElementById("diffSummary");
const diffTopRows = document.getElementById("diffTopRows");
const resultsPanel = document.getElementById("resultsPanel");
const resultsToggleBtn = document.getElementById("resultsToggleBtn");
const copyResultsBtn = document.getElementById("copyResultsBtn");
const clearResultsBtn = document.getElementById("clearResultsBtn");
const resultsPanelBody = document.getElementById("resultsPanelBody");
const resultsOutput = document.getElementById("resultsOutput");
const assertErrorModal = document.getElementById("assertErrorModal");
const assertErrorMessage = document.getElementById("assertErrorMessage");
const assertErrorDetails = document.getElementById("assertErrorDetails");
const assertErrorCloseBtn = document.getElementById("assertErrorCloseBtn");
const previewConfirmModal = document.getElementById("previewConfirmModal");
const previewConfirmMessage = document.getElementById("previewConfirmMessage");
const previewConfirmDetails = document.getElementById("previewConfirmDetails");
const previewGoBtn = document.getElementById("previewGoBtn");
const previewNoGoBtn = document.getElementById("previewNoGoBtn");

const panelScopes = [transformsPanel, optionsPanel, modelsPanel, insightsPanel, resultsPanel];

export {
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
};
