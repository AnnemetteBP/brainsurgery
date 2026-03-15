function bindPanelInteractions({
  panelScopes,
  transformsPanel,
  optionsPanel,
  transformsEl,
  setFocusedPanel,
  runCurrentTransformFromKeyboard,
}) {
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
}

export { bindPanelInteractions };
