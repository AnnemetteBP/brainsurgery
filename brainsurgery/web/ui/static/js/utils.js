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

function regexToBackrefTemplate(raw) {
  let count = 0;
  let out = "";
  let escaped = false;
  let inClass = false;
  for (let i = 0; i < raw.length; i += 1) {
    const ch = raw[i];
    if (escaped) { out += "\\" + ch; escaped = false; continue; }
    if (ch === "\\") { escaped = true; continue; }
    if (ch === "[") { inClass = true; out += ch; continue; }
    if (ch === "]") { inClass = false; out += ch; continue; }
    if (!inClass && ch === "(") {
      const next = raw.slice(i + 1, i + 3);
      if (next === "?:" || next === "?=" || next === "?!" || next === "?<" || next === "?>") { out += ch; continue; }
      count += 1;
      out += "\\" + String(count);
      let depth = 1;
      let j = i + 1;
      let esc2 = false;
      let cls2 = false;
      for (; j < raw.length; j += 1) {
        const c2 = raw[j];
        if (esc2) { esc2 = false; continue; }
        if (c2 === "\\") { esc2 = true; continue; }
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

export {
  buildReferenceFromModel,
  copyFromFilterToToTemplate,
  copyTextToClipboard,
  parseFieldValue,
  tensorCountText,
};
