function createIteratingProgressController({ iteratingProgressEl }) {
  let iteratingProgressTimer = null;
  let iteratingProgressPollInFlight = false;

  function stop() {
    if (iteratingProgressTimer != null) {
      clearInterval(iteratingProgressTimer);
      iteratingProgressTimer = null;
    }
    iteratingProgressPollInFlight = false;
    iteratingProgressEl.classList.add("hidden");
    iteratingProgressEl.textContent = "";
  }

  async function pollIteratingProgressOnce(expectedTransformName) {
    if (iteratingProgressPollInFlight) return;
    iteratingProgressPollInFlight = true;
    try {
      const response = await fetch("/api/progress");
      const data = await response.json();
      if (!response.ok || !data.ok || !data.progress) return;
      const progress = data.progress;
      if (!progress.iterating || progress.transform !== expectedTransformName) return;
      const completed = Number(progress.completed || 0);
      const total = progress.total == null ? null : Number(progress.total);
      const unit = String(progress.unit || "item");
      const desc = String(progress.desc || expectedTransformName);
      const active = Boolean(progress.active);
      if (total != null && Number.isFinite(total) && total > 0) {
        const pct = Math.min(100, Math.floor((completed / total) * 100));
        iteratingProgressEl.textContent =
          (active ? "Running " : "Completed ") +
          desc + ": " + completed + "/" + total + " " + unit + " (" + pct + "%)";
      } else {
        iteratingProgressEl.textContent =
          (active ? "Running " : "Completed ") +
          desc + ": " + completed + " " + unit;
      }
    } catch (_err) {
      // ignore polling errors; the main transform request handles final status
    } finally {
      iteratingProgressPollInFlight = false;
    }
  }

  function start(transformName, shouldTrack) {
    stop();
    if (!shouldTrack) return;
    iteratingProgressEl.textContent = "Preparing progress for " + transformName + "...";
    iteratingProgressEl.classList.remove("hidden");
    pollIteratingProgressOnce(transformName);
    iteratingProgressTimer = setInterval(() => {
      pollIteratingProgressOnce(transformName);
    }, 250);
  }

  return { start, stop };
}

export { createIteratingProgressController };
