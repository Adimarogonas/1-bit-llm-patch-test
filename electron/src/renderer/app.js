const state = { paths: null };
const $ = (id) => document.getElementById(id);

window.addEventListener("DOMContentLoaded", async () => {
  state.paths = await window.bankai.paths();
  $("runtime").textContent = `Python sidecar: ${state.paths.pythonBin}`;
  $("csvSource").value = state.paths.defaultCsv;
  $("outputDir").value = state.paths.evalRunsDir;

  document.querySelectorAll(".nav-item").forEach((button) => {
    button.addEventListener("click", () => switchView(button.dataset.view));
  });
  $("chooseCsv").addEventListener("click", chooseCsv);
  $("chooseOutput").addEventListener("click", chooseOutput);
  $("runAdaptive").addEventListener("click", runAdaptive);
  $("computePreset").addEventListener("change", applyComputePreset);
  $("refreshRuns").addEventListener("click", refreshRuns);
  $("clearLogs").addEventListener("click", () => { $("log").textContent = ""; });

  window.bankai.onSidecarLog(({ stream, text }) => appendLog(`[${stream}] ${text}`));
  await refreshRuns();
});

function switchView(view) {
  document.querySelectorAll(".nav-item").forEach((button) => button.classList.toggle("active", button.dataset.view === view));
  document.querySelectorAll(".view").forEach((panel) => panel.classList.toggle("active", panel.id === view));
}

async function chooseCsv() {
  const path = await window.bankai.chooseCsv();
  if (path) $("csvSource").value = path;
}

async function chooseOutput() {
  const path = await window.bankai.chooseDirectory();
  if (path) $("outputDir").value = path;
}

function adaptiveConfig() {
  return {
    dataset: $("csvSource").value,
    model: $("modelRef").value,
    outputDir: $("outputDir").value,
    maxTokens: Number($("maxTokens").value),
    rounds: Number($("rounds").value),
    pool: Number($("pool").value),
    topk: Number($("topk").value),
    acceptPerRound: Number($("acceptPerRound").value),
    searchMode: $("searchMode").value,
    maxIters: Number($("maxIters").value),
    fitnessMode: $("fitnessMode").value,
    targetProbes: Number($("targetProbes").value),
    controlProbes: Number($("controlProbes").value),
    candidateRows: Number($("candidateRows").value),
    maxFlips: Number($("maxFlips").value),
    controlPenalty: Number($("controlPenalty").value),
    layers: $("layers").value,
    layerProfile: $("layerProfile").value,
    impactWeighted: $("impactWeighted").checked,
    judgeCommand: $("judgeCommand").value,
    judgeTimeout: Number($("judgeTimeout").value)
  };
}

function applyComputePreset() {
  const preset = $("computePreset").value;
  const values = {
    balanced: { maxTokens: 200, rounds: 4, pool: 8, topk: 2, acceptPerRound: 1, searchMode: "shortlist", maxIters: 200, fitnessMode: "mean", targetProbes: 12, controlProbes: 6, candidateRows: 48, maxFlips: 16, layerProfile: "stable", layers: "" },
    large: { maxTokens: 320, rounds: 10, pool: 24, topk: 4, acceptPerRound: 2, searchMode: "greedy", maxIters: 600, fitnessMode: "mean", targetProbes: 32, controlProbes: 16, candidateRows: 96, maxFlips: 32, layerProfile: "balanced", layers: "" },
    max: { maxTokens: 512, rounds: 18, pool: 48, topk: 8, acceptPerRound: 3, searchMode: "greedy", maxIters: 1200, fitnessMode: "min", targetProbes: 64, controlProbes: 32, candidateRows: 0, maxFlips: 48, layerProfile: "aggressive", layers: "" }
  }[preset];
  for (const [key, value] of Object.entries(values)) {
    const element = $(key);
    if (element) element.value = value;
  }
}

async function runAdaptive() {
  if (!$("csvSource").value) {
    alert("Choose a CSV file first.");
    return;
  }
  try {
    setStatus("Running adaptive eval", "Base eval → dynamic probes → patch search → patched eval");
    switchView("logs");
    const result = await window.bankai.runAdaptivePipeline(adaptiveConfig());
    setStatus("Complete", result.manifest || "Adaptive run complete");
    await refreshRuns();
  } catch (error) {
    setStatus("Run failed", error.message);
    appendLog(`[error] ${error.message}\n`);
  }
}

function setStatus(label, detail = "") {
  $("runStatus").textContent = label;
  $("lastArtifact").textContent = detail || $("lastArtifact").textContent;
}

function appendLog(text) {
  const log = $("log");
  log.textContent += text;
  log.scrollTop = log.scrollHeight;
  updateStatusFromLog(text);
}

function updateStatusFromLog(text) {
  if (text.includes("Step 1/4")) setStatus("Running base eval", "Generating base model predictions");
  if (text.includes("Step 2/4")) setStatus("Building probes", "Converting failures into dynamic probes");
  if (text.includes("Step 3/4")) setStatus("Searching patch", "Screening candidate row flips");
  if (text.includes("Step 4/4")) setStatus("Re-evaluating patch", "Running patched model on the same CSV");
  const round = text.match(/\[round (\d+)\/(\d+)\]/);
  if (round) setStatus("Searching patch", `Round ${round[1]} of ${round[2]}`);
  const greedy = text.match(/\[greedy (\d+)\/(\d+)\]/);
  if (greedy) setStatus("Searching patch", `Greedy step ${greedy[1]} of ${greedy[2]}`);
  const accepted = text.match(/ACCEPTED (L[^\s]+)/);
  if (accepted) setStatus("Patch improved", `Accepted ${accepted[1]}`);
  const greedyAccepted = text.match(/ACCEPT (L[^\s]+)/);
  if (greedyAccepted) setStatus("Patch improved", `Accepted ${greedyAccepted[1]}`);
}

async function refreshRuns() {
  const runs = await window.bankai.listRuns();
  const list = $("runList");
  list.innerHTML = "";
  if (!runs.length) {
    list.innerHTML = `<p>No completed eval runs yet.</p>`;
    return;
  }
  for (const run of runs) {
    const item = document.createElement("article");
    item.className = "run-item";
    const rows = run.summary?.summary || [];
    const allRows = rows.filter((row) => row.benchmark === "all");
    const label = allRows.length ? allRows.map((row) => `${row.model}: ${formatAccuracy(row)}`).join(" · ") : "Open run";
    item.innerHTML = `<strong>${escapeHtml(run.name)}</strong><span>${new Date(run.modifiedMs).toLocaleString()}</span><span>${escapeHtml(label)}</span>`;
    item.addEventListener("click", () => loadRun(run.dir));
    list.appendChild(item);
  }
}

async function loadRun(runDir) {
  const run = await window.bankai.readRun(runDir);
  renderRun(run);
  switchView("results");
}

function renderRun(run) {
  $("resultTitle").textContent = run.summaryPath;
  const rows = run.summary?.summary || [];
  const baseRows = run.baseSummary?.summary || [];
  const detailRows = run.details?.rows || [];
  const comparison = buildComparison(run.baseDetails?.rows || [], detailRows);
  const allRows = rows.filter((row) => row.benchmark === "all");
  const baseAll = baseRows.find((row) => row.benchmark === "all");
  const cards = [];
  if (baseAll) {
    cards.push(metricCard("Base model", formatAccuracy(baseAll), `${baseAll.correct}/${baseAll.scored_examples || 0} correct · ${baseAll.examples} rows`));
  }
  for (const row of allRows.slice(0, 2)) {
    cards.push(metricCard(row.model, formatAccuracy(row), `${row.correct}/${row.scored_examples || 0} correct · ${row.examples} rows`));
  }
  if (run.manifest) {
    cards.push(metricCard("Patch", `${run.manifest.patch_flips || 0} flips`, `Probe score ${Number(run.manifest.best_probe_score || 0).toFixed(4)}`));
    cards.push(metricCard("Changed outputs", `${run.manifest.changed_generations || 0}`, "Base vs patched generations"));
    cards.push(metricCard("Fixed", `${comparison.fixed.length}`, "Base failed, patch passed"));
    cards.push(metricCard("Regressions", `${comparison.regressed.length}`, "Base passed, patch failed"));
  }
  $("summaryCards").innerHTML = cards.join("");
  $("summaryTable").innerHTML = [...baseRows, ...rows].map((row) => `
    <tr>
      <td>${escapeHtml(row.model)}</td>
      <td>${escapeHtml(row.benchmark)}</td>
      <td>${row.examples}</td>
      <td>${formatAccuracy(row)}</td>
      <td>${row.errors}</td>
      <td>${Math.round(row.mean_latency_ms)} ms</td>
    </tr>
  `).join("");
  $("comparisonSummary").innerHTML = comparison.total ? `
    <div class="comparison-tabs">
      <span class="pass">Fixed ${comparison.fixed.length}</span>
      <span class="fail">Regressed ${comparison.regressed.length}</span>
      <span>Still wrong ${comparison.stillWrong.length}</span>
      <span>Still right ${comparison.stillRight.length}</span>
    </div>
  ` : "";
  $("comparisonExamples").innerHTML = comparison.total ? renderComparisonExamples(comparison) : "";
  $("examples").innerHTML = comparison.total ? "" : detailRows.slice(0, 40).map((row) => `
    <article class="example">
      <header>
        <strong>${escapeHtml(row.model)} · ${escapeHtml(row.benchmark)} · ${escapeHtml(String(row.id))}</strong>
        <span class="${row.score?.passed ? "pass" : "fail"}">${row.score?.passed ? "pass" : "fail"}</span>
      </header>
      ${row.score?.passed ? "" : `<div class="expected">Expected: <strong>${escapeHtml(row.expected || row.reference || "")}</strong></div>`}
      <div class="prediction">${escapeHtml(row.prediction || row.error || "")}</div>
    </article>
  `).join("");
}

function buildComparison(baseRows, patchedRows) {
  const patchedById = new Map(patchedRows.map((row) => [rowKey(row), row]));
  const result = { total: 0, fixed: [], regressed: [], stillWrong: [], stillRight: [] };
  for (const base of baseRows) {
    const patched = patchedById.get(rowKey(base));
    if (!patched) continue;
    result.total += 1;
    const basePassed = base.score?.passed === true;
    const patchedPassed = patched.score?.passed === true;
    const pair = { base, patched };
    if (!basePassed && patchedPassed) result.fixed.push(pair);
    else if (basePassed && !patchedPassed) result.regressed.push(pair);
    else if (!basePassed && !patchedPassed) result.stillWrong.push(pair);
    else result.stillRight.push(pair);
  }
  return result;
}

function rowKey(row) {
  return `${row.benchmark || row.task || ""}:${row.id}`;
}

function renderComparisonExamples(comparison) {
  const sections = [
    ["Regressions", comparison.regressed, "fail"],
    ["Fixed", comparison.fixed, "pass"],
    ["Still wrong", comparison.stillWrong, "fail"]
  ];
  return sections
    .filter(([, pairs]) => pairs.length)
    .map(([title, pairs, statusClass]) => `
      <section class="comparison-section">
        <h3>${escapeHtml(title)}</h3>
        ${pairs.slice(0, 20).map(({ base, patched }) => renderComparisonPair(base, patched, statusClass)).join("")}
      </section>
    `).join("");
}

function renderComparisonPair(base, patched, statusClass) {
  const expected = base.expected || base.reference || patched.expected || patched.reference || "";
  return `
    <article class="example comparison-pair">
      <header>
        <strong>${escapeHtml(base.benchmark)} · ${escapeHtml(String(base.id))}</strong>
        <span class="${statusClass}">${escapeHtml(base.score?.passed ? "pass -> fail" : patched.score?.passed ? "fail -> pass" : "fail -> fail")}</span>
      </header>
      <div class="expected">Expected: <strong>${escapeHtml(expected)}</strong></div>
      <div class="prediction-grid">
        <div>
          <span>Base</span>
          <pre>${escapeHtml(base.prediction || base.error || "")}</pre>
        </div>
        <div>
          <span>Patched</span>
          <pre>${escapeHtml(patched.prediction || patched.error || "")}</pre>
        </div>
      </div>
    </article>
  `;
}

function metricCard(label, value, detail) {
  return `
    <div class="metric">
      <span>${escapeHtml(label)}</span>
      <strong>${escapeHtml(value)}</strong>
      <span>${escapeHtml(detail)}</span>
    </div>
  `;
}

function formatAccuracy(row) {
  return row.scored_examples ? `${(row.accuracy * 100).toFixed(1)}%` : "ungraded";
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}
