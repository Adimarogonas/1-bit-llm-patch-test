const { app, BrowserWindow, ipcMain, dialog } = require("electron");
const path = require("path");
const fs = require("fs");
const { spawn } = require("child_process");

const repoRoot = path.resolve(__dirname, "..", "..");
const evalRunsDir = path.join(repoRoot, "results", "eval_runs");
const pythonBin = fs.existsSync(path.join(repoRoot, ".venv", "bin", "python"))
  ? path.join(repoRoot, ".venv", "bin", "python")
  : "python3";

let mainWindow;
let activeProcess = null;

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1280,
    height: 840,
    minWidth: 1040,
    minHeight: 720,
    title: "Bankai Eval Studio",
    backgroundColor: "#f6f7f9",
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false
    }
  });
  mainWindow.loadFile(path.join(__dirname, "renderer", "index.html"));
}

app.whenReady().then(createWindow);
app.on("window-all-closed", () => {
  if (process.platform !== "darwin") app.quit();
});
app.on("activate", () => {
  if (BrowserWindow.getAllWindows().length === 0) createWindow();
});

function sidecarArgs(args) {
  return ["-m", "bankai_poc.cli", ...args];
}

function sidecarEnv() {
  return {
    ...process.env,
    PYTHONPATH: path.join(repoRoot, "src"),
    PYTHONUNBUFFERED: "1"
  };
}

function runSidecar(args, eventName) {
  return new Promise((resolve, reject) => {
    if (activeProcess) {
      reject(new Error("A sidecar process is already running."));
      return;
    }
    const child = spawn(pythonBin, sidecarArgs(args), {
      cwd: repoRoot,
      env: sidecarEnv()
    });
    activeProcess = child;
    let stdout = "";
    let stderr = "";

    child.stdout.on("data", (chunk) => {
      const text = chunk.toString();
      stdout += text;
      mainWindow?.webContents.send(eventName, { stream: "stdout", text });
    });
    child.stderr.on("data", (chunk) => {
      const text = chunk.toString();
      stderr += text;
      mainWindow?.webContents.send(eventName, { stream: "stderr", text });
    });
    child.on("error", (error) => {
      activeProcess = null;
      reject(error);
    });
    child.on("close", (code) => {
      activeProcess = null;
      if (code === 0) {
        resolve({ stdout, stderr });
      } else {
        reject(new Error(stderr || stdout || `Sidecar exited with code ${code}`));
      }
    });
  });
}

ipcMain.handle("dataset:generate", async (_event, config) => {
  const args = [
    "eval-dataset",
    "--csv-source",
    config.csvSource,
    "--task-name",
    config.taskName,
    "--prompt-column",
    config.promptColumn,
    "--expected-column",
    config.expectedColumn,
    "--grader-column",
    config.graderColumn,
    "--id-column",
    config.idColumn,
    "--system-column",
    config.systemColumn,
    "--assessment-column",
    config.assessmentColumn,
    "--limit",
    String(config.limit),
    "--seed",
    String(config.seed),
    "--output",
    config.output
  ];
  if (!config.shuffle) args.push("--no-shuffle");
  const result = await runSidecar(args, "sidecar:log");
  return { path: result.stdout.trim().split(/\r?\n/).at(-1) };
});

ipcMain.handle("pipeline:run", async (_event, config) => {
  const runDir = path.join(config.outputDir, `run_${timestampForPath()}`);
  const args = [
    "eval-run",
    "--dataset",
    config.dataset,
    "--output-dir",
    runDir,
    "--max-tokens",
    String(config.maxTokens),
    "--bankai-model",
    config.bankaiModel,
    "--bankai-patch",
    config.bankaiPatch,
    "--reference-model",
    config.referenceModel,
    "--agent-name",
    config.agentName,
    "--agent-command",
    config.agentCommand,
    "--agent-timeout",
    String(config.agentTimeout),
    "--judge-command",
    config.judgeCommand,
    "--judge-timeout",
    String(config.judgeTimeout)
  ];
  if (!config.bankaiEnabled) args.push("--no-bankai");
  if (!config.referenceEnabled) args.push("--no-reference");
  if (!config.agentEnabled) args.push("--no-agent");
  const result = await runSidecar(args, "sidecar:log");
  return { ...parseRunOutput(result.stdout), runDir };
});

ipcMain.handle("pipeline:adapt", async (_event, config) => {
  const args = [
    "eval-adapt",
    "--dataset",
    config.dataset,
    "--model",
    config.model,
    "--output-dir",
    config.outputDir,
    "--max-tokens",
    String(config.maxTokens),
    "--rounds",
    String(config.rounds),
    "--pool",
    String(config.pool),
    "--topk",
    String(config.topk),
    "--accept-per-round",
    String(config.acceptPerRound),
    "--search-mode",
    config.searchMode,
    "--max-iters",
    String(config.maxIters),
    "--fitness-mode",
    config.fitnessMode,
    "--target-probes",
    String(config.targetProbes),
    "--control-probes",
    String(config.controlProbes),
    "--candidate-rows",
    String(config.candidateRows),
    "--max-flips",
    String(config.maxFlips),
    "--control-penalty",
    String(config.controlPenalty),
    "--judge-command",
    config.judgeCommand,
    "--judge-timeout",
    String(config.judgeTimeout)
  ];
  if (config.layers && config.layers.trim()) {
    args.push("--layers", config.layers.trim());
  } else {
    args.push("--layer-profile", config.layerProfile);
  }
  args.push(config.impactWeighted ? "--impact-weighted" : "--no-impact-weighted");
  const result = await runSidecar(args, "sidecar:log");
  const manifest = result.stdout.trim().split(/\r?\n/).find((line) => line.startsWith("manifest="))?.slice("manifest=".length) || null;
  return { manifest };
});

ipcMain.handle("runs:list", async () => listRuns());
ipcMain.handle("runs:read", async (_event, runDir) => readRun(runDir));
ipcMain.handle("dialog:csv", async () => {
  const result = await dialog.showOpenDialog(mainWindow, {
    title: "Choose eval CSV",
    defaultPath: path.join(repoRoot, "data", "practical_evals"),
    filters: [{ name: "CSV files", extensions: ["csv"] }],
    properties: ["openFile"]
  });
  return result.canceled ? null : result.filePaths[0];
});
ipcMain.handle("dialog:directory", async () => {
  const result = await dialog.showOpenDialog(mainWindow, {
    title: "Choose output folder",
    defaultPath: evalRunsDir,
    properties: ["openDirectory", "createDirectory"]
  });
  return result.canceled ? null : result.filePaths[0];
});
ipcMain.handle("app:paths", async () => ({
  repoRoot,
  evalRunsDir,
  defaultDataset: path.join(evalRunsDir, "dataset_latest.jsonl"),
  defaultCsv: path.join(repoRoot, "data", "practical_evals", "example_sentiment.csv"),
  defaultPatch: path.join(repoRoot, "patches", "gsm8k_real_patch.json"),
  pythonBin
}));

function parseRunOutput(stdout) {
  const lines = stdout.trim().split(/\r?\n/);
  const summary = lines.find((line) => line.startsWith("summary="))?.slice("summary=".length) || null;
  const details = lines.find((line) => line.startsWith("details="))?.slice("details=".length) || null;
  return { summary, details };
}

function listRuns() {
  if (!fs.existsSync(evalRunsDir)) return [];
  const runs = [];
  for (const entry of fs.readdirSync(evalRunsDir, { withFileTypes: true })) {
    if (!entry.isDirectory()) continue;
    const dir = path.join(evalRunsDir, entry.name);
    const adaptiveManifest = path.join(dir, "adaptive_manifest.json");
    if (fs.existsSync(adaptiveManifest)) {
      runs.push(runListItem(entry.name, dir, path.join(dir, "patched", "summary.json"), "adaptive"));
      continue;
    }
    runs.push(runListItem(entry.name, dir, path.join(dir, "summary.json"), "standard"));
  }
  return runs.sort((a, b) => b.modifiedMs - a.modifiedMs);
}

function runListItem(name, dir, summaryPath, kind) {
  const stat = fs.statSync(dir);
  let summary = null;
  if (fs.existsSync(summaryPath)) {
    try {
      summary = JSON.parse(fs.readFileSync(summaryPath, "utf8"));
    } catch (_error) {
      summary = null;
    }
  }
  return { name, dir, kind, modifiedMs: stat.mtimeMs, summary };
}

function readRun(runDir) {
  const adaptiveManifest = path.join(runDir, "adaptive_manifest.json");
  const summaryPath = fs.existsSync(adaptiveManifest) ? path.join(runDir, "patched", "summary.json") : path.join(runDir, "summary.json");
  const detailsPath = fs.existsSync(adaptiveManifest) ? path.join(runDir, "patched", "details.json") : path.join(runDir, "details.json");
  return {
    summaryPath,
    detailsPath,
    manifest: fs.existsSync(adaptiveManifest) ? JSON.parse(fs.readFileSync(adaptiveManifest, "utf8")) : null,
    baseSummary: fs.existsSync(adaptiveManifest) && fs.existsSync(path.join(runDir, "base", "summary.json"))
      ? JSON.parse(fs.readFileSync(path.join(runDir, "base", "summary.json"), "utf8"))
      : null,
    baseDetails: fs.existsSync(adaptiveManifest) && fs.existsSync(path.join(runDir, "base", "details.json"))
      ? JSON.parse(fs.readFileSync(path.join(runDir, "base", "details.json"), "utf8"))
      : null,
    summary: fs.existsSync(summaryPath) ? JSON.parse(fs.readFileSync(summaryPath, "utf8")) : null,
    details: fs.existsSync(detailsPath) ? JSON.parse(fs.readFileSync(detailsPath, "utf8")) : null
  };
}

function timestampForPath() {
  const now = new Date();
  const pad = (value) => String(value).padStart(2, "0");
  return `${now.getFullYear()}${pad(now.getMonth() + 1)}${pad(now.getDate())}_${pad(now.getHours())}${pad(now.getMinutes())}${pad(now.getSeconds())}`;
}
