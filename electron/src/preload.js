const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("bankai", {
  paths: () => ipcRenderer.invoke("app:paths"),
  generateDataset: (config) => ipcRenderer.invoke("dataset:generate", config),
  runPipeline: (config) => ipcRenderer.invoke("pipeline:run", config),
  runAdaptivePipeline: (config) => ipcRenderer.invoke("pipeline:adapt", config),
  listRuns: () => ipcRenderer.invoke("runs:list"),
  readRun: (runDir) => ipcRenderer.invoke("runs:read", runDir),
  chooseCsv: () => ipcRenderer.invoke("dialog:csv"),
  chooseDirectory: () => ipcRenderer.invoke("dialog:directory"),
  onSidecarLog: (callback) => {
    const listener = (_event, payload) => callback(payload);
    ipcRenderer.on("sidecar:log", listener);
    return () => ipcRenderer.removeListener("sidecar:log", listener);
  }
});
