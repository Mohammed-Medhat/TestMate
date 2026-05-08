const { contextBridge, ipcRenderer } = require('electron')

contextBridge.exposeInMainWorld('electronAPI', {
  // Spawn python run_pipeline.py with args; resolves when done
  runPipeline: (data) => ipcRenderer.invoke('run-pipeline', data)
  // Open native file picker — returns { canceled, filePaths }
  openFileDialog: () => ipcRenderer.invoke('open-file-dialog'),

  // Read JSON files from srs2testcases/results/
  getResults: () => ipcRenderer.invoke('get-results'),

  // Stream stdout/stderr from the Python process
  // Returns an unsubscribe function
  onPipelineOutput: (callback) => {
    const handler = (_, payload) => callback(payload)
    ipcRenderer.on('pipeline-output', handler)
    return () => ipcRenderer.removeListener('pipeline-output', handler)
  },

  // Fired once when the Python process exits
  // Returns an unsubscribe function
  onPipelineDone: (callback) => {
    const handler = (_, payload) => callback(payload)
    ipcRenderer.on('pipeline-done', handler)
    return () => ipcRenderer.removeListener('pipeline-done', handler)
  },
})
