const { contextBridge, ipcRenderer } = require('electron')

contextBridge.exposeInMainWorld('electronAPI', {
  runPipeline: (data) => ipcRenderer.invoke('run-pipeline', data),
  openFileDialog: (opts) => ipcRenderer.invoke('open-file-dialog', opts),
  getResults: () => ipcRenderer.invoke('get-results'),
  readFile: (filePath) => ipcRenderer.invoke('read-file', { filePath }),

  onPipelineOutput: (callback) => {
    const handler = (_, payload) => callback(payload)
    ipcRenderer.on('pipeline-output', handler)
    return () => ipcRenderer.removeListener('pipeline-output', handler)
  },

  onPipelineDone: (callback) => {
    const handler = (_, payload) => callback(payload)
    ipcRenderer.on('pipeline-done', handler)
    return () => ipcRenderer.removeListener('pipeline-done', handler)
  },

  readTextFile: (filePath) => ipcRenderer.invoke('read-text-file', { filePath }),
  readmeExtract: (content) => ipcRenderer.invoke('readme-extract', { content }),
  readmeGenerateTests: (content, extractionId) =>
    ipcRenderer.invoke('readme-generate-tests', { content, extractionId }),
  readmeExtractRepo: (repoUrl) => ipcRenderer.invoke('readme-extract-repo', { repoUrl }),
  readmeGenerateTestsRepo: (repoUrl, extractionId) =>
    ipcRenderer.invoke('readme-generate-tests-repo', { repoUrl, extractionId }),
})
