import { contextBridge, ipcRenderer } from 'electron'

contextBridge.exposeInMainWorld('api', {
  baseUrl: 'http://127.0.0.1:8080',
  openSettings: () => ipcRenderer.send('open-settings'),
  onToggleSettingsModal: (callback: () => void) => {
    ipcRenderer.on('toggle-settings-modal', callback)
  },
  onBackendCrashed: (callback: (code: number) => void) => {
    ipcRenderer.on('backend-crashed', (_e, code) => callback(code))
  },
  onBackendRestarted: (callback: () => void) => {
    ipcRenderer.on('backend-restarted', callback)
  }
})
