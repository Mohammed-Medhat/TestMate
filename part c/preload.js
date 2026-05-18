/**
 * preload.js — Context-isolated bridge (renderer ↔ main)
 *
 * Only expose the exact API the renderer needs — nothing more.
 * contextIsolation: true keeps the renderer sandboxed.
 */

const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("electronAPI", {
  /** Receive status messages sent during startup (splash screen). */
  onSplashStatus: (callback) =>
    ipcRenderer.on("splash-status", (_event, msg) => callback(msg)),

  /** Ask the main process for the Flask server URL. */
  getFlaskUrl: () => ipcRenderer.invoke("get-flask-url"),

  /** Tell the main process to reload the Flask page. */
  reloadApp: () => ipcRenderer.send("reload-app"),
});
