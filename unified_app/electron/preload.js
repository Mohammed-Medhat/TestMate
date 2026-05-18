/**
 * TestMate Unified — Preload Script
 */
const { contextBridge } = require('electron')

contextBridge.exposeInMainWorld('testmateDesktop', {
  isElectron: true,
  platform: process.platform,
})
