/**
 * TestMate Unified — Electron Main Process
 * ==========================================
 * Spawns unified_app/server.py as a child process,
 * waits for it to be ready, then opens the app window.
 *
 * Follows the same pattern as PartB/desktop/main.js.
 */

const { app, BrowserWindow, shell } = require('electron')
const { spawn } = require('child_process')
const path = require('path')
const http = require('http')

// ── Configuration ───────────────────────────────────────────────────────────
const API_PORT = 8080
const API_HOST = '127.0.0.1'
const API_URL = `http://${API_HOST}:${API_PORT}`
const POLL_INTERVAL_MS = 500
const MAX_WAIT_MS = 30000   // 30 s

let mainWindow = null
let splashWindow = null
let pythonProcess = null

// ── Path helpers ────────────────────────────────────────────────────────────
function getUnifiedAppDir() {
  if (app.isPackaged) {
    return path.join(process.resourcesPath, 'unified_app')
  }
  // Dev: electron/main.js → ../  (unified_app/)
  return path.join(__dirname, '..')
}

function getPythonPath() {
  return process.platform === 'win32' ? 'python' : 'python3'
}

// ── Splash screen ───────────────────────────────────────────────────────────
function createSplashWindow() {
  splashWindow = new BrowserWindow({
    width: 480,
    height: 360,
    frame: false,
    transparent: true,
    resizable: false,
    alwaysOnTop: true,
    webPreferences: { nodeIntegration: false, contextIsolation: true },
  })
  splashWindow.loadFile(path.join(__dirname, 'splash.html'))
  splashWindow.center()
}

// ── Main window ─────────────────────────────────────────────────────────────
function createMainWindow() {
  mainWindow = new BrowserWindow({
    width: 1440,
    height: 900,
    minWidth: 1100,
    minHeight: 700,
    show: false,
    backgroundColor: '#0a0d0c',
    title: 'TestMate',
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      nodeIntegration: false,
      contextIsolation: true,
    },
  })

  mainWindow.loadURL(API_URL)

  mainWindow.once('ready-to-show', () => {
    if (splashWindow) {
      splashWindow.close()
      splashWindow = null
    }
    mainWindow.show()
    mainWindow.focus()
  })

  // Open external links in the system browser, not in Electron
  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    shell.openExternal(url)
    return { action: 'deny' }
  })

  mainWindow.on('closed', () => { mainWindow = null })
}

// ── Start Python backend ────────────────────────────────────────────────────
function startBackend() {
  const unifiedDir = getUnifiedAppDir()
  const serverScript = path.join(unifiedDir, 'server.py')
  const python = getPythonPath()

  console.log(`[Electron] Starting backend: ${python} ${serverScript}`)
  console.log(`[Electron] cwd: ${unifiedDir}`)

  pythonProcess = spawn(
    python,
    [serverScript, '--host', API_HOST, '--port', String(API_PORT)],
    {
      cwd: unifiedDir,
      env: { ...process.env, PYTHONIOENCODING: 'utf-8', PYTHONUTF8: '1' },
      stdio: ['pipe', 'pipe', 'pipe'],
    }
  )

  pythonProcess.stdout.on('data', d => console.log(`[Server] ${d.toString().trim()}`))
  pythonProcess.stderr.on('data', d => console.log(`[Server] ${d.toString().trim()}`))
  pythonProcess.on('error', err => console.error(`[Electron] Backend start failed: ${err.message}`))
  pythonProcess.on('exit', code => {
    console.log(`[Electron] Backend exited (code ${code})`)
    pythonProcess = null
  })
}

// ── Poll until backend is ready ─────────────────────────────────────────────
function waitForBackend() {
  return new Promise((resolve, reject) => {
    const t0 = Date.now()
    const poll = () => {
      const req = http.get(`${API_URL}/api/status`, res => {
        if (res.statusCode === 200) return resolve()
        retry()
      })
      req.on('error', retry)
      req.setTimeout(2000, () => { req.destroy(); retry() })
    }
    const retry = () => {
      if (Date.now() - t0 > MAX_WAIT_MS) {
        return reject(new Error('Backend did not start within 30 s'))
      }
      setTimeout(poll, POLL_INTERVAL_MS)
    }
    poll()
  })
}

// ── Kill backend on exit ────────────────────────────────────────────────────
function killBackend() {
  if (!pythonProcess) return
  console.log('[Electron] Killing backend...')
  if (process.platform === 'win32') {
    spawn('taskkill', ['/pid', String(pythonProcess.pid), '/f', '/t'])
  } else {
    pythonProcess.kill('SIGTERM')
  }
  pythonProcess = null
}

// ── App lifecycle ───────────────────────────────────────────────────────────
app.whenReady().then(async () => {
  createSplashWindow()
  startBackend()

  try {
    await waitForBackend()
    console.log('[Electron] Backend ready — opening window')
  } catch (err) {
    console.error('[Electron] Backend timeout:', err.message)
    // Try to open anyway; the user will see an error in the window
  }

  createMainWindow()
})

app.on('window-all-closed', () => {
  killBackend()
  app.quit()
})

app.on('before-quit', killBackend)

app.on('activate', () => {
  if (BrowserWindow.getAllWindows().length === 0) createMainWindow()
})
