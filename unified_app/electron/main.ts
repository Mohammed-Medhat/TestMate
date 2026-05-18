import { app, shell, BrowserWindow, ipcMain } from 'electron'
import { join } from 'path'
import { spawn, ChildProcess } from 'child_process'
import http from 'http'

const API_PORT = 8080
const API_HOST = '127.0.0.1'
const API_URL = `http://${API_HOST}:${API_PORT}`
const POLL_INTERVAL_MS = 500
const MAX_WAIT_MS = 60000

let mainWindow: BrowserWindow | null = null
let pythonProcess: ChildProcess | null = null

function getUnifiedAppDir(): string {
  // dev: __dirname is out/main/ → go up to unified_app/
  return join(__dirname, '..', '..')
}

function getPython(): string {
  return process.platform === 'win32' ? 'python' : 'python3'
}

function startBackend(): void {
  const dir = getUnifiedAppDir()
  const script = join(dir, 'server.py')
  console.log(`[Electron] Starting backend: ${getPython()} ${script}`)

  pythonProcess = spawn(getPython(), [script, '--host', API_HOST, '--port', String(API_PORT)], {
    cwd: dir,
    env: { ...process.env, PYTHONIOENCODING: 'utf-8', PYTHONUTF8: '1' },
    stdio: ['pipe', 'pipe', 'pipe']
  })

  pythonProcess.stdout?.on('data', d => console.log(`[Server] ${d.toString().trim()}`))
  pythonProcess.stderr?.on('data', d => console.log(`[Server] ${d.toString().trim()}`))
  pythonProcess.on('error', err => console.error(`[Electron] Backend error: ${err.message}`))
  pythonProcess.on('exit', code => {
    console.log(`[Electron] Backend exited (code ${code})`)
    pythonProcess = null
    if (code !== 0 && code !== null) {
      setTimeout(() => {
        startBackend()
        waitForBackend()
          .then(() => mainWindow?.webContents.send('backend-restarted'))
          .catch(console.error)
      }, 2000)
    }
  })
}

function waitForBackend(): Promise<void> {
  return new Promise((resolve, reject) => {
    const t0 = Date.now()
    const poll = (): void => {
      const req = http.get(`${API_URL}/api/status`, res => {
        if (res.statusCode === 200) return resolve()
        retry()
      })
      req.on('error', retry)
      req.setTimeout(2000, () => { req.destroy(); retry() })
    }
    const retry = (): void => {
      if (Date.now() - t0 > MAX_WAIT_MS) return reject(new Error('Backend timeout'))
      setTimeout(poll, POLL_INTERVAL_MS)
    }
    poll()
  })
}

function killBackend(): void {
  if (!pythonProcess) return
  if (process.platform === 'win32') {
    spawn('taskkill', ['/pid', String(pythonProcess.pid), '/f', '/t'])
  } else {
    pythonProcess.kill('SIGTERM')
  }
  pythonProcess = null
}

function createWindow(): void {
  mainWindow = new BrowserWindow({
    width: 1440,
    height: 900,
    minWidth: 1100,
    minHeight: 700,
    show: false,
    autoHideMenuBar: true,
    backgroundColor: '#09090b',
    title: 'TestMate',
    webPreferences: {
      preload: join(__dirname, '../preload/index.cjs'),
      sandbox: false,
      contextIsolation: true,
      nodeIntegration: false
    }
  })

  mainWindow.on('ready-to-show', () => {
    mainWindow?.show()
    mainWindow?.focus()
  })

  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    shell.openExternal(url)
    return { action: 'deny' }
  })

  if (!app.isPackaged && process.env['ELECTRON_RENDERER_URL']) {
    mainWindow.loadURL(process.env['ELECTRON_RENDERER_URL'])
  } else {
    mainWindow.loadFile(join(__dirname, '../renderer/index.html'))
  }

  mainWindow.on('closed', () => { mainWindow = null })
}

app.whenReady().then(async () => {
  if (process.platform === 'win32') app.setAppUserModelId('com.testmate.unified')

  ipcMain.on('open-settings', () => {
    BrowserWindow.getFocusedWindow()?.webContents.send('toggle-settings-modal')
  })

  startBackend()
  try {
    await waitForBackend()
    console.log('[Electron] Backend ready')
  } catch {
    console.error('[Electron] Backend did not start — opening anyway')
  }

  createWindow()
  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow()
  })
})

app.on('window-all-closed', () => { killBackend(); app.quit() })
app.on('before-quit', killBackend)
