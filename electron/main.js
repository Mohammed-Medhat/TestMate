const { app, BrowserWindow, ipcMain, dialog } = require('electron')
const { spawn } = require('child_process')
const path = require('path')
const fs = require('fs')

let mainWindow
let activeProcess = null

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1400,
    height: 900,
    minWidth: 1100,
    minHeight: 700,
    backgroundColor: '#09090b',
    frame: true,
    titleBarStyle: 'default',
    webPreferences: {
      preload: path.join(__dirname, '../preload/index.js'),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: false,
    },
  })

  if (!app.isPackaged && process.env['ELECTRON_RENDERER_URL']) {
    mainWindow.loadURL(process.env['ELECTRON_RENDERER_URL'])
  } else {
    mainWindow.loadFile(path.join(__dirname, '../renderer/index.html'))
  }
}

app.whenReady().then(() => {
  createWindow()
  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow()
  })
})

app.on('window-all-closed', () => {
  if (activeProcess) activeProcess.kill()
  if (process.platform !== 'darwin') app.quit()
})

// ─── IPC: Run Python pipeline ──────────────────────────────────────────────
ipcMain.handle('run-pipeline', async (_event, { inputFile, outputDir }) => {
  if (activeProcess) {
    activeProcess.kill()
    activeProcess = null
  }

  return new Promise((resolve, reject) => {
    const scriptPath = app.isPackaged
      ? path.join(process.resourcesPath, 'srs2testcases', 'run_pipeline.py')
      : path.join(app.getAppPath(), 'srs2testcases', 'run_pipeline.py')

    const args = [scriptPath, '--input', inputFile]
    if (outputDir) args.push('--output', outputDir)

    activeProcess = spawn('python', args, { cwd: path.dirname(scriptPath) })

    activeProcess.stdout.on('data', (data) => {
      mainWindow?.webContents.send('pipeline-output', {
        type: 'stdout',
        data: data.toString(),
      })
    })

    activeProcess.stderr.on('data', (data) => {
      mainWindow?.webContents.send('pipeline-output', {
        type: 'stderr',
        data: data.toString(),
      })
    })

    activeProcess.on('close', (code) => {
      activeProcess = null
      mainWindow?.webContents.send('pipeline-done', { exitCode: code })
      if (code === 0) {
        resolve({ success: true, exitCode: code })
      } else {
        reject(new Error(`Pipeline exited with code ${code}`))
      }
    })

    activeProcess.on('error', (err) => {
      activeProcess = null
      reject(err)
    })
  })
})

// ─── IPC: Open native file dialog ──────────────────────────────────────────
ipcMain.handle('open-file-dialog', async () => {
  const result = await dialog.showOpenDialog(mainWindow, {
    title: 'Select SRS Document',
    properties: ['openFile'],
    filters: [
      { name: 'SRS Documents', extensions: ['docx', 'pdf'] },
      { name: 'All Files', extensions: ['*'] },
    ],
  })
  return result
})

// ─── IPC: Read results directory ───────────────────────────────────────────
ipcMain.handle('get-results', async () => {
  const resultsDir = app.isPackaged
    ? path.join(process.resourcesPath, 'srs2testcases', 'results')
    : path.join(app.getAppPath(), 'srs2testcases', 'results')

  try {
    if (!fs.existsSync(resultsDir)) {
      return { success: false, error: 'Results directory not found', results: [] }
    }
    const files = fs.readdirSync(resultsDir).filter((f) => f.endsWith('.json'))
    const results = files.map((file) => {
      const content = fs.readFileSync(path.join(resultsDir, file), 'utf-8')
      return { file, data: JSON.parse(content) }
    })
    return { success: true, results }
  } catch (err) {
    return { success: false, error: err.message, results: [] }
  }
})
