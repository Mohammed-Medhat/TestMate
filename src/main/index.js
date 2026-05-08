const { app, BrowserWindow, ipcMain, dialog } = require('electron')
const { spawn } = require('child_process')
const path = require('path')
const fs = require('fs')

let mainWindow
let activeProcess = null
let fastapiProcess = null
const FASTAPI_PORT = 8001

function startFastAPI() {
  const appRoot = app.isPackaged ? process.resourcesPath : app.getAppPath()
  const moduleDir = path.join(appRoot, 'readme_feature_extractor')
  const pythonPath = app.isPackaged
    ? 'python'
    : path.join(appRoot, '.venv', 'Scripts', 'python.exe')
  const modelPath = path.join(moduleDir, 'models', 'qwen2.5-coder-7b-instruct-q5_k_m.gguf')

  console.log('Starting FastAPI server...')
  fastapiProcess = spawn(
    pythonPath,
    ['-m', 'uvicorn', 'src.main:app', '--host', '127.0.0.1', '--port', String(FASTAPI_PORT)],
    {
      cwd: moduleDir,
      env: { ...process.env, MODEL_PATH: modelPath },
    }
  )

  fastapiProcess.stdout.on('data', (d) => console.log('[FastAPI]', d.toString().trim()))
  fastapiProcess.stderr.on('data', (d) => console.log('[FastAPI]', d.toString().trim()))
  fastapiProcess.on('error', (err) => console.error('[FastAPI] failed to start:', err))
  fastapiProcess.on('close', (code) => {
    console.log('[FastAPI] exited with code', code)
    fastapiProcess = null
  })
}

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
  startFastAPI()
  createWindow()

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow()
    }
  })
})

app.on('window-all-closed', () => {
  if (activeProcess) activeProcess.kill()
  if (fastapiProcess) fastapiProcess.kill()

  if (process.platform !== 'darwin') {
    app.quit()
  }
})

// ─── IPC: Run Python pipeline ──────────────────────────────────────────────
ipcMain.handle('run-pipeline', async (_event, { inputFile }) => {
  if (activeProcess) {
    activeProcess.kill()
    activeProcess = null
  }

  return new Promise((resolve, reject) => {
    try {
      if (!inputFile) {
        return reject(new Error('Input file is required'))
      }

      const basePath = app.isPackaged
        ? path.join(process.resourcesPath, 'srs2testcases')
        : path.join(app.getAppPath(), 'srs2testcases')

      const scriptPath = path.join(basePath, 'run_pipeline.py')

      // Use venv python in development
      const pythonPath = app.isPackaged
        ? 'python'
        : path.join(app.getAppPath(), '.venv', 'Scripts', 'python.exe')

      // Ensure results directory exists
      const resultsDir = path.join(basePath, 'results')

      if (!fs.existsSync(resultsDir)) {
        fs.mkdirSync(resultsDir, { recursive: true })
      }

      // Create output JSON file path
      const outputPath = path.join(
        resultsDir,
        `${path.parse(inputFile).name}_requirements.json`
      )

      const alignedDatasetPath = path.join(basePath, 'data', 'aligned', 'pure_train_aligned_full.json')

      const args = [
        scriptPath,
        '--input',
        inputFile,
        '--output',
        outputPath,
        '--aligned-dataset',
        alignedDatasetPath,
      ]

      console.log('PYTHON PATH:', pythonPath)
      console.log('SCRIPT PATH:', scriptPath)
      console.log('ARGS:', args)

      let stderrData = ''

      activeProcess = spawn(pythonPath, args, {
        cwd: path.dirname(scriptPath),
      })

      activeProcess.stdout.on('data', (data) => {
        const text = data.toString()

        console.log('PYTHON STDOUT:', text)

        mainWindow?.webContents.send('pipeline-output', {
          type: 'stdout',
          data: text,
        })
      })

      activeProcess.stderr.on('data', (data) => {
        const text = data.toString()

        stderrData += text

        console.error('PYTHON STDERR:', text)

        mainWindow?.webContents.send('pipeline-output', {
          type: 'stderr',
          data: text,
        })
      })

      activeProcess.on('close', (code) => {
        activeProcess = null

        console.log('PIPELINE EXIT CODE:', code)

        mainWindow?.webContents.send('pipeline-done', {
          exitCode: code,
        })

        if (code === 0) {
          resolve({
            success: true,
            exitCode: code,
            outputPath,
          })
        } else {
          reject(
            new Error(
              `Pipeline exited with code ${code}\n\n${stderrData}`
            )
          )
        }
      })

      activeProcess.on('error', (err) => {
        activeProcess = null

        console.error('FAILED TO START PYTHON PROCESS:', err)

        reject(err)
      })
    } catch (err) {
      reject(err)
    }
  })
})

// ─── IPC: Open native file dialog ──────────────────────────────────────────
ipcMain.handle('open-file-dialog', async (_event, opts = {}) => {
  const filters = opts.readme
    ? [
        { name: 'README Files', extensions: ['md', 'txt'] },
        { name: 'All Files', extensions: ['*'] },
      ]
    : [
        { name: 'SRS Documents', extensions: ['docx', 'pdf'] },
        { name: 'All Files', extensions: ['*'] },
      ]
  return dialog.showOpenDialog(mainWindow, {
    title: opts.readme ? 'Select README File' : 'Select SRS Document',
    properties: ['openFile'],
    filters,
  })
})

// ─── IPC: Read a specific file ─────────────────────────────────────────────
ipcMain.handle('read-file', async (_event, { filePath }) => {
  try {
    const content = fs.readFileSync(filePath, 'utf-8')
    return { success: true, data: JSON.parse(content) }
  } catch (err) {
    return { success: false, error: err.message }
  }
})

// ─── IPC: Read a text file (plain text, no JSON parsing) ───────────────────
ipcMain.handle('read-text-file', async (_event, { filePath }) => {
  try {
    const content = fs.readFileSync(filePath, 'utf-8')
    return { success: true, data: content }
  } catch (err) {
    return { success: false, error: err.message }
  }
})

// ─── IPC: README Extractor — extract features from GitHub repo URL ─────────
ipcMain.handle('readme-extract-repo', async (_event, { repoUrl }) => {
  try {
    const res = await fetch(`http://127.0.0.1:${FASTAPI_PORT}/api/v1/extract_features`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ repo_name: repoUrl }),
    })
    if (!res.ok) {
      const err = await res.json().catch(() => ({}))
      return { success: false, error: err.detail || `HTTP ${res.status}` }
    }
    return { success: true, data: await res.json() }
  } catch (err) {
    return { success: false, error: err.message }
  }
})

// ─── IPC: README Extractor — generate tests from GitHub repo URL ────────────
ipcMain.handle('readme-generate-tests-repo', async (_event, { repoUrl, extractionId }) => {
  try {
    const res = await fetch(`http://127.0.0.1:${FASTAPI_PORT}/api/v1/scenarios/generate_scenarios`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ repo_name: repoUrl, extraction_id: extractionId }),
    })
    if (!res.ok) {
      const err = await res.json().catch(() => ({}))
      return { success: false, error: err.detail || `HTTP ${res.status}` }
    }
    return { success: true, data: await res.json() }
  } catch (err) {
    return { success: false, error: err.message }
  }
})

// ─── IPC: README Extractor — extract features from local file ──────────────
ipcMain.handle('readme-extract', async (_event, { content }) => {
  try {
    const res = await fetch(`http://127.0.0.1:${FASTAPI_PORT}/api/v1/extract_local`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ content }),
    })
    if (!res.ok) {
      const err = await res.json().catch(() => ({}))
      return { success: false, error: err.detail || `HTTP ${res.status}` }
    }
    return { success: true, data: await res.json() }
  } catch (err) {
    return { success: false, error: err.message }
  }
})

// ─── IPC: README Extractor — generate test cases ───────────────────────────
ipcMain.handle('readme-generate-tests', async (_event, { content, extractionId }) => {
  try {
    const res = await fetch(`http://127.0.0.1:${FASTAPI_PORT}/api/v1/scenarios/generate_local`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ content, extraction_id: extractionId }),
    })
    if (!res.ok) {
      const err = await res.json().catch(() => ({}))
      return { success: false, error: err.detail || `HTTP ${res.status}` }
    }
    return { success: true, data: await res.json() }
  } catch (err) {
    return { success: false, error: err.message }
  }
})

// ─── IPC: Read results directory ───────────────────────────────────────────
ipcMain.handle('get-results', async () => {
  const resultsDir = app.isPackaged
    ? path.join(process.resourcesPath, 'srs2testcases', 'results')
    : path.join(app.getAppPath(), 'srs2testcases', 'results')

  try {
    if (!fs.existsSync(resultsDir)) {
      return {
        success: false,
        error: 'Results directory not found',
        results: [],
      }
    }

    const files = fs
      .readdirSync(resultsDir)
      .filter((f) => f.endsWith('.json'))

    const results = files.map((file) => {
      const content = fs.readFileSync(
        path.join(resultsDir, file),
        'utf-8'
      )

      return {
        file,
        data: JSON.parse(content),
      }
    })

    return {
      success: true,
      results,
    }
  } catch (err) {
    return {
      success: false,
      error: err.message,
      results: [],
    }
  }
})