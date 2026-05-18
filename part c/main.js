/**
 * main.js — Electron Main Process for TestMate APR
 *
 * Lifecycle:
 *  1. Show splash screen immediately (instant feedback)
 *  2. Spawn Flask (web/app.py) as a child process
 *  3. Poll http://localhost:5000 until Flask is ready
 *  4. Load the Flask app in the main BrowserWindow
 *  5. On app quit → kill Flask cleanly
 */

const { app, BrowserWindow, ipcMain, dialog, shell } = require("electron");
const path = require("path");
const { spawn, execSync } = require("child_process");
const http = require("http");
const fs = require("fs");

// ── Config ────────────────────────────────────────────────────────────
const FLASK_PORT = 5000;
const FLASK_URL = `http://localhost:${FLASK_PORT}`;
const POLL_INTERVAL = 500;   // ms between readiness checks
const POLL_TIMEOUT = 60000; // 60s max wait for Flask to start
const IS_DEV = process.argv.includes("--dev");

// ── State ─────────────────────────────────────────────────────────────
let mainWindow = null;
let splashWindow = null;
let flaskProcess = null;
let pollTimer = null;


// ══════════════════════════════════════════════════════════════════════
// 1. Splash window
// ══════════════════════════════════════════════════════════════════════

function createSplashWindow() {
  splashWindow = new BrowserWindow({
    width: 480,
    height: 320,
    frame: false,
    transparent: true,
    resizable: false,
    alwaysOnTop: true,
    center: true,
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });

  splashWindow.loadFile(path.join(__dirname, "splash.html"));
  splashWindow.on("closed", () => { splashWindow = null; });
}


// ══════════════════════════════════════════════════════════════════════
// 2. Main window  (shown after Flask is ready)
// ══════════════════════════════════════════════════════════════════════

function createMainWindow() {
  mainWindow = new BrowserWindow({
    width: 1280,
    height: 820,
    minWidth: 900,
    minHeight: 600,
    show: false,           // reveal only once Flask responds
    title: "TestMate APR",
    backgroundColor: "#0f1117",
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });

  // Open external links in system browser, not inside Electron
  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    if (!url.startsWith(FLASK_URL)) shell.openExternal(url);
    return { action: "deny" };
  });

  if (IS_DEV) mainWindow.webContents.openDevTools({ mode: "detach" });

  mainWindow.on("closed", () => { mainWindow = null; });
}


// ══════════════════════════════════════════════════════════════════════
// 3. Flask process management
// ══════════════════════════════════════════════════════════════════════

/**
 * Resolve which Python executable to use.
 * Priority: venv inside project → system python3 → python
 */
function resolvePython() {
  const projectRoot = __dirname;

  // Common venv locations
  const candidates = [
    "/home/c/miniconda3/envs/apr/bin/python",
    path.join(projectRoot, "venv", "bin", "python"),
    path.join(projectRoot, "venv", "Scripts", "python.exe"),
    path.join(projectRoot, ".venv", "bin", "python"),
    path.join(projectRoot, ".venv", "Scripts", "python.exe"),
    path.join(projectRoot, "env", "bin", "python"),
  ];

  for (const p of candidates) {
    if (fs.existsSync(p)) return p;
  }

  // Fall back to system Python
  try { execSync("python3 --version", { stdio: "ignore" }); return "python3"; }
  catch (_) { }
  return "python";
}

function startFlask() {
  const python = resolvePython();
  const scriptPath = path.join(__dirname, "web", "app.py");

  console.log(`[Flask] Starting: ${python} ${scriptPath}`);

  flaskProcess = spawn(python, [scriptPath], {
    cwd: __dirname,
    env: { ...process.env, FLASK_RUN_PORT: String(FLASK_PORT), PYTHONUNBUFFERED: "1" },
    stdio: ["ignore", "pipe", "pipe"],
  });

  flaskProcess.stdout.on("data", d => console.log(`[Flask] ${d.toString().trim()}`));
  flaskProcess.stderr.on("data", d => console.error(`[Flask] ${d.toString().trim()}`));

  flaskProcess.on("error", err => {
    console.error("[Flask] Failed to start:", err.message);
    dialog.showErrorBox(
      "Python / Flask Error",
      `Could not start the backend.\n\n${err.message}\n\nMake sure Python is installed and requirements.txt is satisfied.`
    );
    app.quit();
  });

  flaskProcess.on("exit", (code, signal) => {
    if (code !== 0 && code !== null) {
      console.warn(`[Flask] Exited with code ${code} (signal: ${signal})`);
    }
  });
}

function killFlask() {
  if (!flaskProcess) return;
  try {
    // Windows needs SIGTERM → taskkill, Unix is fine with kill
    if (process.platform === "win32") {
      execSync(`taskkill /pid ${flaskProcess.pid} /f /t`, { stdio: "ignore" });
    } else {
      flaskProcess.kill("SIGTERM");
    }
  } catch (e) {
    console.warn("[Flask] Could not kill process:", e.message);
  }
  flaskProcess = null;
}


// ══════════════════════════════════════════════════════════════════════
// 4. Readiness polling
// ══════════════════════════════════════════════════════════════════════

/**
 * Returns a Promise that resolves when Flask responds on FLASK_URL,
 * or rejects after POLL_TIMEOUT ms.
 */
function waitForFlask() {
  return new Promise((resolve, reject) => {
    const start = Date.now();

    function check() {
      http.get(FLASK_URL, res => {
        res.resume();                   // drain the response
        clearTimeout(pollTimer);
        resolve();
      }).on("error", () => {
        if (Date.now() - start > POLL_TIMEOUT) {
          clearTimeout(pollTimer);
          reject(new Error(`Flask did not start within ${POLL_TIMEOUT / 1000}s.`));
          return;
        }
        pollTimer = setTimeout(check, POLL_INTERVAL);
      });
    }

    check();
  });
}

/**
 * Notify the splash window about progress (optional — splash can ignore it).
 */
function splashLog(msg) {
  if (splashWindow && !splashWindow.isDestroyed()) {
    splashWindow.webContents.send("splash-status", msg);
  }
  console.log(`[Status] ${msg}`);
}


// ══════════════════════════════════════════════════════════════════════
// 5. App lifecycle
// ══════════════════════════════════════════════════════════════════════

app.whenReady().then(async () => {
  createSplashWindow();
  createMainWindow();

  splashLog("Starting Python backend…");
  startFlask();

  splashLog("Waiting for Flask to be ready…");
  try {
    await waitForFlask();
  } catch (err) {
    dialog.showErrorBox("Startup Failed", err.message);
    app.quit();
    return;
  }

  splashLog("Loading TestMate…");
  await mainWindow.loadURL(FLASK_URL);

  // Reveal main window, close splash
  mainWindow.show();
  if (splashWindow && !splashWindow.isDestroyed()) splashWindow.close();
});

// IPC: renderer can ask for the Flask URL (e.g. for copy-to-clipboard)
ipcMain.handle("get-flask-url", () => FLASK_URL);

// IPC: renderer can trigger a graceful reload of the Flask page
ipcMain.on("reload-app", () => {
  if (mainWindow && !mainWindow.isDestroyed()) mainWindow.loadURL(FLASK_URL);
});

app.on("window-all-closed", () => {
  killFlask();
  if (process.platform !== "darwin") app.quit();
});

app.on("before-quit", killFlask);

app.on("activate", () => {
  // macOS: re-open window when dock icon is clicked
  if (mainWindow === null) {
    createMainWindow();
    mainWindow.loadURL(FLASK_URL).then(() => mainWindow.show());
  }
});
