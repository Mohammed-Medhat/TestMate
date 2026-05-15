/**
 * Wrapper that clears ELECTRON_RUN_AS_NODE before starting electron-vite.
 * This is needed when launching from inside another Electron app (VS Code,
 * Claude Code) which inherits that env var and makes the Electron binary
 * run as plain Node.js instead of the full Electron runtime.
 */
import { spawn } from 'child_process'

delete process.env.ELECTRON_RUN_AS_NODE

const child = spawn(
  'npx',
  ['electron-vite', 'dev'],
  { stdio: 'inherit', env: process.env, shell: true }
)

child.on('exit', (code) => process.exit(code ?? 0))
