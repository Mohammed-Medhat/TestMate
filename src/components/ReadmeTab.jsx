import React, { useState, useRef } from 'react'
import { Upload, FileText, X, Play, Check, Loader2, AlertTriangle, Link } from 'lucide-react'

const EXTRACT_STAGES = [
  { id: 'fetch', label: 'Fetching' },
  { id: 'clean', label: 'Cleaning' },
  { id: 'ai', label: 'AI Processing' },
]

function StepIcon({ status, index }) {
  if (status === 'completed') {
    return (
      <div className="flex h-9 w-9 items-center justify-center rounded-full bg-emerald-500 shadow-[0_0_12px_rgba(16,185,129,0.4)]">
        <Check size={16} className="text-white" strokeWidth={3} />
      </div>
    )
  }
  if (status === 'active') {
    return (
      <div className="flex h-9 w-9 items-center justify-center rounded-full border-2 border-emerald-400 bg-emerald-500/10">
        <Loader2 size={15} className="animate-spin text-emerald-400" />
      </div>
    )
  }
  return (
    <div className="flex h-9 w-9 items-center justify-center rounded-full border-2 border-zinc-700 bg-zinc-900">
      <span className="text-xs font-bold text-zinc-500">{index + 1}</span>
    </div>
  )
}

function FileCard({ file, onRemove }) {
  const ext = file.name.split('.').pop().toUpperCase()
  return (
    <div
      onClick={(e) => e.stopPropagation()}
      className="flex w-full max-w-sm items-center gap-3 rounded-xl border border-zinc-700 bg-zinc-800 px-4 py-3"
    >
      <div className="flex h-10 w-10 flex-none items-center justify-center rounded-lg bg-zinc-700">
        <FileText size={20} className="text-emerald-400" />
      </div>
      <div className="min-w-0 flex-1">
        <p className="truncate text-sm font-medium text-zinc-100">{file.name}</p>
        <p className="text-xs text-zinc-500">{ext} file</p>
      </div>
      <button
        onClick={(e) => { e.stopPropagation(); onRemove() }}
        className="flex-none rounded-md p-1 text-zinc-500 transition-colors hover:bg-zinc-700 hover:text-red-400"
      >
        <X size={14} />
      </button>
    </div>
  )
}

function TypeBadge({ type }) {
  const lower = (type || '').toLowerCase()
  if (lower === 'positive') {
    return (
      <span className="inline-block rounded px-2 py-0.5 text-[10px] font-bold bg-emerald-500/15 text-emerald-400 border border-emerald-500/30">
        Positive
      </span>
    )
  }
  if (lower === 'negative') {
    return (
      <span className="inline-block rounded px-2 py-0.5 text-[10px] font-bold bg-red-500/15 text-red-400 border border-red-500/30">
        Negative
      </span>
    )
  }
  if (lower === 'boundary') {
    return (
      <span className="inline-block rounded px-2 py-0.5 text-[10px] font-bold bg-purple-500/15 text-purple-400 border border-purple-500/30">
        Boundary
      </span>
    )
  }
  return (
    <span className="inline-block rounded px-2 py-0.5 text-[10px] font-bold bg-yellow-500/15 text-yellow-400 border border-yellow-500/30">
      Edge Case
    </span>
  )
}

function PriorityBadge({ priority }) {
  const p = (priority || 'medium').toLowerCase()
  const styles = {
    high: 'bg-red-500/10 text-red-400 border-red-500/30',
    medium: 'bg-zinc-700/60 text-zinc-400 border-zinc-600',
    low: 'bg-zinc-800 text-zinc-500 border-zinc-700',
  }
  return (
    <span className={`inline-block rounded px-2 py-0.5 text-[10px] font-bold border ${styles[p] || styles.medium}`}>
      {p.charAt(0).toUpperCase() + p.slice(1)}
    </span>
  )
}

function ConfidenceBar({ value }) {
  const pct = Math.round((value || 0) * 100)
  return (
    <div className="flex items-center gap-2 min-w-[80px]">
      <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-zinc-800">
        <div className="h-full rounded-full bg-emerald-500 transition-all" style={{ width: `${pct}%` }} />
      </div>
      <span className="w-8 text-right font-mono text-xs text-zinc-400">{pct}%</span>
    </div>
  )
}

function InfoRow({ label, value, confidence }) {
  return (
    <div className="flex flex-col gap-0.5">
      <span className="text-[10px] font-semibold uppercase tracking-wider text-zinc-500">{label}</span>
      <span className="text-xs text-zinc-200">{value || <span className="text-zinc-600 italic">—</span>}</span>
      {confidence != null && <ConfidenceBar value={confidence} />}
    </div>
  )
}

function delay(ms) {
  return new Promise((r) => setTimeout(r, ms))
}

export default function ReadmeTab() {
  const [mode, setMode] = useState('file') // 'file' | 'url'

  // file mode state
  const [file, setFile] = useState(null)
  const [fileContent, setFileContent] = useState(null)
  const [dragging, setDragging] = useState(false)

  // url mode state
  const [repoUrl, setRepoUrl] = useState('')

  // shared state
  const [stageStates, setStageStates] = useState({})
  const [running, setRunning] = useState(false)
  const [extractionResult, setExtractionResult] = useState(null)
  const [testCases, setTestCases] = useState(null)
  const [error, setError] = useState(null)
  const [generatingTests, setGeneratingTests] = useState(false)
  const [testError, setTestError] = useState(null)

  const inputRef = useRef()

  function resetState() {
    setStageStates({})
    setExtractionResult(null)
    setTestCases(null)
    setError(null)
    setTestError(null)
  }

  function switchMode(m) {
    setMode(m)
    setFile(null)
    setFileContent(null)
    setRepoUrl('')
    resetState()
  }

  function clearFile() {
    setFile(null)
    setFileContent(null)
    setRunning(false)
    resetState()
  }

  async function openBrowse() {
    if (window.electronAPI) {
      const result = await window.electronAPI.openFileDialog({ readme: true })
      if (result.canceled || !result.filePaths?.length) return
      const filePath = result.filePaths[0]
      const name = filePath.replace(/\\/g, '/').split('/').pop()
      setFile({ path: filePath, name })
      resetState()
      const fr = await window.electronAPI.readTextFile(filePath)
      if (fr.success) {
        setFileContent(fr.data)
      } else {
        setError(`Could not read file: ${fr.error}`)
      }
    } else {
      inputRef.current?.click()
    }
  }

  function onDrop(e) {
    e.preventDefault()
    setDragging(false)
    const f = e.dataTransfer.files[0]
    if (!f) return
    setFile({ path: f.path || null, name: f.name })
    resetState()
    const reader = new FileReader()
    reader.onload = (ev) => setFileContent(ev.target.result)
    reader.readAsText(f)
  }

  function getStatus(id) {
    return stageStates[id] ?? 'pending'
  }

  const prevCompleted = (i) => i > 0 && getStatus(EXTRACT_STAGES[i - 1].id) === 'completed'

  async function handleExtract() {
    const isUrl = mode === 'url'
    const canRun = isUrl ? repoUrl.trim() : (file && fileContent)
    if (!canRun || running) return

    setRunning(true)
    setExtractionResult(null)
    setTestCases(null)
    setError(null)
    setTestError(null)
    setStageStates({})

    // Animate stages
    ;(async () => {
      setStageStates({ fetch: 'active' })
      await delay(600)
      setStageStates({ fetch: 'completed', clean: 'active' })
      await delay(500)
      setStageStates({ fetch: 'completed', clean: 'completed', ai: 'active' })
    })()

    try {
      let result
      if (isUrl) {
        const res = window.electronAPI
          ? await window.electronAPI.readmeExtractRepo(repoUrl.trim())
          : await fetchDirect('/api/v1/extract_features', { repo_name: repoUrl.trim() })
        if (!res.success) throw new Error(res.error)
        result = res.data
      } else {
        const res = window.electronAPI
          ? await window.electronAPI.readmeExtract(fileContent)
          : await fetchDirect('/api/v1/extract_local', { content: fileContent })
        if (!res.success) throw new Error(res.error)
        result = res.data
      }
      setStageStates({ fetch: 'completed', clean: 'completed', ai: 'completed' })
      setExtractionResult(result)
    } catch (err) {
      setError(err.message)
      setStageStates({})
    } finally {
      setRunning(false)
    }
  }

  async function handleGenerateTests() {
    if (!extractionResult || generatingTests) return
    const isUrl = mode === 'url'
    if (!isUrl && !fileContent) return

    setGeneratingTests(true)
    setTestCases(null)
    setTestError(null)

    try {
      let result
      if (isUrl) {
        const res = window.electronAPI
          ? await window.electronAPI.readmeGenerateTestsRepo(repoUrl.trim(), extractionResult.extraction_id)
          : await fetchDirect('/api/v1/scenarios/generate_scenarios', {
              repo_name: repoUrl.trim(),
              extraction_id: extractionResult.extraction_id,
            })
        if (!res.success) throw new Error(res.error)
        result = res.data
      } else {
        const res = window.electronAPI
          ? await window.electronAPI.readmeGenerateTests(fileContent, extractionResult.extraction_id)
          : await fetchDirect('/api/v1/scenarios/generate_local', {
              content: fileContent,
              extraction_id: extractionResult.extraction_id,
            })
        if (!res.success) throw new Error(res.error)
        result = res.data
      }
      setTestCases(result.test_scenarios || [])
    } catch (err) {
      setTestError(err.message)
    } finally {
      setGeneratingTests(false)
    }
  }

  const features = extractionResult?.features || {}
  const confidence = extractionResult?.confidence || {}
  const canExtract = mode === 'url' ? repoUrl.trim().length > 0 : (file && fileContent)

  return (
    <div className="mx-auto flex max-w-2xl flex-col gap-6">
      {/* Mode toggle */}
      <div className="flex rounded-lg border border-zinc-800 bg-zinc-900 p-1">
        <ModeButton active={mode === 'file'} onClick={() => switchMode('file')} icon={<Upload size={13} />}>
          Upload File
        </ModeButton>
        <ModeButton active={mode === 'url'} onClick={() => switchMode('url')} icon={<Link size={13} />}>
          GitHub URL
        </ModeButton>
      </div>

      {/* Input area */}
      {mode === 'file' ? (
        <>
          <div
            onDrop={onDrop}
            onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
            onDragLeave={() => setDragging(false)}
            onClick={() => !file && openBrowse()}
            className={`flex min-h-56 cursor-pointer flex-col items-center justify-center gap-5 rounded-2xl border-2 border-dashed py-12 transition-all duration-200 ${
              dragging
                ? 'border-emerald-400 bg-emerald-500/5 scale-[1.01]'
                : file
                ? 'cursor-default border-zinc-700 bg-zinc-900/40'
                : 'border-zinc-700 hover:border-emerald-500/60 hover:bg-zinc-900/40'
            }`}
          >
            {file ? (
              <FileCard file={file} onRemove={clearFile} />
            ) : (
              <>
                <div className="flex h-16 w-16 items-center justify-center rounded-full bg-zinc-800 ring-1 ring-zinc-700">
                  <Upload size={26} className="text-emerald-400" />
                </div>
                <div className="flex flex-col items-center gap-1.5 text-center">
                  <p className="text-sm font-semibold text-zinc-200">
                    Drag &amp; drop your README file here
                  </p>
                  <p className="text-xs text-zinc-500">Supports .md and .txt files</p>
                </div>
                <button
                  onClick={(e) => { e.stopPropagation(); openBrowse() }}
                  className="rounded-lg bg-emerald-500 px-5 py-2 text-sm font-semibold text-white transition-colors hover:bg-emerald-400 active:bg-emerald-600"
                >
                  Browse Files
                </button>
              </>
            )}
          </div>
          <input
            ref={inputRef}
            type="file"
            accept=".md,.txt"
            className="hidden"
            onChange={(e) => {
              const f = e.target.files[0]
              if (!f) return
              setFile({ path: f.path || null, name: f.name })
              resetState()
              const reader = new FileReader()
              reader.onload = (ev) => setFileContent(ev.target.result)
              reader.readAsText(f)
            }}
          />
        </>
      ) : (
        <div className="flex flex-col gap-3 rounded-2xl border border-zinc-800 bg-zinc-900/40 p-6">
          <div className="flex items-center gap-2">
            <Link size={15} className="flex-none text-emerald-400" />
            <span className="text-sm font-semibold text-zinc-200">GitHub Repository URL</span>
          </div>
          <p className="text-xs text-zinc-500">
            Paste a full GitHub URL (e.g. <span className="font-mono text-zinc-400">https://github.com/owner/repo</span>) or a short form like <span className="font-mono text-zinc-400">owner/repo</span>
          </p>
          <input
            type="text"
            value={repoUrl}
            onChange={(e) => { setRepoUrl(e.target.value); resetState() }}
            placeholder="https://github.com/owner/repo"
            className="w-full rounded-lg border border-zinc-700 bg-zinc-800 px-4 py-2.5 text-sm text-zinc-100 placeholder-zinc-600 outline-none transition-colors focus:border-emerald-500/60 focus:ring-1 focus:ring-emerald-500/30"
          />
          {repoUrl.trim() && (
            <div className="flex items-center gap-1.5 text-xs text-zinc-500">
              <Check size={11} className="text-emerald-400" />
              Will fetch README from GitHub and run extraction
            </div>
          )}
        </div>
      )}

      {/* MongoDB warning banner */}
      <div className="flex items-center gap-2 rounded-lg border border-yellow-500/30 bg-yellow-500/10 px-4 py-2.5 text-xs text-yellow-400">
        <AlertTriangle size={13} className="flex-none" />
        MongoDB unavailable — feedback loop disabled
      </div>

      {/* Extract Features button */}
      <button
        onClick={handleExtract}
        disabled={!canExtract || running}
        className={`flex items-center justify-center gap-2 rounded-xl py-4 text-sm font-bold tracking-wide transition-all ${
          canExtract && !running
            ? 'bg-emerald-500 text-white shadow-[0_0_20px_rgba(16,185,129,0.25)] hover:bg-emerald-400'
            : 'cursor-not-allowed bg-zinc-800 text-zinc-600'
        }`}
      >
        {running ? <Loader2 size={16} className="animate-spin" /> : extractionResult ? <Check size={16} /> : <Play size={16} />}
        {running ? 'Extracting…' : extractionResult ? 'Re-extract Features' : 'Extract Features'}
      </button>

      {/* Stepper */}
      <div className="rounded-xl border border-zinc-800 bg-zinc-900 p-6">
        <p className="mb-5 text-[10px] font-semibold uppercase tracking-widest text-zinc-500">
          Extraction Stages
        </p>
        <div className="flex items-start">
          {EXTRACT_STAGES.map((stage, i) => (
            <React.Fragment key={stage.id}>
              {i > 0 && (
                <div className="flex flex-1 items-start pt-[17px] px-2">
                  <div className={`h-[2px] w-full rounded-full transition-colors duration-500 ${prevCompleted(i) ? 'bg-emerald-500' : 'bg-zinc-700'}`} />
                </div>
              )}
              <div className="flex flex-col items-center gap-2">
                <StepIcon status={getStatus(stage.id)} index={i} />
                <span className={`max-w-[72px] text-center text-[11px] font-medium leading-tight transition-colors ${getStatus(stage.id) === 'pending' ? 'text-zinc-600' : 'text-zinc-200'}`}>
                  {stage.label}
                </span>
              </div>
            </React.Fragment>
          ))}
        </div>
      </div>

      {/* Error box */}
      {error && (
        <div className="rounded-lg border border-red-500/40 bg-red-500/10 p-4 text-xs text-red-400">
          <p className="mb-1 font-semibold">Extraction failed</p>
          <pre className="max-h-32 overflow-auto whitespace-pre-wrap opacity-80">{error}</pre>
        </div>
      )}

      {/* Results card */}
      {extractionResult && !error && (
        <div className="rounded-xl border border-zinc-800 bg-zinc-800 p-5 flex flex-col gap-4">
          <p className="text-[10px] font-semibold uppercase tracking-widest text-zinc-500">
            Extracted Features
          </p>

          <div className="grid grid-cols-2 gap-x-6 gap-y-3 text-xs">
            <InfoRow label="Project Name" value={features.project_name} confidence={confidence.project_name} />
            <InfoRow label="License" value={features.license_type} confidence={confidence.license_type} />
            <div className="col-span-2">
              <InfoRow label="Description" value={features.description} confidence={confidence.description} />
            </div>
          </div>

          {features.tech_stack?.length > 0 && (
            <div>
              <p className="mb-1.5 text-[10px] font-semibold text-zinc-500 uppercase tracking-wider">Tech Stack</p>
              <div className="flex flex-wrap gap-1.5">
                {features.tech_stack.map((t) => (
                  <span key={t} className="rounded-full bg-zinc-700 px-2.5 py-0.5 text-[11px] font-medium text-zinc-300">
                    {t}
                  </span>
                ))}
              </div>
            </div>
          )}

          {features.installation_commands?.length > 0 && (
            <div>
              <p className="mb-1.5 text-[10px] font-semibold text-zinc-500 uppercase tracking-wider">Dependencies / Install</p>
              <div className="rounded-lg bg-zinc-900 p-3 font-mono text-[11px] text-zinc-300">
                {features.installation_commands.map((cmd, i) => (
                  <div key={i} className="leading-relaxed">$ {cmd}</div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Generate Test Cases button */}
      {extractionResult && !error && (
        <button
          onClick={handleGenerateTests}
          disabled={generatingTests}
          className={`flex items-center justify-center gap-2 rounded-xl py-4 text-sm font-bold tracking-wide transition-all ${
            !generatingTests
              ? 'bg-emerald-500 text-white shadow-[0_0_20px_rgba(16,185,129,0.25)] hover:bg-emerald-400'
              : 'cursor-not-allowed bg-zinc-800 text-zinc-600'
          }`}
        >
          {generatingTests ? <Loader2 size={16} className="animate-spin" /> : <Play size={16} />}
          {generatingTests ? 'Generating Test Cases…' : 'Generate Test Cases'}
        </button>
      )}

      {/* Test generation error */}
      {testError && (
        <div className="rounded-lg border border-red-500/40 bg-red-500/10 p-4 text-xs text-red-400">
          <p className="mb-1 font-semibold">Test generation failed</p>
          <pre className="max-h-32 overflow-auto whitespace-pre-wrap opacity-80">{testError}</pre>
        </div>
      )}

      {/* Test cases table */}
      {testCases && testCases.length > 0 && (
        <div className="flex flex-col gap-3">
          <p className="text-[10px] font-semibold uppercase tracking-widest text-zinc-500">
            Generated Test Cases — {testCases.length} scenarios
          </p>
          <div className="overflow-hidden rounded-lg border border-zinc-800">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-zinc-800 bg-zinc-900">
                  <th className="py-2.5 pl-4 pr-2 text-left text-[10px] font-semibold uppercase tracking-wider text-zinc-500 w-8">#</th>
                  <th className="px-3 py-2.5 text-left text-[10px] font-semibold uppercase tracking-wider text-zinc-500 w-20">Type</th>
                  <th className="px-3 py-2.5 text-left text-[10px] font-semibold uppercase tracking-wider text-zinc-500">Description</th>
                  <th className="px-3 py-2.5 text-left text-[10px] font-semibold uppercase tracking-wider text-zinc-500">Expected Output</th>
                  <th className="px-3 py-2.5 pr-4 text-left text-[10px] font-semibold uppercase tracking-wider text-zinc-500 w-16">Priority</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-zinc-800/50 bg-zinc-950">
                {testCases.map((tc, i) => (
                  <tr key={i} className="transition-colors hover:bg-zinc-900/60">
                    <td className="py-3 pl-4 pr-2 font-mono text-xs text-zinc-600">{i + 1}</td>
                    <td className="px-3 py-3"><TypeBadge type={tc.type} /></td>
                    <td className="px-3 py-3 text-xs leading-relaxed text-zinc-300">{tc.description}</td>
                    <td className="px-3 py-3 text-xs leading-relaxed text-zinc-400">{tc.expected_result}</td>
                    <td className="px-3 py-3 pr-4"><PriorityBadge priority={tc.priority} /></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  )
}

function ModeButton({ active, onClick, icon, children }) {
  return (
    <button
      onClick={onClick}
      className={`flex flex-1 items-center justify-center gap-2 rounded-md py-2 text-xs font-semibold transition-colors ${
        active
          ? 'bg-emerald-500/15 text-emerald-400 border border-emerald-500/30'
          : 'text-zinc-500 hover:text-zinc-300'
      }`}
    >
      {icon}
      {children}
    </button>
  )
}

async function fetchDirect(path, body) {
  const res = await fetch(`http://127.0.0.1:8001${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    return { success: false, error: err.detail || `HTTP ${res.status}` }
  }
  return { success: true, data: await res.json() }
}
