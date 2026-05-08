import React, { useState, useRef } from 'react'
import { Upload, FileText, X, Play, Check, Loader2 } from 'lucide-react'

const STAGES = [
  { id: 'extract', label: 'Extract Text' },
  { id: 'segment', label: 'Segment Sentences' },
  { id: 'align', label: 'Align Dataset' },
  { id: 'output', label: 'Output Results' },
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
  const sizeLabel =
    file.size != null
      ? file.size >= 1_048_576
        ? `${(file.size / 1_048_576).toFixed(1)} MB`
        : `${Math.round(file.size / 1024)} KB`
      : ext

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
        <p className="text-xs text-zinc-500">{sizeLabel} · {ext}</p>
      </div>
      <button
        onClick={(e) => {
          e.stopPropagation()
          onRemove()
        }}
        className="flex-none rounded-md p-1 text-zinc-500 transition-colors hover:bg-zinc-700 hover:text-red-400"
      >
        <X size={14} />
      </button>
    </div>
  )
}

// file state shape: { path: string, name: string, size: number|null }

export default function UploadTab({ onViewRequirements }) {
  const [file, setFile] = useState(null)
  const [dragging, setDragging] = useState(false)
  const [stageStates, setStageStates] = useState({})
  const [running, setRunning] = useState(false)
  const [done, setDone] = useState(false)
  const [error, setError] = useState(null)
  const [pipelineResult, setPipelineResult] = useState(null)
  const inputRef = useRef()

  function resetRunState() {
    setStageStates({})
    setDone(false)
    setError(null)
    setPipelineResult(null)
  }

  function setFileFromNative(f) {
    if (!f) return
    setFile({ path: f.path || null, name: f.name, size: f.size })
    resetRunState()
  }

  function setFileFromPath(p) {
    const name = p.replace(/\\/g, '/').split('/').pop()
    setFile({ path: p, name, size: null })
    resetRunState()
  }

  function clearFile() {
    setFile(null)
    setRunning(false)
    resetRunState()
  }

  async function openBrowse() {
    if (window.electronAPI) {
      const result = await window.electronAPI.openFileDialog()
      if (result.canceled || !result.filePaths?.length) return
      setFileFromPath(result.filePaths[0])
    } else {
      inputRef.current?.click()
    }
  }

  function onDrop(e) {
    e.preventDefault()
    setDragging(false)
    const f = e.dataTransfer.files[0]
    if (f) setFileFromNative(f)
  }

  function getStatus(id) {
    return stageStates[id] ?? 'pending'
  }

  async function handleRun() {
    if (!file || running) return
    setRunning(true)
    setDone(false)
    setStageStates({})
    setError(null)
    setPipelineResult(null)

    if (window.electronAPI) {
      const unsubOutput = window.electronAPI.onPipelineOutput(({ type, data }) => {
        const line = data.trim().toLowerCase()
        const hitIdx = STAGES.findIndex((s) =>
          line.includes(s.label.toLowerCase().split(' ')[0])
        )
        if (hitIdx !== -1) {
          setStageStates((prev) => {
            const updated = { ...prev, [STAGES[hitIdx].id]: 'active' }
            if (hitIdx > 0 && prev[STAGES[hitIdx - 1].id] === 'active') {
              updated[STAGES[hitIdx - 1].id] = 'completed'
            }
            return updated
          })
        }
      })

      try {
        const result = await window.electronAPI.runPipeline({ inputFile: file.path })
        setStageStates(STAGES.reduce((acc, s) => ({ ...acc, [s.id]: 'completed' }), {}))
        const fileResult = await window.electronAPI.readFile(result.outputPath)
        if (!fileResult.success) throw new Error(fileResult.error)
        const requirements = fileResult.data
          .filter((r) => r.label === 1)
          .sort((a, b) => b.score - a.score)
        setPipelineResult({ rows: requirements, total: requirements.length })
        setDone(true)
      } catch (err) {
        setError(err.message)
      } finally {
        unsubOutput()
        setRunning(false)
      }
    } else {
      // simulate (no Electron API)
      for (const stage of STAGES) {
        setStageStates((p) => ({ ...p, [stage.id]: 'active' }))
        await new Promise((r) => setTimeout(r, 1400))
        setStageStates((p) => ({ ...p, [stage.id]: 'completed' }))
      }
      setDone(true)
      setRunning(false)
    }
  }

  const prevCompleted = (i) => i > 0 && getStatus(STAGES[i - 1].id) === 'completed'

  return (
    <div className="mx-auto flex max-w-2xl flex-col gap-6">
      {/* ── Drop zone ── */}
      <div
        onDrop={onDrop}
        onDragOver={(e) => {
          e.preventDefault()
          setDragging(true)
        }}
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
                Drag &amp; drop your SRS document here
              </p>
              <p className="text-xs text-zinc-500">Supports PDF and DOCX files</p>
            </div>
            <button
              onClick={(e) => {
                e.stopPropagation()
                openBrowse()
              }}
              className="rounded-lg bg-emerald-500 px-5 py-2 text-sm font-semibold text-white transition-colors hover:bg-emerald-400 active:bg-emerald-600"
            >
              Browse Files
            </button>
          </>
        )}
      </div>

      {/* fallback file input for non-Electron environment */}
      <input
        ref={inputRef}
        type="file"
        accept=".pdf,.docx"
        className="hidden"
        onChange={(e) => setFileFromNative(e.target.files[0])}
      />

      {/* ── Run Pipeline button ── */}
      <button
        onClick={handleRun}
        disabled={!file || running}
        className={`flex items-center justify-center gap-2 rounded-xl py-4 text-sm font-bold tracking-wide transition-all ${
          file && !running
            ? 'bg-emerald-500 text-white shadow-[0_0_20px_rgba(16,185,129,0.25)] hover:bg-emerald-400'
            : 'cursor-not-allowed bg-zinc-800 text-zinc-600'
        }`}
      >
        {running ? (
          <Loader2 size={16} className="animate-spin" />
        ) : done ? (
          <Check size={16} />
        ) : (
          <Play size={16} />
        )}
        {running ? 'Pipeline Running…' : done ? 'Re-run Pipeline' : 'Run Pipeline'}
      </button>

      {/* ── Pipeline stepper ── */}
      <div className="rounded-xl border border-zinc-800 bg-zinc-900 p-6">
        <p className="mb-5 text-[10px] font-semibold uppercase tracking-widest text-zinc-500">
          Pipeline Stages
        </p>
        <div className="flex items-start">
          {STAGES.map((stage, i) => (
            <React.Fragment key={stage.id}>
              {i > 0 && (
                <div className="flex flex-1 items-start pt-[17px] px-2">
                  <div
                    className={`h-[2px] w-full rounded-full transition-colors duration-500 ${
                      prevCompleted(i) ? 'bg-emerald-500' : 'bg-zinc-700'
                    }`}
                  />
                </div>
              )}
              <div className="flex flex-col items-center gap-2">
                <StepIcon status={getStatus(stage.id)} index={i} />
                <span
                  className={`max-w-[72px] text-center text-[11px] font-medium leading-tight transition-colors ${
                    getStatus(stage.id) === 'pending' ? 'text-zinc-600' : 'text-zinc-200'
                  }`}
                >
                  {stage.label}
                </span>
              </div>
            </React.Fragment>
          ))}
        </div>
      </div>

      {/* ── Success: View Requirements button ── */}
      {done && !error && pipelineResult && (
        <button
          onClick={() => onViewRequirements(pipelineResult)}
          className="w-full rounded-xl bg-emerald-500 py-3 text-sm font-bold text-white shadow-[0_0_20px_rgba(16,185,129,0.25)] transition-colors hover:bg-emerald-400"
        >
          View Extracted Requirements
        </button>
      )}

      {/* ── Failure: error box ── */}
      {error && (
        <div className="rounded-lg border border-red-500/40 bg-red-500/10 p-4 text-xs text-red-400">
          <p className="mb-1 font-semibold">Pipeline failed</p>
          <pre className="max-h-32 overflow-auto whitespace-pre-wrap opacity-80">{error}</pre>
        </div>
      )}
    </div>
  )
}
