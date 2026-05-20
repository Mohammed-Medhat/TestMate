import { Home, Settings, Search, Rocket, FileCode, Link2, FileText, Bot, Upload, Wrench } from 'lucide-react'
import type { Mode, PartAMode, FileInfo, RunSettings } from '../types'

interface Props {
  mode: Mode
  setMode: (m: Mode) => void
  onHome: () => void
  onSettings: () => void
  settings: RunSettings

  // Part B
  repoUrl: string; setRepoUrl: (v: string) => void
  branch: string; setBranch: (v: string) => void
  files: FileInfo[]; selected: Set<number>; toggleFile: (i: number) => void
  discover: () => void; discovering: boolean; repoName: string

  // Part A
  partAMode: PartAMode; setPartAMode: (m: PartAMode) => void
  srsFile: File | null; setSrsFile: (f: File | null) => void
  readme: string; setReadme: (v: string) => void
  repoNameA: string; setRepoNameA: (v: string) => void
  problems: string; setProblems: (v: string) => void
  expected: string; setExpected: (v: string) => void
  edgeCases: string; setEdgeCases: (v: string) => void
  threshold: number; setThreshold: (n: number) => void

  // Combined
  combSrsFile: File | null; setCombSrsFile: (f: File | null) => void
  combReadme: string; setCombReadme: (v: string) => void
  combTargetFile: string; setCombTargetFile: (v: string) => void
  combImportPath: string; setCombImportPath: (v: string) => void

  // Part C
  partcSourceIdx: number | null; setPartcSourceIdx: (i: number | null) => void
  partcTestIdx: number | null; setPartcTestIdx: (i: number | null) => void

  // Shared
  status: string
  startRun: () => void
}

const MODE_ICONS: { mode: Mode; icon: React.ReactNode; label: string }[] = [
  { mode: 'combined', icon: <Link2 size={18} />,    label: 'Full Pipeline' },
  { mode: 'parta',    icon: <FileText size={18} />, label: 'Part A Only' },
  { mode: 'partb',    icon: <Bot size={18} />,      label: 'Part B Only' },
  { mode: 'partc',    icon: <Wrench size={18} />,   label: 'Bug Fixer' },
]

const PART_A_MODES: { id: PartAMode; label: string }[] = [
  { id: 'srs',    label: 'SRS Pipeline' },
  { id: 'readme', label: 'README Extractor' },
  { id: 'both',   label: 'Both' },
]

export default function LeftSidebar(props: Props) {
  const { mode, setMode, onHome, onSettings, settings, status, startRun } = props

  const activeSettingsCount = [settings.docker, settings.deepScan, settings.intense, settings.hitl, settings.planMode, settings.autoRepair].filter(Boolean).length

  return (
    <div className="flex h-full">
      {/* ── Icon rail (48px) ─────────────────────────────────────────── */}
      <div className="w-12 bg-zinc-950 border-r border-zinc-800 flex flex-col items-center py-3 gap-2">
        {/* Home */}
        <button onClick={onHome}
          className="p-2 rounded hover:bg-zinc-800 text-zinc-400 hover:text-zinc-100 transition-colors mb-2"
          title="Back to Home">
          <Home size={18} />
        </button>

        {/* Mode switchers */}
        {MODE_ICONS.map(({ mode: m, icon, label }) => (
          <button key={m} onClick={() => setMode(m)} title={label}
            className={`p-2 rounded transition-colors ${
              mode === m
                ? 'bg-emerald-500/15 text-emerald-400 ring-1 ring-emerald-500/30'
                : 'text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200'
            }`}>
            {icon}
          </button>
        ))}

        {/* Settings at bottom */}
        <div className="mt-auto">
          <button onClick={onSettings} title="Settings"
            className="p-2 rounded hover:bg-zinc-800 text-zinc-400 hover:text-emerald-400 transition-colors relative">
            <Settings size={18} />
            {activeSettingsCount > 0 && (
              <span className="absolute -top-0.5 -right-0.5 w-4 h-4 bg-emerald-500 rounded-full text-[9px] font-bold text-white flex items-center justify-center">
                {activeSettingsCount}
              </span>
            )}
          </button>
        </div>
      </div>

      {/* ── Main sidebar (224px) ─────────────────────────────────────── */}
      <div className="w-56 bg-zinc-900 border-r border-zinc-800 flex flex-col">
        {mode === 'combined' && <CombinedSidebar {...props} />}
        {mode === 'parta'    && <PartASidebar {...props} />}
        {mode === 'partb'    && <PartBSidebar {...props} />}
        {mode === 'partc'    && <PartCSidebar {...props} />}
      </div>
    </div>
  )
}

/* ── Combined Sidebar (Part A inputs + Part B-style repo + file picker) ── */
function CombinedSidebar({
  combSrsFile, setCombSrsFile, combReadme, setCombReadme,
  // Reuse Part B's repo/file discovery state for the Combined workflow
  repoUrl, setRepoUrl, branch, setBranch,
  files, selected, toggleFile, discover, discovering, repoName,
  settings, status, startRun,
}: Props) {
  const canRun = files.length > 0 && selected.size > 0 && status !== 'running'

  return (
    <>
      {/* Part A inputs (project-level) */}
      <div className="p-3 border-b border-zinc-800">
        <h2 className="text-xs font-semibold text-zinc-400 uppercase tracking-wider mb-3">Part A Inputs</h2>

        <label className="block text-[11px] text-zinc-500 mb-1">SRS Document (optional)</label>
        <label className="flex items-center gap-2 mb-3 cursor-pointer rounded border border-dashed border-zinc-700 px-2 py-2 text-[11px] text-zinc-500 hover:border-emerald-500/40 hover:text-zinc-300 transition-colors">
          <Upload size={12} />
          {combSrsFile ? combSrsFile.name : 'Choose PDF / DOCX…'}
          <input type="file" accept=".pdf,.doc,.docx" className="hidden"
            onChange={e => setCombSrsFile(e.target.files?.[0] ?? null)} />
        </label>

        <label className="block text-[11px] text-zinc-500 mb-1">README / Description (optional)</label>
        <textarea
          className="w-full rounded border border-zinc-800 bg-zinc-950 px-2 py-1.5 text-[11px] text-zinc-300 placeholder-zinc-600 focus:border-emerald-500 focus:outline-none resize-none"
          rows={3} placeholder="Paste README text…"
          value={combReadme} onChange={e => setCombReadme(e.target.value)} />
      </div>

      {/* Repository (Part B-style) */}
      <div className="p-3 border-b border-zinc-800 space-y-1.5">
        <h2 className="text-xs font-semibold text-zinc-400 uppercase tracking-wider">Repository</h2>
        <input value={repoUrl} onChange={e => setRepoUrl(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && discover()}
          placeholder="GitHub URL or local path"
          className="w-full bg-zinc-950 border border-zinc-800 rounded px-2 py-1.5 text-xs text-zinc-200 placeholder-zinc-600 outline-none focus:border-emerald-500" />
        <input value={branch} onChange={e => setBranch(e.target.value)}
          placeholder="branch (optional)"
          className="w-full bg-zinc-950 border border-zinc-800 rounded px-2 py-1.5 text-xs text-zinc-200 placeholder-zinc-600 outline-none focus:border-emerald-500" />
        <button onClick={discover} disabled={discovering || !repoUrl.trim()}
          className="w-full py-1.5 rounded text-xs font-medium bg-emerald-500 hover:bg-emerald-600 text-white disabled:opacity-50 transition-colors flex items-center justify-center gap-1">
          {discovering ? 'Scanning…' : <><Search size={12} /> Discover Files</>}
        </button>
      </div>

      {/* File picker */}
      <div className="p-3 border-b border-zinc-800 flex items-center justify-between">
        <h2 className="text-sm font-semibold text-zinc-300">{repoName ? `${repoName}/` : 'Files'}</h2>
        {files.length > 0 && (
          <button
            onClick={() => {
              if (selected.size === files.length) {
                for (let i = 0; i < files.length; i++) if (selected.has(i)) toggleFile(i)
              } else {
                for (let i = 0; i < files.length; i++) if (!selected.has(i)) toggleFile(i)
              }
            }}
            className="text-[0.65rem] text-emerald-400 hover:text-emerald-300 hover:underline">
            {selected.size === files.length ? 'Deselect All' : 'Select All'}
          </button>
        )}
      </div>
      <div className="flex-1 overflow-y-auto p-2">
        {files.length === 0
          ? <p className="text-xs text-zinc-600 px-2 py-4 text-center">Discover a repo or paste a local path</p>
          : files.map((file, i) => {
            const name = file.path.split('/').pop() ?? file.path
            const checked = selected.has(i)
            return (
              <div key={i} onClick={() => toggleFile(i)}
                className="flex items-center gap-2 px-2 py-1.5 rounded hover:bg-zinc-800 cursor-pointer group">
                <input type="checkbox" checked={checked} onChange={() => toggleFile(i)}
                  onClick={e => e.stopPropagation()} className="accent-emerald-500 w-3 h-3" />
                <FileCode size={14} className="text-emerald-400" />
                <span className={`text-sm ${checked ? 'text-zinc-200' : 'text-zinc-400'} group-hover:text-zinc-200 truncate`}>{name}</span>
                <span className="ml-auto text-[0.6rem] text-zinc-600">{file.functions}fn</span>
              </div>
            )
          })
        }
      </div>

      <ActivePills settings={settings} />

      {files.length > 0 && (
        <div className="p-3 border-t border-zinc-800">
          <button onClick={startRun} disabled={!canRun}
            className="w-full py-2 rounded text-sm font-semibold bg-emerald-500 hover:bg-emerald-600 text-white disabled:opacity-40 transition-colors flex items-center justify-center gap-1.5">
            {status === 'running' ? 'Running...' : <><Rocket size={14} /> Run Full Pipeline</>}
          </button>
          <p className="text-[0.6rem] text-zinc-500 text-center mt-1">
            {selected.size}/{files.length} selected · {settings.maxRetries} retries
          </p>
        </div>
      )}
    </>
  )
}

/* ── Part A Sidebar ──────────────────────────────────────────── */
function PartASidebar({ partAMode, setPartAMode, srsFile, setSrsFile, readme, setReadme,
  repoNameA, setRepoNameA, problems, setProblems, expected, setExpected,
  edgeCases, setEdgeCases, threshold, setThreshold, status, startRun }: Props) {

  return (
    <>
      <div className="p-3 border-b border-zinc-800 flex-1 overflow-y-auto">
        <h2 className="text-xs font-semibold text-zinc-400 uppercase tracking-wider mb-2">Mode</h2>

        {/* Segmented pill */}
        <div className="flex gap-0.5 rounded-lg border border-zinc-800 bg-zinc-950 p-0.5 mb-3">
          {PART_A_MODES.map(m => (
            <button key={m.id} onClick={() => setPartAMode(m.id)}
              className={`flex-1 rounded-md py-1 text-[10px] font-medium transition-colors ${
                partAMode === m.id
                  ? 'bg-emerald-500/15 text-emerald-400 border border-emerald-500/30'
                  : 'text-zinc-500 hover:text-zinc-300'
              }`}>{m.label}</button>
          ))}
        </div>

        {/* SRS inputs */}
        {(partAMode === 'srs' || partAMode === 'both') && (
          <>
            <label className="block text-[11px] text-zinc-500 mb-1">SRS Document</label>
            <label className="flex items-center gap-2 mb-2 cursor-pointer rounded border border-dashed border-zinc-700 px-2 py-2 text-[11px] text-zinc-500 hover:border-emerald-500/40 hover:text-zinc-300 transition-colors">
              <Upload size={12} />
              {srsFile ? srsFile.name : 'Choose PDF / DOCX…'}
              <input type="file" accept=".pdf,.doc,.docx" className="hidden"
                onChange={e => setSrsFile(e.target.files?.[0] ?? null)} />
            </label>
            <div className="flex items-center gap-2 mb-3">
              <label className="text-[11px] text-zinc-500 shrink-0">Threshold</label>
              <input type="range" min={0.5} max={1} step={0.05} value={threshold}
                onChange={e => setThreshold(Number(e.target.value))}
                className="flex-1 accent-emerald-500" />
              <span className="text-[11px] text-zinc-400 w-7 text-right">{threshold}</span>
            </div>
          </>
        )}

        {/* README inputs */}
        {(partAMode === 'readme' || partAMode === 'both') && (
          <>
            <label className="block text-[11px] text-zinc-500 mb-1">README / Description</label>
            <textarea className="w-full mb-2 rounded border border-zinc-800 bg-zinc-950 px-2 py-1.5 text-[11px] text-zinc-300 placeholder-zinc-600 focus:border-emerald-500 focus:outline-none resize-none"
              rows={3} placeholder="Paste README text…" value={readme} onChange={e => setReadme(e.target.value)} />
            <input className="w-full mb-1.5 rounded border border-zinc-800 bg-zinc-950 px-2 py-1.5 text-[11px] text-zinc-300 placeholder-zinc-600 focus:border-emerald-500 focus:outline-none"
              placeholder="Project name (optional)" value={repoNameA} onChange={e => setRepoNameA(e.target.value)} />
            <input className="w-full mb-1.5 rounded border border-zinc-800 bg-zinc-950 px-2 py-1.5 text-[11px] text-zinc-300 placeholder-zinc-600 focus:border-emerald-500 focus:outline-none"
              placeholder="Main problems to address…" value={problems} onChange={e => setProblems(e.target.value)} />
            <input className="w-full mb-1.5 rounded border border-zinc-800 bg-zinc-950 px-2 py-1.5 text-[11px] text-zinc-300 placeholder-zinc-600 focus:border-emerald-500 focus:outline-none"
              placeholder="Expected behaviour…" value={expected} onChange={e => setExpected(e.target.value)} />
            <input className="w-full rounded border border-zinc-800 bg-zinc-950 px-2 py-1.5 text-[11px] text-zinc-300 placeholder-zinc-600 focus:border-emerald-500 focus:outline-none"
              placeholder="Known edge cases…" value={edgeCases} onChange={e => setEdgeCases(e.target.value)} />
          </>
        )}
      </div>

      <RunButton status={status} onClick={startRun} label="Run Part A" />
    </>
  )
}

/* ── Part B Sidebar (verbatim PartB/gui pattern) ─────────────── */
function PartBSidebar({ repoUrl, setRepoUrl, branch, setBranch, files, selected,
  toggleFile, discover, discovering, repoName, settings, status, startRun }: Props) {

  return (
    <>
      <div className="p-3 border-b border-zinc-800 space-y-1.5">
        <h2 className="text-xs font-semibold text-zinc-400 uppercase tracking-wider">Repository</h2>
        <input value={repoUrl} onChange={e => setRepoUrl(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && discover()}
          placeholder="GitHub URL or local path"
          className="w-full bg-zinc-950 border border-zinc-800 rounded px-2 py-1.5 text-xs text-zinc-200 placeholder-zinc-600 outline-none focus:border-emerald-500" />
        <input value={branch} onChange={e => setBranch(e.target.value)}
          placeholder="branch (optional)"
          className="w-full bg-zinc-950 border border-zinc-800 rounded px-2 py-1.5 text-xs text-zinc-200 placeholder-zinc-600 outline-none focus:border-emerald-500" />
        <button onClick={discover} disabled={discovering || !repoUrl.trim()}
          className="w-full py-1.5 rounded text-xs font-medium bg-emerald-500 hover:bg-emerald-600 text-white disabled:opacity-50 transition-colors flex items-center justify-center gap-1">
          {discovering ? 'Scanning...' : <><Search size={12} /> Discover Files</>}
        </button>
      </div>

      <div className="p-3 border-b border-zinc-800 flex items-center justify-between">
        <h2 className="text-sm font-semibold text-zinc-300">{repoName ? `${repoName}/` : 'Projects'}</h2>
        {files.length > 0 && (
          <button
            onClick={() => { files.forEach((_, i) => { if (selected.size < files.length && !selected.has(i)) toggleFile(i); else if (selected.size === files.length && selected.has(i)) toggleFile(i) }) }}
            className="text-[0.65rem] text-emerald-400 hover:text-emerald-300 hover:underline">
            {selected.size === files.length ? 'Deselect All' : 'Select All'}
          </button>
        )}
      </div>

      <div className="flex-1 overflow-y-auto p-2">
        {files.length === 0
          ? <p className="text-xs text-zinc-600 px-2 py-4 text-center">Discover a repo to see files</p>
          : files.map((file, i) => {
            const name = file.path.split('/').pop() ?? file.path
            const checked = selected.has(i)
            return (
              <div key={i} onClick={() => toggleFile(i)}
                className="flex items-center gap-2 px-2 py-1.5 rounded hover:bg-zinc-800 cursor-pointer group">
                <input type="checkbox" checked={checked} onChange={() => toggleFile(i)}
                  onClick={e => e.stopPropagation()} className="accent-emerald-500 w-3 h-3" />
                <FileCode size={14} className="text-emerald-400" />
                <span className={`text-sm ${checked ? 'text-zinc-200' : 'text-zinc-400'} group-hover:text-zinc-200 truncate`}>{name}</span>
                <span className="ml-auto text-[0.6rem] text-zinc-600">{file.functions}fn</span>
              </div>
            )
          })
        }
      </div>

      <ActivePills settings={settings} />

      {files.length > 0 && (
        <div className="p-3 border-t border-zinc-800">
          <button onClick={startRun} disabled={selected.size === 0 || status === 'running'}
            className="w-full py-2 rounded text-sm font-semibold bg-emerald-500 hover:bg-emerald-600 text-white disabled:opacity-40 transition-colors flex items-center justify-center gap-1.5">
            {status === 'running' ? 'Running...' : <><Rocket size={14} /> Generate Tests</>}
          </button>
          <p className="text-[0.6rem] text-zinc-500 text-center mt-1">
            {selected.size}/{files.length} selected · {settings.maxRetries} retries
          </p>
        </div>
      )}
    </>
  )
}

/* ── Part C Sidebar (Bug Fixer) ───────────────────────────────── */
function PartCSidebar({ repoUrl, setRepoUrl, branch, setBranch, files,
  discover, discovering, repoName,
  partcSourceIdx, setPartcSourceIdx, partcTestIdx, setPartcTestIdx,
  status, startRun }: Props) {

  const canRun = partcSourceIdx !== null && partcTestIdx !== null && status !== 'running'
  const sourceFile = partcSourceIdx !== null ? files[partcSourceIdx] : null
  const testFile   = partcTestIdx   !== null ? files[partcTestIdx]   : null

  return (
    <>
      {/* Repo discovery */}
      <div className="p-3 border-b border-zinc-800 space-y-1.5">
        <h2 className="text-xs font-semibold text-zinc-400 uppercase tracking-wider">Repository</h2>
        <input value={repoUrl} onChange={e => setRepoUrl(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && discover()}
          placeholder="GitHub URL or local path"
          className="w-full bg-zinc-950 border border-zinc-800 rounded px-2 py-1.5 text-xs text-zinc-200 placeholder-zinc-600 outline-none focus:border-amber-500" />
        <input value={branch} onChange={e => setBranch(e.target.value)}
          placeholder="branch (optional)"
          className="w-full bg-zinc-950 border border-zinc-800 rounded px-2 py-1.5 text-xs text-zinc-200 placeholder-zinc-600 outline-none focus:border-amber-500" />
        <button onClick={discover} disabled={discovering || !repoUrl.trim()}
          className="w-full py-1.5 rounded text-xs font-medium bg-amber-500 hover:bg-amber-600 text-white disabled:opacity-50 transition-colors flex items-center justify-center gap-1">
          {discovering ? 'Scanning…' : <><Search size={12} /> Discover Files</>}
        </button>
      </div>

      {/* File pickers */}
      <div className="flex-1 overflow-y-auto p-3 space-y-3">
        {files.length === 0 ? (
          <p className="text-xs text-zinc-600 text-center py-4">Discover a repo to pick files</p>
        ) : (
          <>
            <div>
              <h3 className="text-[11px] font-semibold text-zinc-400 mb-1">
                🐛 Buggy Source File
              </h3>
              <div className="space-y-1">
                {files.map((f, i) => {
                  const name = f.path.split('/').pop() ?? f.path
                  const isTest = name.startsWith('test_') || name.endsWith('_test.py')
                  if (isTest) return null
                  return (
                    <button key={i} onClick={() => setPartcSourceIdx(i === partcSourceIdx ? null : i)}
                      className={`w-full flex items-center gap-2 px-2 py-1.5 rounded text-left transition-colors ${
                        partcSourceIdx === i
                          ? 'bg-amber-500/15 border border-amber-500/30 text-amber-300'
                          : 'hover:bg-zinc-800 text-zinc-400 border border-transparent'
                      }`}>
                      <FileCode size={13} className={partcSourceIdx === i ? 'text-amber-400' : 'text-zinc-500'} />
                      <span className="text-xs truncate">{name}</span>
                      <span className="ml-auto text-[0.6rem] text-zinc-600">{f.functions}fn</span>
                    </button>
                  )
                })}
              </div>
            </div>

            <div>
              <h3 className="text-[11px] font-semibold text-zinc-400 mb-1">
                🧪 Test File
              </h3>
              <div className="space-y-1">
                {files.map((f, i) => {
                  const name = f.path.split('/').pop() ?? f.path
                  return (
                    <button key={i} onClick={() => setPartcTestIdx(i === partcTestIdx ? null : i)}
                      className={`w-full flex items-center gap-2 px-2 py-1.5 rounded text-left transition-colors ${
                        partcTestIdx === i
                          ? 'bg-amber-500/15 border border-amber-500/30 text-amber-300'
                          : 'hover:bg-zinc-800 text-zinc-400 border border-transparent'
                      }`}>
                      <FileCode size={13} className={partcTestIdx === i ? 'text-amber-400' : 'text-zinc-500'} />
                      <span className="text-xs truncate">{name}</span>
                    </button>
                  )
                })}
              </div>
            </div>
          </>
        )}
      </div>

      {/* Selection summary */}
      {(sourceFile || testFile) && (
        <div className="px-3 py-2 border-t border-zinc-800 space-y-0.5">
          {sourceFile && <p className="text-[10px] text-amber-400 truncate">🐛 {sourceFile.path.split('/').pop()}</p>}
          {testFile   && <p className="text-[10px] text-amber-400 truncate">🧪 {testFile.path.split('/').pop()}</p>}
        </div>
      )}

      <div className="p-3 border-t border-zinc-800">
        <button onClick={startRun} disabled={!canRun}
          className="w-full py-2 rounded text-sm font-semibold bg-amber-500 hover:bg-amber-600 text-white disabled:opacity-40 transition-colors flex items-center justify-center gap-1.5">
          {status === 'running' ? 'Repairing…' : <><Wrench size={14} /> Repair Bug</>}
        </button>
        {!canRun && files.length > 0 && (
          <p className="text-[0.6rem] text-zinc-500 text-center mt-1">Select source + test file</p>
        )}
      </div>
    </>
  )
}

/* ── Shared helpers ──────────────────────────────────────────── */
function ActivePills({ settings }: { settings: RunSettings }) {
  const any = settings.docker || settings.deepScan || settings.intense || settings.hitl || settings.planMode
  if (!any) return null
  return (
    <div className="px-3 py-2 border-t border-zinc-800 flex flex-wrap gap-1">
      {settings.docker   && <span className="px-1.5 py-0.5 text-[0.6rem] rounded bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">Docker</span>}
      {settings.deepScan && <span className="px-1.5 py-0.5 text-[0.6rem] rounded bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">Deep Scan</span>}
      {settings.intense  && <span className="px-1.5 py-0.5 text-[0.6rem] rounded bg-amber-500/20 text-amber-400 border border-amber-500/30">Intense</span>}
      {settings.hitl     && <span className="px-1.5 py-0.5 text-[0.6rem] rounded bg-indigo-500/20 text-indigo-400 border border-indigo-500/30">HITL</span>}
      {settings.planMode    && <span className="px-1.5 py-0.5 text-[0.6rem] rounded bg-violet-500/20 text-violet-400 border border-violet-500/30">Plan</span>}
      {settings.useBaseOnly && <span className="px-1.5 py-0.5 text-[0.6rem] rounded bg-amber-500/20 text-amber-400 border border-amber-500/30">Base Only</span>}
      {settings.autoRepair  && <span className="px-1.5 py-0.5 text-[0.6rem] rounded bg-amber-500/20 text-amber-400 border border-amber-500/30">🔧 Repair</span>}
      <span className={`px-1.5 py-0.5 text-[0.6rem] rounded border ${
        settings.qualityMode === 'best'     ? 'bg-amber-500/20 text-amber-400 border-amber-500/30'
        : settings.qualityMode === 'balanced' ? 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30'
        : 'bg-zinc-800/60 text-zinc-500 border-zinc-700'
      }`}>
        {settings.qualityMode === 'best' ? '💎 Best' : settings.qualityMode === 'balanced' ? '⚖️ Balanced' : '⚡ Fast'}
      </span>
    </div>
  )
}

function RunButton({ status, onClick, label, disabled = false }: {
  status: string; onClick: () => void; label: string; disabled?: boolean
}) {
  return (
    <div className="p-3 border-t border-zinc-800 mt-auto">
      <button onClick={onClick}
        disabled={disabled || status === 'running'}
        className="w-full py-2 rounded text-sm font-semibold bg-emerald-500 hover:bg-emerald-600 text-white disabled:opacity-40 transition-colors flex items-center justify-center gap-1.5">
        {status === 'running' ? 'Running...' : <><Rocket size={14} />{label}</>}
      </button>
    </div>
  )
}
