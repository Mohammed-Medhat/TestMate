import { useState, useEffect, useRef } from 'react'
import { ChevronRight, Bug, Shield, FlaskConical, X, Eye, FileCode,
  AlertTriangle, CheckCircle, XCircle, Pencil, Pause, ClipboardList, MessageSquare, Send } from 'lucide-react'
import type { Mode, ActiveTab, ReviewRequest, PlanRequest, Requirement, Scenario, PartCResult, PrepassSummary, StaleDetail } from '../types'
import RequirementsList from './RequirementsList'
import ScenariosList from './ScenariosList'
import CodeViewer from './CodeViewer'
import PipelineStepper, { type Stage } from './PipelineStepper'

interface Props {
  mode: Mode
  results: any
  logs: string[]
  streamedCode: string
  aiStatus: string
  status: string
  progress: { current: number; total: number; file: string }
  stages: Stage[]
  elapsedSec: number
  activeTab: ActiveTab
  setActiveTab: (t: ActiveTab) => void
  selectedBug: any; setSelectedBug: (b: any) => void
  selectedCoverageFile: string; setSelectedCoverageFile: (f: string) => void
  reviewRequest: ReviewRequest | null
  onReviewDecision: (decision: string, editedCode?: string) => void
  planRequest: PlanRequest | null
  onPlanDecision: (plan: string) => void
  // Part A results
  requirements: Requirement[]
  scenarios: Scenario[]
  features: unknown
  // Pre-pass results
  prepassResults?: PrepassSummary[]
  staleDetails?: StaleDetail[]
}

/* ── Severity badge ────────────────────────────────────── */
function SeverityBadge({ severity }: { severity: string }) {
  const colors: Record<string, string> = {
    Critical: 'bg-red-500/20 text-red-400 border-red-500/30',
    High:     'bg-orange-500/20 text-orange-400 border-orange-500/30',
    Medium:   'bg-yellow-500/20 text-yellow-400 border-yellow-500/30',
    Low:      'bg-blue-500/20 text-blue-400 border-blue-500/30',
  }
  return <span className={`px-2 py-0.5 rounded text-xs font-medium border ${colors[severity] || colors.Medium}`}>{severity}</span>
}

/* ── Bug detail modal ──────────────────────────────────── */
function BugDetailModal({ bug, index, onClose }: { bug: any; index: number; onClose: () => void }) {
  return (
    <div className="fixed inset-0 bg-black/60 flex items-center justify-center z-50" onClick={onClose}>
      <div className="bg-zinc-900 rounded-xl border border-zinc-700 w-[600px] max-h-[80vh] overflow-hidden shadow-2xl animate-modal-in" onClick={e => e.stopPropagation()}>
        <div className="flex items-center justify-between p-4 border-b border-zinc-800">
          <div className="flex items-center gap-3">
            <Bug size={18} className="text-red-400" />
            <h2 className="text-lg font-semibold text-zinc-100">ZD-{String(index + 1).padStart(3, '0')}</h2>
            <SeverityBadge severity={bug.confidence || 'High'} />
          </div>
          <button onClick={onClose} className="p-1 hover:bg-zinc-800 rounded transition-colors"><X size={18} className="text-zinc-400" /></button>
        </div>
        <div className="p-5 space-y-4 overflow-y-auto max-h-[calc(80vh-70px)]">
          <div><h3 className="text-xs font-semibold text-zinc-400 uppercase tracking-wider mb-1">Bug Type</h3><p className="text-sm text-zinc-200">{bug.bug_type || 'Logic error'}</p></div>
          <div><h3 className="text-xs font-semibold text-zinc-400 uppercase tracking-wider mb-1">Description</h3><p className="text-sm text-zinc-300 leading-relaxed">{bug.description || bug.bug_type || 'No description'}</p></div>
          {bug.evidence && (
            <div><h3 className="text-xs font-semibold text-zinc-400 uppercase tracking-wider mb-1">Evidence</h3>
              <pre className="bg-zinc-950 border border-zinc-800 rounded-lg p-3 text-xs text-zinc-300 font-mono overflow-auto max-h-48 whitespace-pre-wrap">{bug.evidence}</pre>
            </div>
          )}
          {bug.suggestion && (
            <div className="bg-emerald-500/10 border border-emerald-500/20 rounded-lg p-3">
              <h3 className="text-xs font-semibold text-emerald-400 uppercase tracking-wider mb-1">💡 Suggestion</h3>
              <p className="text-sm text-zinc-300">{bug.suggestion}</p>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

/* ── Coverage view ─────────────────────────────────────── */
function CoverageView({ results, selectedFile, setSelectedFile }: { results: any; selectedFile: string; setSelectedFile: (f: string) => void }) {
  const covMap = results?.coverage_map || {}
  const fileNames = Object.keys(covMap)
  const fileData = covMap[selectedFile]

  if (!fileNames.length) return (
    <div className="p-8 text-center text-zinc-500 text-sm">
      <Shield size={32} className="mx-auto mb-3 text-zinc-600" />
      <p>No coverage data available.</p>
      <p className="text-xs mt-1">Run a test evaluation to see line-by-line coverage.</p>
    </div>
  )

  const sourceLines = (fileData?.source || '').split('\n')
  const covered   = new Set(fileData?.covered_lines || [])
  const uncovered = new Set(fileData?.uncovered_lines || [])

  return (
    <div className="flex flex-col h-full">
      <div className="flex items-center gap-2 p-3 border-b border-zinc-800 bg-zinc-900/50 flex-wrap">
        {fileNames.map(fname => (
          <button key={fname} onClick={() => setSelectedFile(fname)}
            className={`px-3 py-1 text-xs rounded-full border transition-colors ${selectedFile === fname ? 'bg-emerald-500/20 border-emerald-500/50 text-emerald-400' : 'border-zinc-700 text-zinc-400 hover:border-zinc-600 hover:text-zinc-300'}`}>
            <FileCode size={12} className="inline mr-1" />{fname}
            <span className="ml-1 text-zinc-500">{covMap[fname]?.line_coverage_pct?.toFixed(1) ?? 0}%</span>
          </button>
        ))}
      </div>
      {fileData && (
        <div className="flex items-center gap-4 px-4 py-2 border-b border-zinc-800 text-xs">
          <span className="text-zinc-400">Statements: <span className="text-zinc-200 font-medium">{fileData.num_statements || 0}</span></span>
          <span className="text-zinc-400">Coverage: <span className="text-emerald-400 font-medium">{fileData.line_coverage_pct?.toFixed(1) ?? 0}%</span></span>
        </div>
      )}
      <div className="flex-1 overflow-auto font-mono text-xs">
        {sourceLines.map((line: string, i: number) => {
          const ln = i + 1, isCov = covered.has(ln), isUncov = uncovered.has(ln)
          return (
            <div key={i} className={`flex hover:bg-zinc-800/40 ${isCov ? 'bg-emerald-500/8' : isUncov ? 'bg-red-500/10' : ''}`}>
              <span className={`w-12 text-right pr-3 py-0.5 select-none shrink-0 border-r border-zinc-800/50 ${isCov ? 'text-emerald-600' : isUncov ? 'text-red-500' : 'text-zinc-600'}`}>{ln}</span>
              {isCov   && <span className="w-1 bg-emerald-500/60 shrink-0" />}
              {isUncov && <span className="w-1 bg-red-500/60 shrink-0" />}
              {!isCov && !isUncov && <span className="w-1 shrink-0" />}
              <span className="px-3 py-0.5 whitespace-pre text-zinc-300">{line || ' '}</span>
            </div>
          )
        })}
      </div>
    </div>
  )
}

/* ── Mutations view ────────────────────────────────────── */
function MutationsView({ results }: { results: any }) {
  const bugs = results?.bug_reports || []
  const survivors = bugs.filter((b: any) => b.bug_type === 'mutation_survivor')
  const mutScore = results?.mutation_score ?? 0
  return (
    <div className="p-4 space-y-4">
      <div className="flex items-center gap-4 p-4 bg-zinc-900 rounded-lg border border-zinc-800">
        <FlaskConical size={24} className="text-amber-400" />
        <div><p className="text-sm text-zinc-400">Mutation Score</p><p className="text-2xl font-bold text-zinc-100">{typeof mutScore === 'number' ? mutScore.toFixed(1) : mutScore}%</p></div>
        <div className="ml-auto text-right"><p className="text-xs text-zinc-500">Survivors: {survivors.length}</p></div>
      </div>
      {survivors.length === 0
        ? <div className="p-8 text-center text-zinc-500 text-sm"><FlaskConical size={32} className="mx-auto mb-3 text-zinc-600" /><p>{results ? 'All mutations killed — great coverage!' : 'Run evaluation to see mutation analysis.'}</p></div>
        : survivors.map((mut: any, i: number) => (
          <div key={i} className="bg-zinc-900 rounded-lg border border-zinc-800 p-3 hover:border-amber-500/30 transition-colors">
            <div className="flex items-start gap-3">
              <AlertTriangle size={16} className="text-amber-400 mt-0.5 shrink-0" />
              <div><span className="text-xs font-mono text-amber-400">{mut.target || 'unknown'}</span>
                <p className="text-sm text-zinc-300 mt-1">{mut.description || 'Mutation survived'}</p></div>
            </div>
          </div>
        ))
      }
    </div>
  )
}

/* ── Streamed code with typewriter ─────────────────────── */
function StreamedCodeBlock({ code, isRunning }: { code: string; isRunning: boolean }) {
  const scrollRef = useRef<HTMLDivElement>(null)
  const [visible, setVisible] = useState<string[]>([])
  const prevLen = useRef(0)

  useEffect(() => {
    const lines = code.split('\n')
    if (lines.length > prevLen.current) {
      const newLines = lines.slice(prevLen.current)
      let delay = 0
      newLines.forEach(line => { setTimeout(() => setVisible(p => [...p, line]), delay); delay += 30 })
      prevLen.current = lines.length
    }
  }, [code])

  useEffect(() => { if (!code) { setVisible([]); prevLen.current = 0 } }, [code])
  useEffect(() => { if (scrollRef.current) scrollRef.current.scrollTop = scrollRef.current.scrollHeight }, [visible])

  if (!code) return null
  return (
    <div className="bg-zinc-900 rounded-lg border border-zinc-800 overflow-hidden">
      <div className="p-3 border-b border-zinc-800 flex items-center gap-2">
        <span className="text-emerald-400 text-sm font-semibold">📝 Generated Test Code</span>
        {isRunning && <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />}
      </div>
      <div ref={scrollRef} className="overflow-auto max-h-[400px]">
        <div className="p-4 font-mono text-xs leading-relaxed">
          {visible.map((line, i) => <div key={i} className="code-line-enter whitespace-pre-wrap text-zinc-300">{line || ' '}</div>)}
          {isRunning && <span className="inline-block w-1.5 h-3.5 bg-emerald-400 ml-0.5 animate-pulse" />}
        </div>
      </div>
    </div>
  )
}

/* ── Review panel (HITL) ───────────────────────────────── */
function ReviewPanel({ review, onDecision }: { review: ReviewRequest; onDecision: (d: string, c?: string) => void }) {
  const [editMode, setEditMode] = useState(false)
  const [edited, setEdited] = useState(review.test_code)
  useEffect(() => { setEdited(review.test_code); setEditMode(false) }, [review.test_code])
  return (
    <div className="m-4 bg-amber-500/5 border-2 border-amber-500/30 rounded-xl overflow-hidden animate-modal-in">
      <div className="p-4 bg-amber-500/10 border-b border-amber-500/20 flex items-center gap-3">
        <Pause size={20} className="text-amber-400" />
        <div><h2 className="text-lg font-semibold text-zinc-100">Human Review Required</h2>
          <p className="text-xs text-zinc-400">File {review.file_index}/{review.file_total}: <span className="text-amber-400 font-mono">{review.target_file}</span> · {review.num_tests} tests</p>
        </div>
      </div>
      <div className="border-b border-amber-500/20">
        <div className="p-2 bg-zinc-900/50 border-b border-zinc-800 flex items-center gap-2">
          <FileCode size={14} className="text-emerald-400" />
          <span className="text-xs text-zinc-400 font-mono">{review.test_file}</span>
          <button onClick={() => setEditMode(!editMode)} className={`ml-auto px-2 py-0.5 text-xs rounded flex items-center gap-1 transition-colors ${editMode ? 'bg-amber-500/20 text-amber-400 border border-amber-500/30' : 'text-zinc-500 hover:text-zinc-300 hover:bg-zinc-800'}`}>
            <Pencil size={10} />{editMode ? 'Editing' : 'Edit'}
          </button>
        </div>
        {editMode
          ? <textarea value={edited} onChange={e => setEdited(e.target.value)} className="w-full bg-zinc-950 text-zinc-300 font-mono text-xs p-4 outline-none resize-none leading-relaxed" rows={Math.min(20, edited.split('\n').length + 2)} spellCheck={false} />
          : <pre className="p-4 font-mono text-xs text-zinc-300 overflow-auto max-h-[350px] leading-relaxed whitespace-pre-wrap bg-zinc-950">{review.test_code}</pre>
        }
      </div>
      <div className="p-4 flex items-center gap-3">
        <button onClick={() => onDecision('approve')} className="px-4 py-2 text-sm font-medium bg-emerald-500 hover:bg-emerald-600 text-white rounded-lg transition-colors flex items-center gap-1.5"><CheckCircle size={14} />Approve</button>
        {editMode && <button onClick={() => onDecision('edit', edited)} className="px-4 py-2 text-sm font-medium bg-amber-500 hover:bg-amber-600 text-white rounded-lg transition-colors flex items-center gap-1.5"><Pencil size={14} />Save & Approve</button>}
        <button onClick={() => onDecision('reject')} className="px-4 py-2 text-sm font-medium bg-red-500/20 hover:bg-red-500/30 text-red-400 border border-red-500/30 rounded-lg transition-colors flex items-center gap-1.5"><XCircle size={14} />Reject</button>
        <span className="ml-auto text-xs text-zinc-500">Waiting for your decision...</span>
      </div>
    </div>
  )
}

/* ── Plan review panel ─────────────────────────────────── */
function PlanReviewPanel({ plan, onDecision }: { plan: PlanRequest; onDecision: (p: string) => void }) {
  const [edited, setEdited] = useState(plan.plan)
  const [comment, setComment] = useState('')
  const [isEditing, setIsEditing] = useState(false)
  useEffect(() => { setEdited(plan.plan); setComment(''); setIsEditing(false) }, [plan.plan])

  const handleApprove = () => {
    let final = edited
    if (comment.trim()) final += `\n\n## User Notes:\n${comment.trim()}`
    onDecision(final)
  }

  return (
    <div className="m-4 bg-violet-500/5 border-2 border-violet-500/30 rounded-xl overflow-hidden animate-modal-in">
      <div className="p-4 bg-violet-500/10 border-b border-violet-500/20 flex items-center gap-3">
        <ClipboardList size={20} className="text-violet-400" />
        <div><h2 className="text-lg font-semibold text-zinc-100">Test Plan Review</h2>
          <p className="text-xs text-zinc-400">Target: <span className="text-violet-400 font-mono">{plan.target}</span></p>
        </div>
      </div>
      <div className="border-b border-violet-500/20">
        <div className="p-2 bg-zinc-900/50 border-b border-zinc-800 flex items-center gap-2">
          <ClipboardList size={14} className="text-violet-400" />
          <span className="text-xs text-zinc-400">Generated Test Plan</span>
          <button onClick={() => setIsEditing(!isEditing)} className={`ml-auto px-2 py-0.5 text-xs rounded flex items-center gap-1 transition-colors ${isEditing ? 'bg-violet-500/20 text-violet-400 border border-violet-500/30' : 'text-zinc-500 hover:text-zinc-300 hover:bg-zinc-800'}`}>
            <Pencil size={10} />{isEditing ? 'Editing' : 'Edit Plan'}
          </button>
        </div>
        {isEditing
          ? <textarea value={edited} onChange={e => setEdited(e.target.value)} className="w-full bg-zinc-950 text-zinc-300 font-mono text-xs p-4 outline-none resize-none leading-relaxed" rows={Math.min(18, edited.split('\n').length + 2)} spellCheck={false} />
          : <pre className="p-4 font-mono text-xs text-zinc-300 overflow-auto max-h-[300px] leading-relaxed whitespace-pre-wrap bg-zinc-950">{plan.plan}</pre>
        }
      </div>
      <div className="p-3 border-b border-violet-500/20 bg-zinc-900/30">
        <div className="flex items-center gap-2 mb-2"><MessageSquare size={14} className="text-zinc-500" /><span className="text-xs text-zinc-500">Add notes (optional)</span></div>
        <textarea value={comment} onChange={e => setComment(e.target.value)} placeholder="e.g. 'Also test with negative numbers'" className="w-full bg-zinc-950 text-zinc-300 text-xs p-2.5 rounded-lg border border-zinc-800 outline-none resize-none placeholder:text-zinc-700 focus:border-violet-500/40" rows={2} />
      </div>
      <div className="p-4 flex items-center gap-3">
        <button onClick={handleApprove} className="px-4 py-2 text-sm font-medium bg-violet-500 hover:bg-violet-600 text-white rounded-lg transition-colors flex items-center gap-1.5"><Send size={14} />{isEditing || comment.trim() ? 'Approve with Changes' : 'Approve Plan'}</button>
        <span className="ml-auto text-xs text-zinc-500">Model will follow this plan...</span>
      </div>
    </div>
  )
}

/* ── Combined Test Code view (per-file picker + code viewer) ─────────── */
function CombinedTestCodeView({ results }: { results: any }) {
  const partB = results?.part_b
  const partC: any[] = Array.isArray(results?.part_c) ? results.part_c : []
  const files: any[] = Array.isArray(partB)
    ? partB
    : (partB && typeof partB === 'object' && partB.test_code)
      ? [{ file: partB.test_file ?? 'test.py', test_code: partB.test_code, success: partB.success }]
      : []
  const [selectedIdx, setSelectedIdx] = useState(0)

  if (!files.length) {
    return <EmptyState
      icon={<Bug size={32} className="text-zinc-600" />}
      text="No test code generated yet"
      sub="Run the full pipeline to generate tests for selected files." />
  }

  const cur = files[Math.min(selectedIdx, files.length - 1)]

  // Find repair result for current file
  const curRepair = partC.find((r: any) =>
    r.source_file && cur?.target && r.source_file === cur.target.split('/').pop()
  )

  const repairsOk = partC.filter((r: any) => r.repair_success).length

  return (
    <div className="flex flex-col h-full">

      {/* Auto-repair summary banner — shown if any repairs ran */}
      {partC.length > 0 && (
        <div className={`px-4 py-2 flex items-center gap-3 text-xs shrink-0 border-b ${
          repairsOk > 0
            ? 'bg-emerald-500/10 border-emerald-500/20 text-emerald-400'
            : 'bg-red-500/10 border-red-500/20 text-red-400'
        }`}>
          <span className="text-base">{repairsOk > 0 ? '🔧' : '❌'}</span>
          <span className="font-medium">
            Auto-Repair: {repairsOk}/{partC.length} file{partC.length !== 1 ? 's' : ''} patched
          </span>
          {partC.map((r: any, i: number) => (
            <span key={i} className={`px-2 py-0.5 rounded-full border text-[10px] ${
              r.repair_success
                ? 'bg-emerald-500/15 border-emerald-500/30 text-emerald-400'
                : 'bg-red-500/10 border-red-500/20 text-red-400'
            }`}>
              {r.repair_success ? '✅' : '❌'} {r.source_file}
            </span>
          ))}
        </div>
      )}

      {/* File picker */}
      <div className="flex items-center gap-2 p-3 border-b border-zinc-800 bg-zinc-900/50 flex-wrap shrink-0">
        <span className="text-[10px] uppercase tracking-wider text-zinc-500 mr-1">
          {files.length} test file{files.length !== 1 ? 's' : ''}
        </span>
        {files.map((f, i) => {
          const repair = partC.find((r: any) => r.source_file === f.target?.split('/').pop())
          return (
            <button key={i} onClick={() => setSelectedIdx(i)}
              className={`px-3 py-1 text-xs rounded-full border transition-colors flex items-center gap-1.5 ${
                i === selectedIdx
                  ? 'bg-emerald-500/20 border-emerald-500/50 text-emerald-400'
                  : f.success === false
                    ? 'border-red-500/30 text-red-400 hover:border-red-500/50'
                    : 'border-zinc-700 text-zinc-400 hover:border-zinc-600 hover:text-zinc-300'
              }`}>
              <FileCode size={11} />
              {f.file?.split('/').pop() ?? f.test_file ?? `test_${i}.py`}
              {f.success === false && <X size={10} className="text-red-400" />}
              {repair?.repair_success && <span className="text-[9px] text-emerald-400">🔧</span>}
              {repair && !repair.repair_success && <span className="text-[9px] text-red-400">❌</span>}
            </button>
          )
        })}
      </div>

      {/* Per-file meta */}
      {cur && (
        <div className="px-4 py-1.5 border-b border-zinc-800 bg-zinc-900/30 text-[11px] text-zinc-500 flex gap-4 shrink-0 flex-wrap">
          {cur.matched_requirements_count !== undefined && (
            <span>📋 {cur.matched_requirements_count} requirements matched</span>
          )}
          {cur.iterations !== undefined && (
            <span className={cur.success ? 'text-emerald-500/70' : 'text-amber-400'}>
              🔄 {cur.iterations} iteration{cur.iterations !== 1 ? 's' : ''}
              {!cur.success && ` — max retries reached`}
            </span>
          )}
          {cur.error && <span className="text-red-400">❌ {cur.error}</span>}
          {/* Repair status for this file */}
          {curRepair?.repair_success && (
            <span className="text-emerald-400">🔧 Bug auto-repaired by PartC</span>
          )}
          {curRepair && !curRepair.repair_success && (
            <span className="text-red-400">❌ PartC could not repair</span>
          )}
        </div>
      )}

      {/* Patched code (if repaired) or generated test code */}
      <div className="flex-1 overflow-auto">
        {curRepair?.repair_success && curRepair.patched ? (
          <div>
            <div className="px-4 py-1.5 bg-emerald-500/5 border-b border-emerald-500/20 text-[11px] text-emerald-400">
              Showing: PartC repaired source code
            </div>
            <CodeViewer code={curRepair.patched} filename={curRepair.source_file ?? 'source.py'} />
          </div>
        ) : cur?.test_code ? (
          <CodeViewer code={cur.test_code} filename={cur.test_file ?? cur.file ?? 'test.py'} />
        ) : (
          <EmptyState icon={<Bug size={32} className="text-zinc-600" />}
            text="No test code for this file" sub={cur?.error ?? 'Generation may have failed silently.'} />
        )}
      </div>
    </div>
  )
}

/* ── Main content shell ────────────────────────────────── */
export default function MainContent({
  mode, results, logs, streamedCode, aiStatus, status, progress,
  stages, elapsedSec,
  activeTab, setActiveTab, selectedBug, setSelectedBug,
  selectedCoverageFile, setSelectedCoverageFile,
  reviewRequest, onReviewDecision, planRequest, onPlanDecision,
  requirements, scenarios, features,
  prepassResults = [], staleDetails = [],
}: Props) {
  const bugs = results?.bug_reports?.filter((b: any) => b.bug_type !== 'mutation_survivor') || []
  const mutations = results?.bug_reports?.filter((b: any) => b.bug_type === 'mutation_survivor') || []

  const partcResult = results as PartCResult | null

  // Tabs per mode
  const totalStale    = prepassResults.reduce((s, p) => s + p.stale_tests_fixed, 0)
  const totalUncovered = prepassResults.reduce((s, p) => s + p.uncovered_funcs.length, 0)
  const totalRealBugs  = prepassResults.reduce((s, p) => s + p.real_bugs_found, 0)

  const tabs: { key: ActiveTab; label: string; count?: number; icon: React.ReactNode }[] = mode === 'partb'
    ? [
        { key: 'bugs',      label: 'Zero-Day Bugs', count: bugs.length + totalRealBugs, icon: <Bug size={14} /> },
        { key: 'stale',     label: 'Stale Tests',   count: totalStale > 0 ? totalStale : undefined, icon: <Pencil size={14} /> },
        { key: 'coverage',  label: 'Coverage',                                icon: <Shield size={14} /> },
        { key: 'mutations', label: 'Mutations',      count: mutations.length,  icon: <FlaskConical size={14} /> },
      ]
    : mode === 'parta'
    ? [
        { key: 'requirements', label: 'Requirements', count: requirements.length, icon: <FileCode size={14} /> },
        { key: 'scenarios',    label: 'Scenarios',    count: scenarios.length,    icon: <Eye size={14} /> },
        { key: 'features',     label: 'Features',                                  icon: <Shield size={14} /> },
      ]
    : mode === 'partc'
    ? [
        { key: 'sbfl',    label: 'SBFL',    count: partcResult?.suspicious?.length, icon: <AlertTriangle size={14} /> },
        { key: 'patches', label: 'Patches', count: partcResult?.attempts?.length,   icon: <Pencil size={14} /> },
      ]
    : [
        { key: 'requirements', label: 'Requirements', icon: <FileCode size={14} /> },
        { key: 'scenarios',    label: 'Scenarios',    icon: <Eye size={14} /> },
        { key: 'testcode',     label: 'Test Code',    icon: <Bug size={14} /> },
      ]

  return (
    <div className="flex-1 flex flex-col bg-zinc-950 min-w-0">
      {/* Top bar */}
      <div className="h-12 bg-zinc-900 border-b border-zinc-800 flex items-center px-4 gap-2 shrink-0">
        <div className="flex items-center gap-1 text-sm">
          <span className="text-zinc-500">testmate</span>
          <ChevronRight size={14} className="text-zinc-600" />
          <span className="text-emerald-400 font-medium">{progress.file || (mode === 'parta' ? 'requirements' : 'test_core.py')}</span>
        </div>
        {aiStatus && (
          <div className="ml-4 flex items-center gap-2 text-xs text-zinc-400">
            <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
            <span className="truncate max-w-[300px]">{aiStatus}</span>
          </div>
        )}
        <div className="ml-auto flex items-center gap-1">
          {tabs.map(tab => (
            <button key={tab.key} onClick={() => setActiveTab(tab.key)}
              className={`px-3 py-1.5 text-sm rounded transition-all flex items-center gap-1.5 ${
                activeTab === tab.key
                  ? 'text-emerald-400 bg-emerald-500/10 border border-emerald-500/30'
                  : 'text-zinc-500 hover:text-zinc-300 hover:bg-zinc-800 border border-transparent'
              }`}>
              {tab.icon}{tab.label}
              {tab.count !== undefined && tab.count > 0 && (
                <span className={`px-1.5 py-0.5 text-[10px] rounded-full font-medium ${tab.key === 'bugs' ? 'bg-red-500/20 text-red-400' : tab.key === 'mutations' ? 'bg-amber-500/20 text-amber-400' : 'bg-emerald-500/15 text-emerald-400'}`}>
                  {tab.count}
                </span>
              )}
            </button>
          ))}
        </div>
      </div>

      {/* Pipeline stepper — shows status across all modes */}
      <PipelineStepper
        stages={stages}
        currentFile={progress.file}
        progress={progress.total > 0 ? { current: progress.current, total: progress.total } : undefined}
        elapsedSec={status === 'running' || status === 'done' ? elapsedSec : undefined}
      />

      {/* Body */}
      <div className="flex-1 overflow-auto flex flex-col">
        {streamedCode && !reviewRequest && (
          <div className="p-4 pb-0"><StreamedCodeBlock code={streamedCode} isRunning={status === 'running'} /></div>
        )}
        {reviewRequest && <ReviewPanel review={reviewRequest} onDecision={onReviewDecision} />}
        {planRequest   && <PlanReviewPanel plan={planRequest} onDecision={onPlanDecision} />}

        {/* Tab content */}
        {!reviewRequest && !planRequest && (
          <div className="flex-1">
            {/* Part B — Stale Tests tab */}
            {mode === 'partb' && activeTab === 'stale' && (
              <StaleTestsView prepassResults={prepassResults} staleDetails={staleDetails} />
            )}

            {/* Part B tabs */}
            {mode === 'partb' && activeTab === 'bugs' && (
              bugs.length === 0
                ? <EmptyState icon={<Bug size={32} className="text-zinc-600" />} text="No zero-day bugs detected" sub={results ? 'Clean run!' : 'Run evaluation to see results.'} />
                : <div className="p-4 space-y-2">
                    {bugs.map((bug: any, i: number) => {
                      const repair = bug.repair
                      return (
                        <div key={i} onClick={() => setSelectedBug({ bug, index: i })}
                          className={`flex items-center gap-3 p-3 bg-zinc-900 rounded-lg border cursor-pointer transition-colors ${
                            repair?.success ? 'border-emerald-500/30 hover:border-emerald-500/50'
                            : repair?.attempted ? 'border-red-500/20 hover:border-red-500/30'
                            : 'border-zinc-800 hover:border-red-500/30'
                          }`}>
                          <Bug size={16} className="text-red-400 shrink-0" />
                          <div className="flex-1 min-w-0">
                            <p className="text-sm text-zinc-200 truncate">{bug.description || bug.bug_type}</p>
                            <div className="flex items-center gap-1.5 mt-0.5">
                              <p className="text-xs text-zinc-500 font-mono">{bug.target}</p>
                              {/* Bug source type badge */}
                              {bug.bug_type === 'mutation_survivor' && (
                                <span className="text-[9px] px-1 py-0.5 rounded bg-amber-500/10 text-amber-500 border border-amber-500/20">mutmut</span>
                              )}
                              {bug.verdict === 'confirmed' && bug.bug_type === 'logic_error' && !bug.repair && (
                                <span className="text-[9px] px-1 py-0.5 rounded bg-red-500/10 text-red-400 border border-red-500/20">pre-pass</span>
                              )}
                              {bug.confidence_score !== undefined && (
                                <span className="text-[9px] px-1 py-0.5 rounded bg-zinc-800 text-zinc-500 border border-zinc-700">
                                  {bug.confidence_score}/100
                                </span>
                              )}
                            </div>
                          </div>
                          {/* Repair status badge */}
                          {repair?.success && (
                            <span className="text-[10px] px-1.5 py-0.5 rounded bg-emerald-500/15 text-emerald-400 border border-emerald-500/30 shrink-0">
                              🔧 Fixed
                            </span>
                          )}
                          {repair?.attempted && !repair?.success && (
                            <span className="text-[10px] px-1.5 py-0.5 rounded bg-red-500/10 text-red-400 border border-red-500/20 shrink-0">
                              ❌ Unfixable
                            </span>
                          )}
                          <SeverityBadge severity={bug.confidence || 'High'} />
                          <button className="text-zinc-500 hover:text-zinc-200"><Eye size={14} /></button>
                        </div>
                      )
                    })}
                  </div>
            )}
            {mode === 'partb' && activeTab === 'coverage' && <CoverageView results={results} selectedFile={selectedCoverageFile} setSelectedFile={setSelectedCoverageFile} />}
            {mode === 'partb' && activeTab === 'mutations' && <MutationsView results={results} />}

            {/* Part A tabs */}
            {mode === 'parta' && activeTab === 'requirements' && (
              requirements.length === 0
                ? <EmptyState icon={<FileCode size={32} className="text-zinc-600" />} text="No requirements yet" sub="Run Part A to extract requirements." />
                : <RequirementsList requirements={requirements} />
            )}
            {mode === 'parta' && activeTab === 'scenarios' && <ScenariosList scenarios={scenarios} />}
            {mode === 'parta' && activeTab === 'features' && (
              features
                ? <pre className="p-4 text-xs text-zinc-300 font-mono leading-relaxed whitespace-pre-wrap">{JSON.stringify(features, null, 2)}</pre>
                : <EmptyState icon={<Shield size={32} className="text-zinc-600" />} text="No features extracted yet" sub="Run Part A in README mode." />
            )}

            {/* Combined tabs */}
            {mode === 'combined' && activeTab === 'requirements' && <RequirementsList requirements={requirements} />}
            {mode === 'combined' && activeTab === 'scenarios'    && <ScenariosList scenarios={scenarios} />}
            {mode === 'combined' && activeTab === 'testcode'     && <CombinedTestCodeView results={results} />}

            {/* Part C tabs */}
            {mode === 'partc' && activeTab === 'sbfl' && <SbflView result={partcResult} />}
            {mode === 'partc' && activeTab === 'patches' && <PatchesView result={partcResult} />}

            {/* Idle welcome */}
            {status === 'idle' && !requirements.length && !results && (
              <EmptyState
                icon={<span className="text-5xl">{mode === 'combined' ? '🔗' : mode === 'parta' ? '📄' : mode === 'partb' ? '🤖' : '🔧'}</span>}
                text={mode === 'combined' ? 'Fill in the sidebar and run the full pipeline' : mode === 'parta' ? 'Configure inputs and run Part A' : mode === 'partb' ? 'Discover a repository and generate tests' : 'Select a buggy file and test suite, then run repair'}
                sub=""
              />
            )}
          </div>
        )}
      </div>

      {/* Bug detail modal */}
      {selectedBug && (
        <BugDetailModal bug={selectedBug.bug} index={selectedBug.index} onClose={() => setSelectedBug(null)} />
      )}
    </div>
  )
}

/* ── Stale Tests view ─────────────────────────────────── */
function StaleTestsView({ prepassResults, staleDetails }: {
  prepassResults: PrepassSummary[]
  staleDetails: StaleDetail[]
}) {
  const [expanded, setExpanded] = useState<number | null>(null)

  const filesWithExisting = prepassResults.filter(p => p.has_existing)
  const totalStale        = prepassResults.reduce((s, p) => s + p.stale_tests_fixed, 0)
  const totalUncovered    = prepassResults.reduce((s, p) => s + p.uncovered_funcs.length, 0)
  const totalRealBugs     = prepassResults.reduce((s, p) => s + p.real_bugs_found, 0)

  if (filesWithExisting.length === 0) return (
    <EmptyState
      icon={<Pencil size={32} className="text-zinc-600" />}
      text="No existing tests found"
      sub="PartB will generate fresh test_<source>_testmate.py files."
    />
  )

  return (
    <div className="p-4 space-y-4">
      {/* Summary row */}
      <div className="grid grid-cols-3 gap-3">
        {[
          { label: 'Real Bugs Found', value: totalRealBugs, color: 'text-red-400', bg: 'bg-red-500/10 border-red-500/20' },
          { label: 'Stale Tests Fixed', value: totalStale,   color: 'text-amber-400', bg: 'bg-amber-500/10 border-amber-500/20' },
          { label: 'Coverage Gaps',  value: totalUncovered, color: 'text-blue-400',  bg: 'bg-blue-500/10 border-blue-500/20' },
        ].map(m => (
          <div key={m.label} className={`rounded-lg border p-3 text-center ${m.bg}`}>
            <p className={`text-2xl font-bold ${m.color}`}>{m.value}</p>
            <p className="text-xs text-zinc-500 mt-1">{m.label}</p>
          </div>
        ))}
      </div>

      {/* Per-file breakdown */}
      {filesWithExisting.map((p, i) => (
        <div key={i} className="bg-zinc-900 border border-zinc-800 rounded-lg overflow-hidden">
          <button
            className="w-full flex items-center gap-3 px-4 py-3 text-left hover:bg-zinc-800/50 transition-colors"
            onClick={() => setExpanded(expanded === i ? null : i)}
          >
            <FileCode size={15} className="text-zinc-400 shrink-0" />
            <span className="text-sm font-medium text-zinc-200">{p.source_file}</span>
            <div className="flex items-center gap-2 ml-auto">
              {p.all_pass && <span className="text-[10px] px-1.5 py-0.5 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">All pass</span>}
              {p.real_bugs_found > 0 && <span className="text-[10px] px-1.5 py-0.5 rounded bg-red-500/10 text-red-400 border border-red-500/20">{p.real_bugs_found} bug{p.real_bugs_found > 1 ? 's' : ''}</span>}
              {p.stale_tests_fixed > 0 && <span className="text-[10px] px-1.5 py-0.5 rounded bg-amber-500/10 text-amber-400 border border-amber-500/20">{p.stale_tests_fixed} stale</span>}
              {p.uncovered_funcs.length > 0 && <span className="text-[10px] px-1.5 py-0.5 rounded bg-blue-500/10 text-blue-400 border border-blue-500/20">{p.uncovered_funcs.length} gaps</span>}
              <ChevronRight size={14} className={`text-zinc-500 transition-transform ${expanded === i ? 'rotate-90' : ''}`} />
            </div>
          </button>

          {expanded === i && (
            <div className="px-4 pb-4 space-y-3 border-t border-zinc-800">
              {/* Existing test files */}
              <div className="pt-3">
                <p className="text-[11px] text-zinc-500 uppercase tracking-wider mb-1">Existing test files</p>
                {p.existing_test_files.map(f => (
                  <span key={f} className="inline-block mr-2 text-xs text-zinc-400 font-mono bg-zinc-800 px-2 py-0.5 rounded">{f}</span>
                ))}
              </div>

              {/* Coverage gaps */}
              {p.uncovered_funcs.length > 0 && (
                <div>
                  <p className="text-[11px] text-blue-400 uppercase tracking-wider mb-1">Coverage gaps — generating tests for:</p>
                  <div className="flex flex-wrap gap-1.5">
                    {p.uncovered_funcs.map(fn => (
                      <span key={fn} className="text-xs text-blue-300 font-mono bg-blue-500/10 border border-blue-500/20 px-2 py-0.5 rounded">{fn}()</span>
                    ))}
                  </div>
                </div>
              )}

              {/* Stale test details */}
              {p.stale_details?.length > 0 && (
                <div>
                  <p className="text-[11px] text-amber-400 uppercase tracking-wider mb-1">Stale tests auto-updated (.testmate.bak backup created)</p>
                  {p.stale_details.map((sd, j) => (
                    <div key={j} className="mb-2 rounded border border-amber-500/20 bg-amber-500/5 px-3 py-2">
                      <p className="text-xs font-medium text-amber-300">{sd.func_name}() in {sd.test_file}</p>
                      <p className="text-[10px] text-zinc-500 mt-0.5">Backup: {sd.backup_path}</p>
                      {sd.fresh_code && (
                        <pre className="text-[10px] font-mono text-zinc-400 mt-1 overflow-x-auto max-h-20 bg-zinc-950 rounded p-2">
                          {sd.fresh_code}
                        </pre>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      ))}
    </div>
  )
}

/* ── SBFL view ─────────────────────────────────────────── */
function SbflView({ result }: { result: PartCResult | null }) {
  const lines = result?.suspicious ?? []
  if (!lines.length) return (
    <EmptyState
      icon={<AlertTriangle size={32} className="text-zinc-600" />}
      text="No SBFL data yet"
      sub="Run a repair to see suspicious lines ranked by Ochiai score."
    />
  )

  const maxScore = Math.max(...lines.map(l => l.score), 0.001)
  return (
    <div className="p-4 space-y-2">
      <p className="text-xs text-zinc-500 mb-3">Top suspicious lines by Ochiai fault-localization score</p>
      {lines.map((ln, i) => {
        const pct = Math.round((ln.score / maxScore) * 100)
        return (
          <div key={i} className="bg-zinc-900 border border-zinc-800 rounded-lg p-3 hover:border-amber-500/30 transition-colors">
            <div className="flex items-center gap-3 mb-2">
              <span className="text-[10px] font-bold text-amber-400 bg-amber-500/10 border border-amber-500/20 px-2 py-0.5 rounded font-mono">
                L{ln.line}
              </span>
              <div className="flex-1 h-1.5 bg-zinc-800 rounded-full overflow-hidden">
                <div className="h-full bg-amber-500 transition-all" style={{ width: `${pct}%` }} />
              </div>
              <span className="text-xs text-amber-400 font-mono w-14 text-right">{ln.score.toFixed(3)}</span>
            </div>
            <pre className="text-xs text-zinc-300 font-mono bg-zinc-950 rounded px-3 py-2 overflow-x-auto">
              {ln.code || '(empty line)'}
            </pre>
          </div>
        )
      })}
    </div>
  )
}

/* ── Patches view ──────────────────────────────────────── */
function PatchesView({ result }: { result: PartCResult | null }) {
  const [showDiff, setShowDiff] = useState<number | null>(null)
  const attempts = result?.attempts ?? []

  if (!attempts.length) return (
    <EmptyState
      icon={<Pencil size={32} className="text-zinc-600" />}
      text="No repair attempts yet"
      sub="Run a repair to see per-attempt patch history."
    />
  )

  return (
    <div className="p-4 space-y-3">
      {/* Summary bar */}
      {result && (
        <div className={`flex items-center gap-3 p-3 rounded-lg border ${
          result.success
            ? 'bg-emerald-500/10 border-emerald-500/30'
            : 'bg-red-500/10 border-red-500/30'
        }`}>
          {result.success
            ? <CheckCircle size={20} className="text-emerald-400" />
            : <XCircle size={20} className="text-red-400" />
          }
          <div>
            <p className="text-sm font-medium text-zinc-200">
              {result.success ? 'Bug fixed!' : 'Repair failed'}
            </p>
            <p className="text-xs text-zinc-500">
              {attempts.length} attempt{attempts.length !== 1 ? 's' : ''} · {result.elapsed_sec}s
            </p>
          </div>
        </div>
      )}

      {/* Attempt cards */}
      {attempts.map((att, i) => (
        <div key={i} className={`bg-zinc-900 border rounded-lg overflow-hidden ${
          att.status === 'success' ? 'border-emerald-500/30' : 'border-zinc-800'
        }`}>
          <div className="flex items-center gap-3 px-3 py-2 border-b border-zinc-800">
            {att.status === 'success'
              ? <CheckCircle size={15} className="text-emerald-400" />
              : <XCircle size={15} className="text-red-400" />
            }
            <span className="text-sm font-medium text-zinc-200">Attempt {att.n}</span>
            <span className="text-xs text-zinc-500 ml-auto">{att.result}</span>
            {att.patched && (
              <button onClick={() => setShowDiff(showDiff === i ? null : i)}
                className="text-xs text-amber-400 hover:text-amber-300 ml-2">
                {showDiff === i ? 'Hide' : 'View'} patch
              </button>
            )}
          </div>
          {showDiff === i && att.patched && (
            <pre className="text-xs text-zinc-300 font-mono p-3 overflow-x-auto bg-zinc-950 max-h-64">
              {att.patched}
            </pre>
          )}
        </div>
      ))}

      {/* Final patched code */}
      {result?.success && result.patched && (
        <div className="mt-4">
          <h3 className="text-xs font-semibold text-emerald-400 uppercase tracking-wider mb-2">Fixed Source</h3>
          <pre className="text-xs text-zinc-300 font-mono bg-zinc-900 border border-emerald-500/20 rounded-lg p-4 overflow-x-auto max-h-80">
            {result.patched}
          </pre>
        </div>
      )}
    </div>
  )
}

function EmptyState({ icon, text, sub }: { icon: React.ReactNode; text: string; sub: string }) {
  return (
    <div className="flex flex-col items-center justify-center h-full gap-3 text-zinc-500 py-16">
      {icon}
      <p className="text-sm text-zinc-400">{text}</p>
      {sub && <p className="text-xs text-zinc-600">{sub}</p>}
    </div>
  )
}
