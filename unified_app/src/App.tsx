import { useState, useCallback, useEffect, useRef } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Toaster, toast } from 'sonner'
import Landing from './components/Landing'
import LeftSidebar from './components/LeftSidebar'
import MainContent from './components/MainContent'
import RightSidebar from './components/RightSidebar'
import ProjectSettingsModal from './components/ProjectSettingsModal'
import HelpModal from './components/HelpModal'
import { getInitialStages, advanceStages, detectStageFromLog, type Stage } from './components/PipelineStepper'
import type {
  View, Mode, PartAMode, ActiveTab, RunSettings, FileInfo,
  ReviewRequest, PlanRequest, Requirement, Scenario, HistoryItem, PartAStats, Progress,
  PartCResult, PrepassSummary, StaleDetail,
} from './types'

const API = (window as any).api?.baseUrl ?? 'http://127.0.0.1:8080'

const DEFAULT_SETTINGS: RunSettings = {
  docker: false, deepScan: false, maxRetries: 3, hitl: false, intense: false,
  planMode: false, useBaseOnly: false, qualityMode: 'fast', autoRepair: false,
}

export default function App() {
  // ── View & mode ──────────────────────────────────────────────────────────
  const [view, setView]           = useState<View>('landing')
  const [mode, setMode]           = useState<Mode>('combined')
  const [isModalOpen, setModal]   = useState(false)
  const [isHelpOpen,  setHelp]    = useState(false)
  const [settings, setSettings]   = useState<RunSettings>(DEFAULT_SETTINGS)
  const [history, setHistory]     = useState<HistoryItem[]>([])

  // ── Part B state ─────────────────────────────────────────────────────────
  const [repoUrl, setRepoUrl]     = useState('')
  const [branch, setBranch]       = useState('')
  const [files, setFiles]         = useState<FileInfo[]>([])
  const [selected, setSelected]   = useState<Set<number>>(new Set())
  const [discovering, setDisc]    = useState(false)
  const [repoName, setRepoName]   = useState('')

  // ── Part A state ─────────────────────────────────────────────────────────
  const [partAMode, setPartAMode] = useState<PartAMode>('srs')
  const [srsFile, setSrsFile]     = useState<File | null>(null)
  const [readme, setReadme]       = useState('')
  const [repoNameA, setRepoNameA] = useState('')
  const [problems, setProblems]   = useState('')
  const [expected, setExpected]   = useState('')
  const [edgeCases, setEdgeCases] = useState('')
  const [threshold, setThreshold] = useState(0.85)

  // ── Combined state ───────────────────────────────────────────────────────
  const [combSrsFile, setCombSrsFile]       = useState<File | null>(null)
  const [combReadme, setCombReadme]         = useState('')
  const [combTargetFile, setCombTargetFile] = useState('')
  const [combImportPath, setCombImportPath] = useState('')

  // ── Shared run state ─────────────────────────────────────────────────────
  const [status, setStatus]       = useState<'idle'|'running'|'done'|'error'>('idle')
  const [logs, setLogs]           = useState<string[]>([])
  const [results, setResults]     = useState<any>(null)
  const [streamedCode, setStream] = useState('')
  const [aiStatus, setAiStatus]   = useState('')
  const [progress, setProgress]   = useState<Progress>({ current: 0, total: 0, file: '' })
  const [stages, setStages]       = useState<Stage[]>(getInitialStages('combined', 'fast'))
  const [elapsedSec, setElapsed]  = useState(0)
  const runStartRef = useRef<number>(0)
  const timerRef    = useRef<number | null>(null)
  const [activeTab, setActiveTab] = useState<ActiveTab>('bugs')
  const [selectedBug, setSelectedBug]   = useState<any>(null)
  const [selectedCovFile, setSelectedCovFile] = useState('')
  const [reviewRequest, setReview]     = useState<ReviewRequest | null>(null)
  const [planRequest, setPlan]         = useState<PlanRequest | null>(null)

  // Part A results
  const [requirements, setRequirements] = useState<Requirement[]>([])
  const [scenarios, setScenarios]       = useState<Scenario[]>([])
  const [features, setFeatures]         = useState<unknown>(null)
  const [partAStats, setPartAStats]     = useState<PartAStats>({ total:0, requirements:0, non_requirements:0, unlabeled:0 })

  // Part C state
  const [partcSourceIdx, setPartcSourceIdx] = useState<number | null>(null)
  const [partcTestIdx,   setPartcTestIdx]   = useState<number | null>(null)

  // Pre-pass state (existing test triage results)
  const [prepassResults, setPrepassResults] = useState<PrepassSummary[]>([])
  const [staleDetails,   setStaleDetails]   = useState<StaleDetail[]>([])

  // ── Load history on mount ────────────────────────────────────────────────
  useEffect(() => {
    fetch(`${API}/api/history`).then(r => r.json()).then(setHistory).catch(() => {})
  }, [])

  // ── Tick elapsed-time clock while running ────────────────────────────────
  useEffect(() => {
    if (status === 'running') {
      runStartRef.current = Date.now()
      setElapsed(0)
      timerRef.current = window.setInterval(() => {
        setElapsed((Date.now() - runStartRef.current) / 1000)
      }, 1000)
    } else if (timerRef.current) {
      clearInterval(timerRef.current)
      timerRef.current = null
    }
    return () => { if (timerRef.current) clearInterval(timerRef.current) }
  }, [status])

  // ── Centralised log appender (also drives the stage stepper) ─────────────
  const addLog = useCallback((msg: string) => {
    setLogs(p => [...p, msg])
    const stageId = detectStageFromLog(mode, msg)
    if (stageId) setStages(s => advanceStages(s, stageId))
  }, [mode])

  const refreshHistory = () => {
    fetch(`${API}/api/history`).then(r => r.json()).then(setHistory).catch(() => {})
  }

  const saveRun = (modeName: string, summary: string, st: string) => {
    fetch(`${API}/api/history/record`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ mode: modeName, summary, status: st })
    }).then(refreshHistory).catch(() => {})
  }

  // ── Card transition state (fill → split panels) ─────────────────────────
  const [cardTx, setCardTx] = useState<{
    rect: DOMRect; color: string; phase: 'fill' | 'split'
  } | null>(null)

  // ── Navigation ───────────────────────────────────────────────────────────
  const enterMode = (m: Mode, rect?: DOMRect, color?: string) => {
    setMode(m)
    if (rect) {
      setCardTx({ rect, color: color ?? '#10b981', phase: 'fill' })
      setTimeout(() => {
        setView('workspace')
        setCardTx(ct => ct ? { ...ct, phase: 'split' } : null)
      }, 460)
      setTimeout(() => setCardTx(null), 960)
    } else {
      setView('workspace')
    }
  }
  const goHome = () => { setView('landing'); refreshHistory() }

  // ── Part B: discover ─────────────────────────────────────────────────────
  const discover = useCallback(async () => {
    if (!repoUrl.trim()) return
    setDisc(true)
    try {
      const res = await fetch(`${API}/api/partb/discover`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url: repoUrl, branch }),
      })
      const data = await res.json()
      if (data.error) { addLog('❌ ' + data.error) }
      else {
        setFiles(data.files ?? [])
        setRepoName(data.repo_name ?? '')
        setSelected(new Set((data.files ?? []).map((_: any, i: number) => i)))
        addLog(`✅ Found ${(data.files ?? []).length} files in ${data.repo_name}`)
      }
    } catch (e: any) { addLog('❌ ' + e.message) }
    setDisc(false)
  }, [repoUrl, branch])

  const toggleFile = (i: number) => {
    setSelected(prev => { const n = new Set(prev); n.has(i) ? n.delete(i) : n.add(i); return n })
  }

  // ── HITL callbacks ────────────────────────────────────────────────────────
  const handleReview = useCallback(async (decision: string, editedCode?: string) => {
    try {
      await fetch(`${API}/api/review`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ decision, edited_code: editedCode || '' }),
      })
    } catch {}
    setReview(null)
  }, [])

  const handlePlan = useCallback(async (plan: string) => {
    try {
      await fetch(`${API}/api/plan_review`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ plan }),
      })
    } catch {}
    setPlan(null)
  }, [])

  // ── SSE helper ────────────────────────────────────────────────────────────
  const connectSSE = (url: string, onMsg: (evt: any) => void, onDone: (ok: boolean) => void) => {
    const es = new EventSource(url)
    es.onmessage = e => {
      try {
        const msg = JSON.parse(e.data)
        if (msg.type === 'complete') { es.close(); onDone(true) }
        else if (msg.type === 'error') { es.close(); onDone(false) }
        else onMsg(msg)
      } catch {}
    }
    es.onerror = () => { es.close(); onDone(false) }
  }

  // ── Main run dispatcher ───────────────────────────────────────────────────
  const startRun = useCallback(async () => {
    setStatus('running'); setLogs([]); setResults(null); setStream('')
    setReview(null); setPlan(null); setSelectedBug(null)
    setStages(getInitialStages(mode, settings.qualityMode))
    setElapsed(0)

    toast.loading('Starting run…', { id: 'run', duration: Infinity })
    try {
      if (mode === 'partb') {
        if (selected.size === 0) { setStatus('idle'); toast.dismiss('run'); return }
        const selectedFiles = [...selected].map(i => files[i])
        setActiveTab('bugs')
        setPrepassResults([])
        setStaleDetails([])
        const res = await fetch(`${API}/api/partb/run`, {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            files: selectedFiles, url: repoUrl, branch,
            use_docker: settings.docker, deep_scan: settings.deepScan,
            max_retries: settings.maxRetries, hitl: settings.hitl,
            intense: settings.intense, plan_mode: settings.planMode,
            use_base_only: settings.useBaseOnly, quality_mode: settings.qualityMode,
            auto_repair: settings.autoRepair,
          }),
        })
        const { job_id } = await res.json()
        connectSSE(`${API}/api/partb/stream?job_id=${job_id}`, msg => {
          if (msg.type === 'log')         addLog(msg.message ?? '')
          if (msg.type === 'ai_status')   setAiStatus(`${msg.status}: ${msg.detail ?? ''}`)
          if (msg.type === 'progress')    setProgress({ current: msg.current, total: msg.total, file: msg.file })
          if (msg.type === 'code_stream') setStream(p => p + (p ? '\n' : '') + (msg.code ?? ''))
          if (msg.type === 'code_clear')  setStream('')
          if (msg.type === 'review_request') setReview(msg as ReviewRequest)
          if (msg.type === 'plan_request')   setPlan(msg as PlanRequest)
          // per-file result — append to part_b array
          if (msg.type === 'file_result') {
            setResults((prev: any) => {
              const existing: any[] = Array.isArray(prev?.part_b) ? prev.part_b : []
              return { ...prev, part_b: [...existing, msg.data] }
            })
            setActiveTab('mutations')  // switch to Test Code tab
          }
          if (msg.type === 'result') {
            setResults(msg.data)
            const cov = (msg.data as any)?.coverage_map
            if (cov) setSelectedCovFile(Object.keys(cov)[0] ?? '')
          }
          if (msg.type === 'prepass_result') {
            const p = msg.data as PrepassSummary
            setPrepassResults(prev => [...prev, p])
            if (p.stale_tests_fixed > 0) setActiveTab('stale')
            if (p.real_bugs_found > 0)   addLog(`🐛 Pre-pass: ${p.real_bugs_found} real bug(s) in ${p.source_file} → queued for PartC`)
            if (p.stale_tests_fixed > 0) addLog(`🔄 Pre-pass: ${p.stale_tests_fixed} stale test(s) updated in ${p.source_file}`)
            if (p.uncovered_funcs.length > 0) addLog(`📊 Pre-pass: generating tests for ${p.uncovered_funcs.length} uncovered function(s) in ${p.source_file}`)
          }
          if (msg.type === 'stale_fixed') {
            setStaleDetails(prev => [...prev, msg.data as StaleDetail])
          }
          if (msg.type === 'repair_start') {
            addLog(`🔧 Bug Repair starting — ${msg.data?.total_files} file(s)`)
            toast.loading(`Repairing ${msg.data?.total_files} file(s)…`, { id: 'repair' })
          }
          if (msg.type === 'repair_complete') {
            const d = msg.data as any
            addLog(d?.repair_success ? `✅ Repaired: ${d?.source_file}` : `❌ Could not repair: ${d?.source_file}`)
            if (d?.repair_success) toast.success(`Fixed: ${d?.source_file}`, { id: 'repair' })
            else toast.error(`Could not fix: ${d?.source_file}`, { id: 'repair' })
            setResults((prev: any) => ({
              ...prev,
              part_c: [...(Array.isArray(prev?.part_c) ? prev.part_c : []), d],
            }))
          }
          if (msg.type === 'verdict_update') {
            const d = msg.data as any
            addLog(`  ${d?.target}: ${d?.old_verdict} → ${d?.new_verdict}`)
          }
        }, ok => {
          toast.dismiss('run')
          if (ok) toast.success('Tests generated successfully')
          else toast.error('Run failed — check the trace log')
          setStatus(ok ? 'done' : 'error'); if (ok) setStages(s => advanceStages(s, 'done'))
          saveRun('partb', `Generated tests for ${repoName}`, ok ? 'success' : 'error')
        })

      } else if (mode === 'parta') {
        setRequirements([]); setScenarios([]); setFeatures(null)
        setActiveTab('requirements')

        if (partAMode === 'srs' || partAMode === 'both') {
          if (!srsFile) { addLog('⚠️ SRS file required'); setStatus('error'); return }
          addLog(`Processing ${srsFile.name}…`)
          const fd = new FormData(); fd.append('file', srsFile); fd.append('fuzzy_threshold', String(threshold))
          const res = await fetch(`${API}/api/parta/srs/run`, { method: 'POST', body: fd })
          const d = await res.json()
          if (d.status !== 'success') { addLog('❌ ' + d.message); setStatus('error'); return }
          setRequirements(d.requirements ?? [])
          const st = d.stats ?? {}
          setPartAStats({ total: st.total ?? 0, requirements: st.requirements ?? 0, non_requirements: st.non_requirements ?? 0, unlabeled: st.unlabeled ?? 0 })
          const scoreAvg = (d.requirements ?? []).reduce((a: number, r: any) => a + (r.score ?? 0), 0) / Math.max((d.requirements ?? []).length, 1)
          setPartAStats(p => ({ ...p, matchScoreAvg: scoreAvg } as any))
          addLog(`✅ SRS done — ${st.requirements ?? 0} requirements`)
        }

        if (partAMode === 'readme' || partAMode === 'both') {
          if (readme.trim()) {
            addLog('Extracting README features…')
            const res = await fetch(`${API}/api/parta/readme/run`, {
              method: 'POST', headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ content: readme, repo_name: repoNameA,
                user_input: { problems, expected, edge_cases: edgeCases } })
            })
            const d = await res.json()
            if (d.status === 'success') {
              setScenarios(d.test_scenarios ?? [])
              setFeatures(d.features ?? null)
              addLog(`✅ README done — ${(d.test_scenarios ?? []).length} scenarios`)
              if (partAMode === 'readme') setActiveTab('scenarios')
            } else { addLog('❌ ' + d.message) }
          }
        }

        toast.dismiss('run')
        toast.success('Extraction complete')
        saveRun('parta', `Extracted requirements from ${srsFile?.name ?? readme.slice(0, 30) + '…'}`, 'success')
        setStatus('done'); setStages(s => advanceStages(s, 'done'))

      } else if (mode === 'partc') {
        // Part C: SBFL-guided bug repair
        if (partcSourceIdx === null || partcTestIdx === null) {
          addLog('⚠️ Select a source file and a test file first')
          setStatus('idle')
          return
        }
        const srcFile  = files[partcSourceIdx]
        const testFile = files[partcTestIdx]
        setActiveTab('sbfl')
        addLog(`🔧 Starting APR on ${srcFile.path.split('/').pop()} + ${testFile.path.split('/').pop()}…`)

        const res = await fetch(`${API}/api/partc/run`, {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            source_file:  srcFile.abs_path,
            test_file:    testFile.abs_path,
            max_attempts: settings.maxRetries,
          }),
        })
        const { job_id, error } = await res.json()
        if (error) { addLog('❌ ' + error); setStatus('error'); return }

        connectSSE(`${API}/api/partc/stream?job_id=${job_id}`, msg => {
          if (msg.type === 'log')            addLog(msg.message ?? '')
          if (msg.type === 'sbfl_result')    setResults((prev: any) => ({ ...prev, suspicious: msg.data }))
          if (msg.type === 'attempt_complete') {
            const att = msg.data as any
            setResults((prev: any) => {
              const existing = Array.isArray(prev?.attempts) ? prev.attempts : []
              return { ...prev, attempts: [...existing, att] }
            })
            setActiveTab('patches')
          }
          if (msg.type === 'result') {
            setResults(msg.data as PartCResult)
          }
        }, ok => {
          toast.dismiss('run')
          if (ok) toast.success('Bug repair complete')
          else toast.error('Repair failed — check the trace log')
          setStatus(ok ? 'done' : 'error')
          if (ok) setStages(s => advanceStages(s, 'done'))
          saveRun('partc',
            `Repaired ${srcFile.path.split('/').pop() ?? 'file'} from ${repoName || 'repo'}`,
            ok ? 'success' : 'error')
        })

      } else {
        // Combined — multi-file: SRS + README + discovered repo files
        if (selected.size === 0) { addLog('⚠️ Select at least one file to test'); setStatus('idle'); return }
        const selectedFiles = [...selected].map(i => files[i])
        setRequirements([]); setScenarios([]); setActiveTab('requirements')
        setPrepassResults([]); setStaleDetails([])
        addLog(`Starting full pipeline on ${selectedFiles.length} file(s)…`)

        const fd = new FormData()
        if (combSrsFile) fd.append('srs_file', combSrsFile)
        fd.append('readme_content', combReadme)
        fd.append('files_json', JSON.stringify(selectedFiles))
        fd.append('deep_scan', String(settings.deepScan))
        fd.append('max_retries', String(settings.maxRetries))
        fd.append('plan_mode', String(settings.planMode))
        fd.append('use_base_only', String(settings.useBaseOnly))
        fd.append('quality_mode', settings.qualityMode ?? 'fast')
        fd.append('auto_repair', String(settings.autoRepair))
        fd.append('use_docker', String(settings.docker))

        const res = await fetch(`${API}/api/combined/run`, { method: 'POST', body: fd })
        const { job_id, error } = await res.json()
        if (error) { addLog('❌ ' + error); setStatus('error'); return }
        connectSSE(`${API}/api/combined/stream?job_id=${job_id}`, msg => {
          if (msg.type === 'log')         addLog(msg.message ?? '')
          if (msg.type === 'progress')    setProgress({ current: msg.current, total: msg.total, file: msg.file })
          if (msg.type === 'code_stream') setStream(p => p + (p ? '\n' : '') + (msg.code ?? ''))

          // ── Part A results ────────────────────────────────────────────────
          if (msg.type === 'result') {
            const d = msg.data as any
            if (d?.part_a?.requirements?.length) {
              setRequirements(d.part_a.requirements)
              setActiveTab('requirements')
            }
            if (d?.part_a?.scenarios?.length) setScenarios(d.part_a.scenarios)
            setResults((prev: any) => ({
              ...d,
              part_b: Array.isArray(d.part_b) && d.part_b.length
                ? d.part_b
                : (Array.isArray(prev?.part_b) ? prev.part_b : [])
            }))
          }

          // ── Per-file Part B result ────────────────────────────────────────
          if (msg.type === 'file_result') {
            const fileData = msg.data as any
            setResults((prev: any) => {
              const existing: any[] = Array.isArray(prev?.part_b) ? prev.part_b : []
              return { ...prev, part_b: [...existing, fileData] }
            })
            setActiveTab('testcode')
          }

          // ── Pre-pass results (existing test triage) ───────────────────────
          if (msg.type === 'prepass_result') {
            const p = msg.data as PrepassSummary
            setPrepassResults(prev => [...prev, p])
            if (p.real_bugs_found > 0)    addLog(`🐛 Pre-pass: ${p.real_bugs_found} real bug(s) in ${p.source_file} → PartC queued`)
            if (p.stale_tests_fixed > 0)  addLog(`🔄 Pre-pass: ${p.stale_tests_fixed} stale test(s) updated in ${p.source_file}`)
            if (p.uncovered_funcs.length) addLog(`📊 Pre-pass: generating for ${p.uncovered_funcs.length} uncovered fn(s)`)
          }
          if (msg.type === 'stale_fixed') {
            setStaleDetails(prev => [...prev, msg.data as StaleDetail])
          }

          // ── PartC repair events ───────────────────────────────────────────
          if (msg.type === 'repair_start') {
            addLog(`🔧 Auto-Repair starting (${msg.data?.total_files ?? '?'} file(s))`)
            setStages(s => advanceStages(s, 'repair'))
          }
          if (msg.type === 'repair_complete') {
            const d = msg.data as any
            addLog(d?.repair_success
              ? `✅ Repaired: ${d?.source_file}`
              : `❌ Could not repair: ${d?.source_file}`)
            setResults((prev: any) => ({
              ...prev,
              part_c: [...(Array.isArray(prev?.part_c) ? prev.part_c : []), d],
            }))
          }
          if (msg.type === 'verdict_update') {
            const d = msg.data as any
            addLog(`  ${d?.target}: ${d?.old_verdict} → ${d?.new_verdict}`)
          }
          if (msg.type === 'sbfl_result') {
            setResults((prev: any) => ({ ...prev, suspicious: msg.data }))
          }
        }, ok => {
          toast.dismiss('run')
          if (ok) toast.success(`Pipeline complete — ${selectedFiles.length} file(s) processed`)
          else toast.error('Pipeline failed — check the trace log')
          setStatus(ok ? 'done' : 'error'); if (ok) setStages(s => advanceStages(s, 'done'))
          saveRun('combined', `Full pipeline on ${selectedFiles.length} file(s) from ${repoName || 'repo'}`,
                  ok ? 'success' : 'error')
        })
      }

    } catch (e: any) {
      toast.dismiss('run')
      toast.error((e as Error).message ?? 'Unknown error')
      addLog('❌ ' + (e as Error).message)
      setStatus('error')
    }
  }, [mode, selected, files, repoUrl, branch, repoName, settings, partAMode, srsFile, readme, repoNameA, problems, expected, edgeCases, threshold, combSrsFile, combReadme, partcSourceIdx, partcTestIdx])

  // ── Keyboard shortcuts (placed after startRun is defined) ─────────────────
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      const ctrl = e.ctrlKey || e.metaKey
      if (e.key === 'Escape') {
        if (isHelpOpen)  { setHelp(false);  return }
        if (isModalOpen) { setModal(false); return }
      }
      if (e.key === '?' && !ctrl && !e.altKey && e.target === document.body) { setHelp(true); return }
      if (view !== 'workspace') return
      if (ctrl && e.key === 'Enter' && status !== 'running') { startRun(); return }
      if (!ctrl && !e.altKey && !e.shiftKey && e.target === document.body) {
        if (e.key === '1') setMode('combined')
        if (e.key === '2') setMode('parta')
        if (e.key === '3') setMode('partb')
        if (e.key === '4') setMode('partc')
      }
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [view, status, isModalOpen, isHelpOpen, startRun])

  const partAStatsForSidebar = {
    total: partAStats.total,
    requirements: partAStats.requirements,
    scenarios: scenarios.length,
    matchScoreAvg: (partAStats as any).matchScoreAvg ?? 0,
  }

  return (
    <div className="relative h-screen w-screen bg-zinc-950 text-zinc-100 overflow-hidden">
      <Toaster
        position="bottom-right"
        theme="dark"
        toastOptions={{ style: { background: '#18181b', border: '1px solid #3f3f46', color: '#f4f4f5' } }}
      />

      {/* ── Card transition: phase 1 — color fills from card via clip-path ── */}
      {cardTx && cardTx.phase === 'fill' && (() => {
        const r = cardTx.rect
        const iw = window.innerWidth, ih = window.innerHeight
        const clipStart = `inset(${r.top}px ${iw - r.right}px ${ih - r.bottom}px ${r.left}px round 16px)`
        return (
          <motion.div
            className="pointer-events-none fixed inset-0"
            style={{ background: cardTx.color, zIndex: 200 }}
            initial={{ clipPath: clipStart }}
            animate={{ clipPath: 'inset(0px 0px 0px 0px round 0px)' }}
            transition={{ duration: 0.46, ease: [0.4, 0, 0.15, 1] }}
          />
        )
      })()}

      {/* ── Card transition: phase 2 — 4 panels fly to corners ── */}
      {cardTx && cardTx.phase === 'split' && ([
        { k: 'tl', top: 0,    left:  0,  bottom: undefined, right: undefined, dx: '-105%', dy: '-105%' },
        { k: 'tr', top: 0,    right: 0,  bottom: undefined, left:  undefined, dx:  '105%', dy: '-105%' },
        { k: 'bl', bottom: 0, left:  0,  top:    undefined, right: undefined, dx: '-105%', dy:  '105%' },
        { k: 'br', bottom: 0, right: 0,  top:    undefined, left:  undefined, dx:  '105%', dy:  '105%' },
      ] as const).map(({ k, top, right, bottom, left, dx, dy }) => (
        <motion.div
          key={k}
          className="pointer-events-none fixed"
          style={{ top, right, bottom, left, width: '50vw', height: '50vh', background: cardTx.color, zIndex: 200 }}
          initial={{ x: 0, y: 0 }}
          animate={{ x: dx, y: dy }}
          transition={{ duration: 0.44, ease: [0.55, 0, 0.1, 1] }}
        />
      ))}

      {/* ── Landing ── */}
      <AnimatePresence>
        {view === 'landing' && (
          <motion.div
            key="landing"
            className="absolute inset-0 z-10"
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, transition: { duration: 0.08 } }}
            transition={{ duration: 0.32, ease: 'easeOut' }}
          >
            <Landing onPickMode={enterMode} onOpenSettings={() => setModal(true)} onOpenHelp={() => setHelp(true)} history={history} />
          </motion.div>
        )}
      </AnimatePresence>

      {/* ── Workspace ── */}
      <AnimatePresence>
        {view === 'workspace' && (
          <motion.div
            key="workspace"
            className="absolute inset-0 flex"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0, transition: { duration: 0.2 } }}
            transition={{ duration: 0.18, delay: 0 }}
          >
            {/* Left sidebar slides in from the left */}
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.35, delay: 0.08, ease: [0.22, 0, 0.1, 1] }}
            >
              <LeftSidebar
                mode={mode} setMode={m => {
                  setMode(m)
                  setActiveTab(m === 'partb' ? 'bugs' : m === 'partc' ? 'sbfl' : 'requirements')
                  setStages(getInitialStages(m))
                }}
                onHome={goHome} onSettings={() => setModal(true)} onHelp={() => setHelp(true)} settings={settings}
                repoUrl={repoUrl} setRepoUrl={setRepoUrl}
                branch={branch} setBranch={setBranch}
                files={files} selected={selected} toggleFile={toggleFile}
                discover={discover} discovering={discovering} repoName={repoName}
                partAMode={partAMode} setPartAMode={setPartAMode}
                srsFile={srsFile} setSrsFile={setSrsFile}
                readme={readme} setReadme={setReadme}
                repoNameA={repoNameA} setRepoNameA={setRepoNameA}
                problems={problems} setProblems={setProblems}
                expected={expected} setExpected={setExpected}
                edgeCases={edgeCases} setEdgeCases={setEdgeCases}
                threshold={threshold} setThreshold={setThreshold}
                combSrsFile={combSrsFile} setCombSrsFile={setCombSrsFile}
                combReadme={combReadme} setCombReadme={setCombReadme}
                combTargetFile={combTargetFile} setCombTargetFile={setCombTargetFile}
                combImportPath={combImportPath} setCombImportPath={setCombImportPath}
                partcSourceIdx={partcSourceIdx} setPartcSourceIdx={setPartcSourceIdx}
                partcTestIdx={partcTestIdx} setPartcTestIdx={setPartcTestIdx}
                status={status} startRun={startRun}
              />
            </motion.div>

            {/* Main content rises from below */}
            <motion.div
              className="flex-1 min-w-0 flex flex-col"
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.35, delay: 0.14, ease: [0.22, 0, 0.1, 1] }}
            >
              <MainContent
                mode={mode} results={results} logs={logs} streamedCode={streamedCode}
                aiStatus={aiStatus} status={status} progress={progress}
                stages={stages} elapsedSec={elapsedSec}
                activeTab={activeTab} setActiveTab={setActiveTab}
                selectedBug={selectedBug} setSelectedBug={setSelectedBug}
                selectedCoverageFile={selectedCovFile} setSelectedCoverageFile={setSelectedCovFile}
                reviewRequest={reviewRequest} onReviewDecision={handleReview}
                planRequest={planRequest} onPlanDecision={handlePlan}
                requirements={requirements} scenarios={scenarios} features={features}
                prepassResults={prepassResults} staleDetails={staleDetails}
              />
            </motion.div>

            {/* Right sidebar slides in from the right */}
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.35, delay: 0.2, ease: [0.22, 0, 0.1, 1] }}
            >
              <RightSidebar mode={mode} results={results} logs={logs} partAStats={partAStatsForSidebar} prepassResults={prepassResults} />
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {isModalOpen && (
        <ProjectSettingsModal onClose={() => setModal(false)} settings={settings} onSave={setSettings} />
      )}

      {isHelpOpen && (
        <HelpModal onClose={() => setHelp(false)} />
      )}
    </div>
  )
}
