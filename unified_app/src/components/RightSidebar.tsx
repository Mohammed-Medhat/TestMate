import { useEffect, useRef, useState } from 'react'
import { motion } from 'framer-motion'
import type { Mode, PrepassSummary } from '../types'

/* ── Animated number that tweens from prev to next value ─── */
function AnimatedNumber({ value, suffix = '' }: { value: number; suffix?: string }) {
  const [display, setDisplay] = useState(value)
  const rafRef = useRef<number>(0)

  useEffect(() => {
    const from = display
    const to   = value
    if (from === to) return
    const dur = 600
    const start = performance.now()
    const tick = (now: number) => {
      const t = Math.min((now - start) / dur, 1)
      const ease = 1 - (1 - t) ** 3         // cubic ease-out
      setDisplay(Math.round(from + (to - from) * ease))
      if (t < 1) rafRef.current = requestAnimationFrame(tick)
    }
    rafRef.current = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(rafRef.current)
  }, [value])                                // eslint-disable-line react-hooks/exhaustive-deps

  return <>{display}{suffix}</>
}

/* ── Circular SVG ring ──────────────────────────────────────── */
function CircularProgress({ percentage = 0, color, size = 60 }: {
  percentage?: number; color: string; size?: number
}) {
  const sw = 6, r = (size - sw) / 2, circ = r * 2 * Math.PI
  const offset = circ - (Math.min(percentage, 100) / 100) * circ
  return (
    <svg width={size} height={size} className="transform -rotate-90">
      <circle cx={size/2} cy={size/2} r={r} stroke="#27272a" strokeWidth={sw} fill="none" />
      <circle cx={size/2} cy={size/2} r={r} stroke={color} strokeWidth={sw}
        fill="none" strokeDasharray={circ} strokeDashoffset={offset} strokeLinecap="round"
        style={{ transition: 'stroke-dashoffset 0.8s cubic-bezier(0.4,0,0.2,1)' }} />
    </svg>
  )
}

/* ── Metric card with animated number + hero first card ─────── */
function MetricCard({ label, value, subValue, percentage, color, hero = false }: {
  label: string; value: string; subValue?: string; percentage?: number; color: string; hero?: boolean
}) {
  const numericVal = parseFloat(value)
  const isNumeric  = !isNaN(numericVal) && isFinite(numericVal)
  const suffix     = value.replace(String(Math.round(numericVal)), '').trim()

  return (
    <motion.div
      layout
      className={`bg-zinc-900 rounded-lg flex flex-col items-center ${hero ? 'p-5' : 'p-4'}`}
    >
      {percentage !== undefined ? (
        <div className="relative">
          <CircularProgress percentage={percentage} color={color} size={hero ? 72 : 60} />
          <div className="absolute inset-0 flex items-center justify-center">
            <span className={`font-medium ${hero ? 'text-sm' : 'text-xs'}`} style={{ color }}>
              <AnimatedNumber value={percentage} suffix="%" />
            </span>
          </div>
        </div>
      ) : (
        <div
          className={`flex items-center justify-center rounded-full border-2 border-zinc-700 ${hero ? 'w-[72px] h-[72px]' : 'w-[60px] h-[60px]'}`}
          style={{ borderColor: color + '55' }}
        >
          <span className={`font-bold ${hero ? 'text-xl' : 'text-base'}`} style={{ color }}>
            {isNumeric ? Math.round(numericVal) : '—'}
          </span>
        </div>
      )}
      <p className="text-xs text-zinc-400 mt-3">{label}</p>
      {value && (
        <p className={`font-semibold text-zinc-100 mt-1 ${hero ? 'text-2xl' : 'text-lg'}`}>
          {isNumeric && suffix === '%'
            ? <><AnimatedNumber value={numericVal} />{suffix}</>
            : value}
        </p>
      )}
      {subValue && <p className="text-xs text-zinc-500 text-center">{subValue}</p>}
    </motion.div>
  )
}

/* ── Log filter input ───────────────────────────────────────── */
function AgentTrace({ logs }: { logs: string[] }) {
  const [filter, setFilter] = useState('')
  const scrollRef = useRef<HTMLDivElement>(null)

  const filtered = logs
    .filter(l => l && !l.startsWith('─') && !l.startsWith('═'))
    .filter(l => !filter || l.toLowerCase().includes(filter.toLowerCase()))
    .slice(-30)

  // Scroll the overflow container itself — never scrollIntoView (which scrolls the whole page)
  useEffect(() => {
    const el = scrollRef.current
    if (el) el.scrollTop = el.scrollHeight
  }, [filtered.length])

  if (logs.length === 0) return null
  return (
    <div className="border-t border-zinc-800 flex flex-col" style={{ maxHeight: 220 }}>
      <div className="p-2 border-b border-zinc-800 flex items-center gap-2">
        <h3 className="text-xs font-semibold text-zinc-400 flex-1">Agent Trace</h3>
        <input
          value={filter}
          onChange={e => setFilter(e.target.value)}
          placeholder="filter…"
          className="w-20 bg-zinc-950 border border-zinc-800 rounded px-1.5 py-0.5 text-[10px] text-zinc-300 placeholder-zinc-600 outline-none focus:border-emerald-500/50"
        />
      </div>
      <div ref={scrollRef} className="flex-1 overflow-y-auto px-3 py-2 space-y-0.5">
        {filtered.map((log, i) => (
          <div key={i} className={`text-[0.6rem] truncate transition-colors ${
            log.startsWith('❌') ? 'text-red-400'
            : log.startsWith('✅') ? 'text-emerald-400'
            : log.startsWith('🔧') ? 'text-amber-400'
            : 'text-zinc-500'
          }`} title={log}>
            {log}
          </div>
        ))}
      </div>
    </div>
  )
}

interface Props {
  mode: Mode
  results: any
  logs: string[]
  partAStats?: { total: number; requirements: number; scenarios: number; matchScoreAvg: number }
  prepassResults?: PrepassSummary[]
}

export default function RightSidebar({ mode, results, logs, partAStats, prepassResults = [] }: Props) {
  const cov       = results?.line_coverage  ?? 0
  const mutScore  = results?.mutation_score ?? 0
  const bugsFound = results?.bugs_found     ?? 0

  const metrics = (() => {
    if (mode === 'partb') {
      const totalStale    = prepassResults.reduce((s, p) => s + p.stale_tests_fixed, 0)
      const totalRealBugs = prepassResults.reduce((s, p) => s + p.real_bugs_found, 0)
      const repairs       = (results?.repairs ?? []) as any[]
      const repairsOk     = repairs.filter((r: any) => r.success).length
      return [
        { label: 'Coverage',       value: `${typeof cov === 'number' ? cov.toFixed(1) : cov}%`,  percentage: Math.round(cov),      color: '#10b981', hero: true },
        { label: 'Mutation Score', value: `${typeof mutScore === 'number' ? mutScore.toFixed(0) : mutScore}%`, percentage: Math.round(mutScore), color: '#f59e0b' },
        { label: 'Bugs Detected',  value: (bugsFound + totalRealBugs) > 0 ? `${bugsFound + totalRealBugs} Found` : 'None', color: '#ef4444' },
        { label: 'Stale Tests',    value: totalStale > 0 ? `${totalStale} Fixed` : 'None', color: '#f59e0b' },
        { label: 'Auto-Repairs',   value: repairs.length > 0 ? `${repairsOk}/${repairs.length}` : '—',
          subValue: repairs.length > 0 ? (repairsOk === repairs.length ? 'All fixed' : 'Some failed') : '',
          percentage: repairs.length > 0 ? Math.round(repairsOk / repairs.length * 100) : 0, color: '#10b981' },
      ]
    }
    if (mode === 'parta') {
      const req   = partAStats?.requirements  ?? 0
      const total = partAStats?.total         ?? 0
      const scen  = partAStats?.scenarios     ?? 0
      const pct   = total > 0 ? Math.round((req / total) * 100) : 0
      const scoreAvg = partAStats?.matchScoreAvg ?? 0
      return [
        { label: 'Requirements',  value: String(req),  percentage: pct, color: '#10b981', hero: true },
        { label: 'Non-Req',       value: String(total - req), percentage: total > 0 ? 100 - pct : 0, color: '#ef4444' },
        { label: 'Scenarios',     value: String(scen), color: '#6366f1' },
        { label: 'Match Score',   value: `${Math.round(scoreAvg * 100)}%`, percentage: Math.round(scoreAvg * 100), color: '#f59e0b' },
      ]
    }
    // combined
    const req        = results?.part_a?.stats?.requirements ?? 0
    const scen       = (results?.part_a?.scenarios ?? []).length
    const partBFiles: any[] = Array.isArray(results?.part_b) ? results.part_b : []
    const covTotal   = partBFiles.reduce((s: number, f: any) => s + (f.srs_coverage?.total   ?? 0), 0)
    const covCovered = partBFiles.reduce((s: number, f: any) => s + (f.srs_coverage?.covered ?? 0), 0)
    const srsCovPct  = covTotal > 0 ? Math.round(covCovered / covTotal * 100) : 0
    const filesOk    = partBFiles.filter((f: any) => f.success).length
    const filesTotal = partBFiles.length
    const partCItems: any[] = Array.isArray(results?.part_c) ? results.part_c : []
    const repairsOk  = partCItems.filter((r: any) => r.repair_success || r.success).length
    const totalRealBugs = prepassResults.reduce((s, p) => s + p.real_bugs_found, 0)
    // full circle always: red = bugs found, green = clean, undefined = not run yet
    const bugPct = filesTotal > 0 ? 100 : undefined
    return [
      // Hero gauge: SRS coverage when available, else Test Files success rate
      covTotal > 0
        ? { label: 'SRS Coverage', value: `${covCovered}/${covTotal}`,
            subValue: `${srsCovPct}% covered`, percentage: srsCovPct, color: '#f59e0b', hero: true }
        : { label: 'Test Files', value: filesTotal > 0 ? `${filesOk}/${filesTotal}` : '—',
            subValue: filesTotal > 0 ? `${Math.round(filesOk/filesTotal*100)}% succeeded` : 'Run pipeline first',
            percentage: filesTotal > 0 ? Math.round(filesOk/filesTotal*100) : 0, color: '#6366f1', hero: true },
      { label: 'Requirements', value: req > 0 ? String(req) : '—', color: '#10b981',
        subValue: req > 0 ? (scen > 0 ? `${scen} scenarios` : 'extracted') : 'No SRS uploaded' },
      ...(covTotal > 0 ? [{ label: 'Test Files', value: `${filesOk}/${filesTotal}`,
        percentage: filesTotal > 0 ? Math.round(filesOk/filesTotal*100) : 0, color: '#6366f1' }] : []),
      { label: 'Bugs Detected',
        value: totalRealBugs > 0 ? `${totalRealBugs} Found` : filesTotal > 0 ? 'None' : '—',
        percentage: bugPct, color: totalRealBugs > 0 ? '#ef4444' : '#22c55e',
        subValue: totalRealBugs > 0 ? 'Pre-pass confirmed' : filesTotal > 0 ? 'Code looks clean' : '' },
      ...(partCItems.length > 0 ? [{
        label: 'Auto-Repaired', value: `${repairsOk}/${partCItems.length}`,
        subValue: repairsOk === partCItems.length ? 'All patched ✅' : 'Partial fix',
        percentage: partCItems.length > 0 ? Math.round(repairsOk / partCItems.length * 100) : 0,
        color: '#10b981', hero: false,
      }] : []),
    ]
  })()

  return (
    <div className="w-64 bg-zinc-900 border-l border-zinc-800 flex flex-col">
      <div className="p-4 border-b border-zinc-800">
        <h2 className="text-sm font-semibold text-zinc-300">Global Metrics</h2>
      </div>
      <div className="overflow-y-auto p-4 space-y-4 flex-1">
        {metrics.map((m, i) => (
          <MetricCard key={m.label} label={m.label} value={m.value}
            subValue={(m as any).subValue} percentage={m.percentage} color={m.color}
            hero={(m as any).hero} />
        ))}
      </div>
      <AgentTrace logs={logs} />
    </div>
  )
}
