import type { Mode } from '../types'

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
        style={{ transition: 'stroke-dashoffset 1s ease' }} />
    </svg>
  )
}

function MetricCard({ label, value, subValue, percentage, color }: {
  label: string; value: string; subValue?: string; percentage?: number; color: string
}) {
  return (
    <div className="bg-zinc-900 rounded-lg p-4 flex flex-col items-center">
      <div className="relative">
        <CircularProgress percentage={percentage} color={color} />
        <div className="absolute inset-0 flex items-center justify-center">
          {percentage !== undefined && (
            <span className="text-xs font-medium" style={{ color }}>{percentage}%</span>
          )}
        </div>
      </div>
      <p className="text-xs text-zinc-400 mt-3">{label}</p>
      {value && <p className="text-lg font-semibold text-zinc-100 mt-1">{value}</p>}
      {subValue && <p className="text-xs text-zinc-500">{subValue}</p>}
    </div>
  )
}

interface Props {
  mode: Mode
  results: any
  logs: string[]
  partAStats?: { total: number; requirements: number; scenarios: number; matchScoreAvg: number }
}

export default function RightSidebar({ mode, results, logs, partAStats }: Props) {
  const traceLogs = logs.filter(l => l && !l.startsWith('─') && !l.startsWith('═')).slice(-20)

  // PartB metrics (verbatim from PartB/gui)
  const cov       = results?.line_coverage     ?? 0
  const passRate  = results?.pass_rate         ?? 0
  const mutScore  = results?.mutation_score    ?? 0
  const totalTests= results?.total_tests       ?? 0
  const passedTests=results?.passed_tests      ?? 0
  const bugsFound = results?.bugs_found        ?? 0

  const metrics = (() => {
    if (mode === 'partb') {
      return [
        { label: 'Mutation Score', value: `${typeof mutScore === 'number' ? mutScore.toFixed(0) : mutScore}%`, percentage: Math.round(mutScore), color: '#f59e0b' },
        { label: 'Coverage',       value: `${typeof cov === 'number' ? cov.toFixed(1) : cov}%`,              percentage: Math.round(cov),      color: '#10b981' },
        { label: 'Security Issues',value: bugsFound > 0 ? `${bugsFound} Found` : 'None',                      color: '#ef4444' },
        { label: 'Test Results',   value: totalTests > 0 ? `${passedTests}/${totalTests}` : '—',
          subValue: totalTests > 0 ? `${typeof passRate === 'number' ? passRate.toFixed(0) : passRate}% pass rate` : '',
          percentage: Math.round(passRate), color: '#6366f1' },
      ]
    }
    if (mode === 'parta') {
      const req   = partAStats?.requirements  ?? 0
      const total = partAStats?.total         ?? 0
      const scen  = partAStats?.scenarios     ?? 0
      const pct   = total > 0 ? Math.round((req / total) * 100) : 0
      const scoreAvg = partAStats?.matchScoreAvg ?? 0
      return [
        { label: 'Requirements',   value: String(req),                   percentage: pct,                               color: '#10b981' },
        { label: 'Non-Req',        value: String(total - req),           percentage: total > 0 ? 100 - pct : 0,        color: '#ef4444' },
        { label: 'Scenarios',      value: String(scen),                  color: '#6366f1' },
        { label: 'Match Score',    value: `${Math.round(scoreAvg * 100)}%`, percentage: Math.round(scoreAvg * 100),    color: '#f59e0b' },
      ]
    }
    // combined
    const req    = results?.part_a?.stats?.requirements ?? 0
    const scen   = (results?.part_a?.scenarios ?? []).length
    const partBFiles: any[] = Array.isArray(results?.part_b) ? results.part_b : []
    // Aggregate SRS coverage across all processed files
    const covTotal   = partBFiles.reduce((s: number, f: any) => s + (f.srs_coverage?.total   ?? 0), 0)
    const covCovered = partBFiles.reduce((s: number, f: any) => s + (f.srs_coverage?.covered ?? 0), 0)
    const srsCovPct  = covTotal > 0 ? Math.round(covCovered / covTotal * 100) : 0
    const filesOk    = partBFiles.filter((f: any) => f.success).length
    const filesTotal = partBFiles.length
    return [
      { label: 'Requirements',  value: String(req),                     percentage: 0,        color: '#10b981' },
      { label: 'SRS Coverage',  value: covTotal > 0 ? `${covCovered}/${covTotal}` : '—',
        subValue: covTotal > 0 ? `${srsCovPct}% of requirements` : 'Run pipeline first',
        percentage: srsCovPct,  color: '#f59e0b' },
      { label: 'Test Files',    value: filesTotal > 0 ? `${filesOk}/${filesTotal}` : '—',
        percentage: filesTotal > 0 ? Math.round(filesOk/filesTotal*100) : 0, color: '#6366f1' },
      { label: 'Scenarios',     value: String(scen),                    color: '#a855f7' },
    ]
  })()

  return (
    <div className="w-64 bg-zinc-900 border-l border-zinc-800 flex flex-col">
      <div className="p-4 border-b border-zinc-800">
        <h2 className="text-sm font-semibold text-zinc-300">Global Metrics</h2>
      </div>
      <div className="overflow-y-auto p-4 space-y-4 flex-1">
        {metrics.map((m, i) => (
          <MetricCard key={i} label={m.label} value={m.value}
            subValue={(m as any).subValue} percentage={m.percentage} color={m.color} />
        ))}
      </div>

      {traceLogs.length > 0 && (
        <div className="border-t border-zinc-800 flex flex-col min-h-0" style={{ maxHeight: 200 }}>
          <div className="p-3 border-b border-zinc-800">
            <h3 className="text-xs font-semibold text-zinc-400">Agent Trace</h3>
          </div>
          <div className="flex-1 overflow-y-auto px-3 py-2 space-y-0.5">
            {traceLogs.map((log, i) => (
              <div key={i} className="text-[0.6rem] text-zinc-500 truncate" title={log}>
                {log.substring(0, 60)}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
