import type { Scenario } from '../types'

interface Props { scenarios: Scenario[] }

const TYPE_CLS: Record<string, string> = {
  positive: 'bg-emerald-500/15 text-emerald-400 border border-emerald-500/30',
  negative: 'bg-red-500/15 text-red-400 border border-red-500/30',
  edge:     'bg-blue-500/15 text-blue-400 border border-blue-500/30',
  boundary: 'bg-yellow-500/15 text-yellow-400 border border-yellow-500/30'
}

const PRIO_CLS: Record<string, string> = {
  high:   'bg-red-500/10 text-red-400',
  medium: 'bg-zinc-700/60 text-zinc-400',
  low:    'bg-zinc-800/60 text-zinc-500'
}

export default function ScenariosList({ scenarios }: Props) {
  if (!scenarios.length)
    return <div className="flex items-center justify-center h-32 text-zinc-500 text-sm">No scenarios yet</div>

  return (
    <div className="space-y-2 p-3">
      {scenarios.map((s, i) => (
        <div key={i} className="rounded-xl border border-zinc-800 bg-zinc-900 p-4">
          <div className="flex items-start gap-2 mb-2">
            <span className="font-medium text-zinc-100 text-sm flex-1">{s.name}</span>
            <span className={`shrink-0 rounded px-1.5 py-0.5 text-[10px] font-bold ${TYPE_CLS[s.type] ?? ''}`}>
              {s.type}
            </span>
            <span className={`shrink-0 rounded px-1.5 py-0.5 text-[10px] font-bold ${PRIO_CLS[s.priority] ?? ''}`}>
              {s.priority}
            </span>
          </div>
          <p className="text-xs text-zinc-400 leading-relaxed mb-1">{s.description}</p>
          {s.expected_result && (
            <p className="text-xs text-emerald-400/80 leading-relaxed">✓ {s.expected_result}</p>
          )}
        </div>
      ))}
    </div>
  )
}
