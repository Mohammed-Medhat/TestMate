import type { Requirement } from '../types'

interface Props { requirements: Requirement[] }

const LABEL: Record<number, { text: string; cls: string }> = {
  1:  { text: 'REQ',     cls: 'bg-emerald-500/15 text-emerald-400 border border-emerald-500/30' },
  0:  { text: 'NON-REQ', cls: 'bg-red-500/15 text-red-400 border border-red-500/30' },
  [-1]: { text: '?',     cls: 'bg-zinc-700/50 text-zinc-400 border border-zinc-700' }
}

export default function RequirementsList({ requirements }: Props) {
  if (!requirements.length)
    return <div className="flex items-center justify-center h-32 text-zinc-500 text-sm">No requirements yet</div>

  return (
    <div className="divide-y divide-zinc-800/60">
      {requirements.slice(0, 300).map((r, i) => {
        const lbl = LABEL[r.label] ?? LABEL[-1]
        return (
          <div key={i} className="flex items-start gap-3 px-4 py-2.5 hover:bg-zinc-900/60 transition-colors">
            <span className={`shrink-0 mt-0.5 rounded px-1.5 py-0.5 text-[10px] font-bold ${lbl.cls}`}>
              {lbl.text}
            </span>
            <span className="text-xs text-zinc-300 leading-relaxed">{r.text}</span>
            {r.score > 0 && (
              <span className="shrink-0 ml-auto text-[10px] text-zinc-600">{Math.round(r.score * 100)}%</span>
            )}
          </div>
        )
      })}
      {requirements.length > 300 && (
        <div className="px-4 py-2 text-xs text-zinc-500 text-center">
          … and {requirements.length - 300} more
        </div>
      )}
    </div>
  )
}
