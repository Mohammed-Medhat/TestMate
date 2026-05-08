import React from 'react'

// SVG donut: r=36, cx=cy=45, viewBox="0 0 90 90"
const R = 36
const CIRC = 2 * Math.PI * R // ≈ 226.19

function DonutChart({ pct, color, label, sublabel }) {
  const filled = (pct / 100) * CIRC
  const gap = CIRC - filled

  return (
    <div className="relative mx-auto" style={{ width: 110, height: 110 }}>
      <svg width={110} height={110} viewBox="0 0 90 90">
        {/* Background track */}
        <circle cx="45" cy="45" r={R} fill="none" stroke="#27272a" strokeWidth="7" />
        {/* Progress arc */}
        {pct > 0 && (
          <circle
            cx="45"
            cy="45"
            r={R}
            fill="none"
            stroke={color}
            strokeWidth="7"
            strokeLinecap="round"
            strokeDasharray={`${filled} ${gap}`}
            transform="rotate(-90 45 45)"
          />
        )}
      </svg>
      {/* Center label */}
      <div className="absolute inset-0 flex flex-col items-center justify-center text-center">
        <span className="text-sm font-bold leading-none text-zinc-100">{label}</span>
        {sublabel && (
          <span className="mt-0.5 px-1 text-[9px] leading-tight text-zinc-500">{sublabel}</span>
        )}
      </div>
    </div>
  )
}

const METRICS = [
  {
    title: 'Mutation Score',
    pct: 0,
    color: '#52525b',
    label: '0%',
    sublabel: 'No data yet',
  },
  {
    title: 'Coverage',
    pct: 84.3,
    color: '#34d399',
    label: '84.3%',
    sublabel: 'Passing',
  },
  {
    title: 'Security Issues',
    pct: 33,
    color: '#dc2626',
    label: '2 Critical',
    sublabel: 'High: 3 | Med: 5',
  },
  {
    title: 'Test Exec Time',
    pct: 25,
    color: '#a855f7',
    label: '4m 12s',
    sublabel: '156 tests',
  },
]

export default function MetricsPanel() {
  return (
    <div className="flex w-72 flex-none flex-col border-l border-zinc-800 bg-zinc-900">
      <div className="border-b border-zinc-800 px-4 py-2.5">
        <span className="text-[10px] font-semibold uppercase tracking-widest text-zinc-500">
          Global Metrics
        </span>
      </div>

      <div className="flex flex-1 flex-col gap-5 overflow-y-auto p-5">
        {METRICS.map((m) => (
          <div key={m.title} className="flex flex-col items-center gap-2">
            <span className="text-[11px] font-medium text-zinc-400">{m.title}</span>
            <DonutChart {...m} />
          </div>
        ))}
      </div>
    </div>
  )
}
