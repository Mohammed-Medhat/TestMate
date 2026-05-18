import { useState, useEffect } from 'react'
import { Settings, Clock, CheckCircle, XCircle, Link2, FileText, Bot } from 'lucide-react'
import type { Mode, HistoryItem } from '../types'

// ── Typewriter hook ──────────────────────────────────────────────────────────
function useTypewriter(text: string, speed = 40, startDelay = 400) {
  const [displayed, setDisplayed] = useState('')
  const [done, setDone] = useState(false)

  useEffect(() => {
    setDisplayed('')
    setDone(false)
    let i = 0
    const start = setTimeout(() => {
      const id = setInterval(() => {
        i++
        setDisplayed(text.slice(0, i))
        if (i >= text.length) { clearInterval(id); setDone(true) }
      }, speed)
      return () => clearInterval(id)
    }, startDelay)
    return () => clearTimeout(start)
  }, [text, speed, startDelay])

  return { displayed, done }
}

// ── Mode card config ─────────────────────────────────────────────────────────
const CARDS: {
  mode: Mode; icon: string; title: string; desc: string
  border: string; glow: string; iconBg: string; accent: string; textHover: string
}[] = [
  {
    mode: 'combined', icon: '🔗', title: 'Full Pipeline',
    desc: 'Upload an SRS document and target code file. Part A extracts requirements, Part B generates test cases — all in one automated flow.',
    border: 'border-zinc-800 hover:border-emerald-500/60',
    glow: 'hover:shadow-[0_8px_32px_rgba(16,185,129,0.18)]',
    iconBg: 'bg-emerald-500/10', accent: '#10b981', textHover: 'group-hover:text-emerald-400',
  },
  {
    mode: 'parta', icon: '📄', title: 'Part A Only',
    desc: 'Extract and classify requirements from SRS documents or README files. Outputs labeled requirement lists and test scenarios.',
    border: 'border-zinc-800 hover:border-blue-500/60',
    glow: 'hover:shadow-[0_8px_32px_rgba(96,165,250,0.18)]',
    iconBg: 'bg-blue-500/10', accent: '#60a5fa', textHover: 'group-hover:text-blue-400',
  },
  {
    mode: 'partb', icon: '🤖', title: 'Part B Only',
    desc: 'AI-powered test generation for any Python repository. Discovers testable files, generates pytest suites, and reports coverage & bugs.',
    border: 'border-zinc-800 hover:border-violet-500/60',
    glow: 'hover:shadow-[0_8px_32px_rgba(167,139,250,0.18)]',
    iconBg: 'bg-violet-500/10', accent: '#a78bfa', textHover: 'group-hover:text-violet-400',
  },
]

const MODE_ICON: Record<Mode, React.ReactNode> = {
  combined: <Link2 size={11} />,
  parta:    <FileText size={11} />,
  partb:    <Bot size={11} />,
}

const MODE_LABEL: Record<Mode, string> = {
  combined: 'Full Pipeline',
  parta:    'Part A',
  partb:    'Part B',
}

// ── History row ──────────────────────────────────────────────────────────────
function HistoryRow({ item }: { item: HistoryItem }) {
  const date = new Date(item.timestamp)
  const when = date.toLocaleString(undefined, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })
  return (
    <div className="flex items-center gap-3 px-3 py-2 rounded-lg bg-zinc-900/70 border border-zinc-800 hover:border-zinc-700 transition-colors backdrop-blur-sm">
      {item.status === 'success'
        ? <CheckCircle size={13} className="text-emerald-400 shrink-0" />
        : <XCircle size={13} className="text-red-400 shrink-0" />
      }
      <div className="flex-1 min-w-0">
        <p className="text-xs text-zinc-200 truncate">{item.summary}</p>
        <div className="flex items-center gap-1.5 mt-0.5">
          <span className="text-[10px] text-zinc-500">{when}</span>
          <span className="text-[10px] text-zinc-700">·</span>
          <span className="flex items-center gap-1 text-[10px] text-zinc-500">
            {MODE_ICON[item.mode]}{MODE_LABEL[item.mode]}
          </span>
        </div>
      </div>
    </div>
  )
}

// ── Main landing component ───────────────────────────────────────────────────
interface Props {
  onPickMode: (mode: Mode) => void
  onOpenSettings: () => void
  history: HistoryItem[]
}

export default function Landing({ onPickMode, onOpenSettings, history }: Props) {
  const TAGLINE = 'AI-Powered Test Generation Platform'
  const { displayed, done } = useTypewriter(TAGLINE, 38, 600)
  const [pickedMode, setPickedMode] = useState<Mode | null>(null)
  const [leaving, setLeaving] = useState(false)

  const handlePick = (mode: Mode) => {
    if (pickedMode) return                  // already animating
    setPickedMode(mode)
    // give the card-expand animation time to play, then trigger landing fade-out + parent transition
    setTimeout(() => setLeaving(true), 250)
    setTimeout(() => onPickMode(mode), 600)  // total ~600ms before workspace mounts
  }

  return (
    <div className={`relative h-full w-full overflow-y-auto bg-zinc-950 landing-entering ${leaving ? 'landing-leaving' : ''}`}>

      {/* ── Animated dot-grid background ── */}
      <div className="pointer-events-none absolute inset-0 landing-grid opacity-40" />

      {/* ── Floating gradient orbs ── */}
      <div className="pointer-events-none absolute inset-0 overflow-hidden">
        <div className="orb-1 absolute top-[8%] left-[12%] w-[380px] h-[380px] rounded-full bg-emerald-500/8 blur-[90px]" />
        <div className="orb-2 absolute top-[30%] right-[8%]  w-[300px] h-[300px] rounded-full bg-violet-500/7  blur-[80px]" />
        <div className="orb-3 absolute bottom-[15%] left-[35%] w-[260px] h-[260px] rounded-full bg-cyan-500/6   blur-[70px]" />
      </div>

      {/* ── Content (above background) ── */}
      <div className="relative z-10">

        {/* Top bar */}
        <header className="flex items-center justify-between px-8 py-4">
          <div className="text-xs font-bold text-zinc-700 uppercase tracking-[0.2em]">TestMate</div>
          <button
            onClick={onOpenSettings}
            className="p-2 rounded-lg hover:bg-zinc-800/70 text-zinc-500 hover:text-zinc-200 transition-colors backdrop-blur-sm"
            title="Settings"
          >
            <Settings size={17} />
          </button>
        </header>

        {/* Hero */}
        <section className="text-center pt-10 pb-14 px-8">
          <div className="text-6xl mb-5 select-none" style={{ filter: 'drop-shadow(0 0 24px rgba(16,185,129,0.4))' }}>
            🧪
          </div>
          <h1 className="text-5xl font-extrabold tracking-tight mb-4">
            <span className="bg-gradient-to-r from-emerald-400 via-teal-300 to-emerald-500 bg-clip-text text-transparent">
              TestMate
            </span>
          </h1>

          {/* Typewriter tagline */}
          <p className="text-lg text-zinc-300 h-8 flex items-center justify-center">
            <span>{displayed}</span>
            {!done && <span className="typewriter-cursor" />}
          </p>
          <p className="text-sm text-zinc-600 mt-2 max-w-md mx-auto">
            From SRS documents to full test suites — powered by Qwen2.5-Coder and a 3-layer RAG agent.
          </p>
        </section>

        {/* Mode cards */}
        <section className="grid grid-cols-3 gap-5 max-w-5xl mx-auto px-8">
          {CARDS.map(card => {
            const isClicked = pickedMode === card.mode
            const isDimmed  = pickedMode !== null && !isClicked
            return (
            <button
              key={card.mode}
              onClick={() => handlePick(card.mode)}
              disabled={pickedMode !== null}
              className={`group mode-card relative rounded-2xl border bg-zinc-900/60 backdrop-blur-sm p-6 text-left ${card.border} ${card.glow} ${isClicked ? 'card-clicked' : ''} ${isDimmed ? 'card-dimmed' : ''}`}
            >
              {/* Glow accent line at top */}
              <div
                className="absolute top-0 left-8 right-8 h-px opacity-0 group-hover:opacity-100 transition-opacity duration-300 rounded-full"
                style={{ background: `linear-gradient(90deg, transparent, ${card.accent}, transparent)` }}
              />

              {/* Icon */}
              <div className={`card-icon w-14 h-14 rounded-xl ${card.iconBg} flex items-center justify-center text-3xl mb-4`}>
                {card.icon}
              </div>

              {/* Text */}
              <h3 className={`text-base font-semibold text-zinc-100 mb-2 transition-colors ${card.textHover}`}>
                {card.title}
              </h3>
              <p className="text-xs text-zinc-400 leading-relaxed">
                {card.desc}
              </p>

              {/* CTA */}
              <div className="mt-5 flex items-center gap-1 text-[11px] text-zinc-600 group-hover:text-zinc-400 transition-colors">
                <span>Open</span>
                <span className="transition-transform group-hover:translate-x-1 inline-block">→</span>
              </div>
            </button>
          )})}
        </section>

        {/* Recent runs */}
        <section className="max-w-5xl mx-auto mt-10 mb-14 px-8">
          <div className="flex items-center gap-2 mb-3">
            <Clock size={13} className="text-zinc-600" />
            <h2 className="text-[10px] font-bold text-zinc-600 uppercase tracking-[0.15em]">Recent Runs</h2>
          </div>

          {history.length === 0 ? (
            <p className="text-xs text-zinc-700 pl-1">No runs yet — pick a mode above to get started.</p>
          ) : (
            <div className="grid grid-cols-2 gap-2">
              {history.slice(0, 6).map(h => <HistoryRow key={h.id} item={h} />)}
            </div>
          )}
        </section>

      </div>
    </div>
  )
}
