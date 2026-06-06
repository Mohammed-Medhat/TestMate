import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { Settings, HelpCircle, Clock, CheckCircle, XCircle, Link2, FileText, Bot, Wrench } from 'lucide-react'
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

// ── Animated logo mark ───────────────────────────────────────────────────────
function LogoMark() {
  return (
    <div className="relative w-[88px] h-[88px] mx-auto mb-6 flex items-center justify-center select-none">
      {/* Rotating dashed outer ring */}
      <motion.div
        className="absolute inset-0 rounded-[20px] border border-dashed border-emerald-500/25"
        animate={{ rotate: 360 }}
        transition={{ duration: 28, repeat: Infinity, ease: 'linear' }}
      />
      {/* Static frosted inner box */}
      <div className="absolute inset-2 rounded-[14px] bg-emerald-500/8 border border-emerald-500/15 backdrop-blur-sm" />

      {/* Breathing corner brackets */}
      <motion.div
        className="absolute inset-0"
        animate={{ opacity: [0.35, 0.7, 0.35] }}
        transition={{ duration: 2.8, repeat: Infinity, ease: 'easeInOut' }}
      >
        {/* TL */}
        <span className="absolute top-0 left-0 w-3.5 h-3.5"
          style={{ borderTop: '2px solid #34d399', borderLeft: '2px solid #34d399', borderRadius: '6px 0 0 0' }} />
        {/* TR */}
        <span className="absolute top-0 right-0 w-3.5 h-3.5"
          style={{ borderTop: '2px solid #34d399', borderRight: '2px solid #34d399', borderRadius: '0 6px 0 0' }} />
        {/* BL */}
        <span className="absolute bottom-0 left-0 w-3.5 h-3.5"
          style={{ borderBottom: '2px solid #34d399', borderLeft: '2px solid #34d399', borderRadius: '0 0 0 6px' }} />
        {/* BR */}
        <span className="absolute bottom-0 right-0 w-3.5 h-3.5"
          style={{ borderBottom: '2px solid #34d399', borderRight: '2px solid #34d399', borderRadius: '0 0 6px 0' }} />
      </motion.div>

      {/* Code monogram */}
      <span className="relative z-10 font-mono text-[17px] font-bold text-emerald-400 tracking-tight">
        {'</>'}
      </span>
    </div>
  )
}

// ── Card config ──────────────────────────────────────────────────────────────
interface CardDef {
  mode: Mode
  num: string
  icon: React.ReactNode
  title: string
  desc: string
  tags: string[]
  accent: string
  accentHex: string      // used for ripple color
  topBar: string         // Tailwind bg class
  ring: string           // Tailwind hover:border class
  glow: string           // Tailwind hover:shadow class
  iconBg: string
  iconColor: string
  tagClass: string
  numClass: string
  titleHover: string
}

const CARDS: CardDef[] = [
  {
    mode: 'combined', num: '1', icon: <Link2 size={22} />,
    title: 'Full Pipeline',
    desc: 'SRS document to pytest suites in one automated flow. Part A extracts requirements, Part B generates self-correcting tests.',
    tags: ['PDF · DOCX', 'Qwen2.5-Coder', '3-layer RAG', 'Auto-Repair'],
    accent: '#10b981', accentHex: '#10b981',
    topBar:     'bg-emerald-500',
    ring:       'hover:border-emerald-500/50',
    glow:       'hover:shadow-[0_12px_40px_rgba(16,185,129,0.18)]',
    iconBg:     'bg-emerald-500/10',
    iconColor:  'text-emerald-400',
    tagClass:   'border-emerald-500/20 text-emerald-400/60 bg-emerald-500/5',
    numClass:   'text-emerald-500/25',
    titleHover: 'group-hover:text-emerald-300',
  },
  {
    mode: 'parta', num: '2', icon: <FileText size={22} />,
    title: 'Part A: Requirements',
    desc: 'Extract and classify requirements from SRS documents or README files using spaCy NLP and the PURE dataset.',
    tags: ['spaCy NLP', 'Fuzzy Match', 'PDF · DOCX', 'PURE Dataset'],
    accent: '#60a5fa', accentHex: '#3b82f6',
    topBar:     'bg-blue-400',
    ring:       'hover:border-blue-500/50',
    glow:       'hover:shadow-[0_12px_40px_rgba(96,165,250,0.18)]',
    iconBg:     'bg-blue-500/10',
    iconColor:  'text-blue-400',
    tagClass:   'border-blue-500/20 text-blue-400/60 bg-blue-500/5',
    numClass:   'text-blue-500/25',
    titleHover: 'group-hover:text-blue-300',
  },
  {
    mode: 'partb', num: '3', icon: <Bot size={22} />,
    title: 'Part B: Test Generator',
    desc: 'Self-correcting AI test generation. Discovers testable files, generates pytest suites, reports coverage and bugs.',
    tags: ['LoRA Fine-tune', 'FAISS RAG', 'Mutation Tests', 'HITL Review'],
    accent: '#a78bfa', accentHex: '#7c3aed',
    topBar:     'bg-violet-400',
    ring:       'hover:border-violet-500/50',
    glow:       'hover:shadow-[0_12px_40px_rgba(167,139,250,0.18)]',
    iconBg:     'bg-violet-500/10',
    iconColor:  'text-violet-400',
    tagClass:   'border-violet-500/20 text-violet-400/60 bg-violet-500/5',
    numClass:   'text-violet-500/25',
    titleHover: 'group-hover:text-violet-300',
  },
  {
    mode: 'partc', num: '4', icon: <Wrench size={22} />,
    title: 'Bug Fixer (APR)',
    desc: 'Automated program repair using SBFL fault localization. Pick a buggy source file — the LLM patches and re-verifies.',
    tags: ['Ochiai SBFL', 'LLM Repair', 'Re-verify', 'Diff View'],
    accent: '#fbbf24', accentHex: '#d97706',
    topBar:     'bg-amber-400',
    ring:       'hover:border-amber-500/50',
    glow:       'hover:shadow-[0_12px_40px_rgba(251,191,36,0.18)]',
    iconBg:     'bg-amber-500/10',
    iconColor:  'text-amber-400',
    tagClass:   'border-amber-500/20 text-amber-400/60 bg-amber-500/5',
    numClass:   'text-amber-500/25',
    titleHover: 'group-hover:text-amber-300',
  },
]

const MODE_ICON: Record<Mode, React.ReactNode> = {
  combined: <Link2 size={11} />,
  parta:    <FileText size={11} />,
  partb:    <Bot size={11} />,
  partc:    <Wrench size={11} />,
}
const MODE_LABEL: Record<Mode, string> = {
  combined: 'Full Pipeline', parta: 'Part A', partb: 'Part B', partc: 'Bug Fixer',
}

// ── History row ──────────────────────────────────────────────────────────────
function HistoryRow({ item, delay }: { item: HistoryItem; delay: number }) {
  const when = new Date(item.timestamp).toLocaleString(undefined, {
    month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit',
  })
  return (
    <motion.div
      initial={{ opacity: 0, x: -6 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ delay, duration: 0.25 }}
      className="flex items-center gap-3 px-3 py-2.5 rounded-xl bg-zinc-900/70 border border-zinc-800/80 hover:border-zinc-700 transition-colors"
    >
      {item.status === 'success'
        ? <CheckCircle size={13} className="text-emerald-400 shrink-0" />
        : <XCircle size={13} className="text-red-400 shrink-0" />}
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
    </motion.div>
  )
}

// ── Landing ──────────────────────────────────────────────────────────────────
interface Props {
  /** Called immediately on click with the card's bounding rect and accent hex */
  onPickMode: (mode: Mode, rect: DOMRect, color: string) => void
  onOpenSettings: () => void
  onOpenHelp: () => void
  history: HistoryItem[]
}

export default function Landing({ onPickMode, onOpenSettings, onOpenHelp, history }: Props) {
  const { displayed, done } = useTypewriter('AI-Powered Test Generation Platform', 36, 700)
  const [pickedMode, setPickedMode] = useState<Mode | null>(null)

  // Call parent immediately — App.tsx owns the card transition overlay and timing
  const handlePick = (mode: Mode, e: React.MouseEvent<HTMLButtonElement>) => {
    if (pickedMode) return
    const rect  = e.currentTarget.getBoundingClientRect()
    const color = CARDS.find(c => c.mode === mode)?.accentHex ?? '#10b981'
    setPickedMode(mode)
    onPickMode(mode, rect, color)
  }

  return (
    <div className="relative h-full w-full overflow-y-auto bg-zinc-950">

      {/* ── Dot grid ── */}
      <div className="pointer-events-none absolute inset-0 landing-grid opacity-25" />

      {/* ── Gradient orbs ── */}
      <div className="pointer-events-none absolute inset-0 overflow-hidden">
        <div className="orb-1 absolute top-[5%]  left-[8%]   w-[440px] h-[440px] rounded-full bg-emerald-500/6  blur-[110px]" />
        <div className="orb-2 absolute top-[35%] right-[4%]  w-[340px] h-[340px] rounded-full bg-violet-500/5   blur-[95px]"  />
        <div className="orb-3 absolute bottom-[8%] left-[38%] w-[300px] h-[300px] rounded-full bg-cyan-500/4    blur-[85px]"  />
      </div>

      {/* ── Page content ── */}
      <div className="relative z-10 flex flex-col">

        {/* Header */}
        <header className="flex items-center justify-between px-8 py-4">
          <div className="flex items-center gap-2">
            <span className="text-xs font-bold text-zinc-600 uppercase tracking-[0.22em]">TestMate</span>
            <span className="px-1.5 py-0.5 text-[9px] font-mono rounded border border-emerald-500/20 text-emerald-500/50 bg-emerald-500/5">
              v1.0
            </span>
          </div>
          <div className="flex items-center gap-1">
            <button onClick={onOpenHelp} aria-label="Help"
              className="p-2 rounded-lg hover:bg-zinc-800/70 text-zinc-500 hover:text-zinc-200 transition-colors focus-ring">
              <HelpCircle size={17} />
            </button>
            <button onClick={onOpenSettings} aria-label="Settings"
              className="p-2 rounded-lg hover:bg-zinc-800/70 text-zinc-500 hover:text-zinc-200 transition-colors focus-ring">
              <Settings size={17} />
            </button>
          </div>
        </header>

        {/* Hero */}
        <section className="text-center pt-8 pb-10 px-8">
          <LogoMark />

          <motion.h1
            className="text-5xl font-extrabold tracking-tight mb-4"
            initial={{ opacity: 0, y: 14 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.55, delay: 0.1 }}
          >
            <span className="bg-gradient-to-r from-emerald-400 via-teal-300 to-emerald-500 bg-clip-text text-transparent">
              TestMate
            </span>
          </motion.h1>

          <motion.p
            className="text-lg text-zinc-300 h-8 flex items-center justify-center"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.3 }}
          >
            <span>{displayed}</span>
            {!done && <span className="typewriter-cursor" />}
          </motion.p>

          <motion.p
            className="text-sm text-zinc-600 mt-2 max-w-md mx-auto"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.6 }}
          >
            From SRS documents to full test suites — powered by Qwen2.5-Coder and a 3-layer RAG agent.
          </motion.p>
        </section>

        {/* ── Mode cards ── */}
        <section className="grid grid-cols-2 gap-4 max-w-4xl mx-auto w-full px-8">
          {CARDS.map((card, i) => {
            const isClicked = pickedMode === card.mode
            const isDimmed  = pickedMode !== null && !isClicked

            return (
              <motion.button
                key={card.mode}
                onClick={e => handlePick(card.mode, e)}
                disabled={pickedMode !== null}
                // entrance stagger
                initial={{ opacity: 0, y: 22 }}
                animate={{
                  opacity: isDimmed ? 0.28 : 1,
                  y:       0,
                  scale:   isClicked ? 1.04 : 1,
                  filter:  isDimmed ? 'blur(1px)' : 'blur(0px)',
                }}
                transition={{
                  duration: pickedMode ? 0.28 : 0.4,
                  delay:    pickedMode ? 0 : 0.15 + i * 0.08,
                  ease:     'easeOut',
                }}
                whileHover={pickedMode ? {} : { y: -6, scale: 1.018 }}
                whileTap={pickedMode ? {} : { scale: 0.97 }}
                className={`group relative rounded-2xl border bg-zinc-900/55 backdrop-blur-sm text-left
                  overflow-hidden transition-[border-color,box-shadow] duration-300
                  border-zinc-800 ${card.ring} ${card.glow}
                  ${isClicked ? 'ring-1 ring-white/10' : ''}
                  focus-ring
                `}
              >
                {/* Top accent bar — always visible, brightens on hover */}
                <div className={`absolute top-0 inset-x-0 h-[2px] ${card.topBar} opacity-30 group-hover:opacity-70 transition-opacity duration-300`} />

                {/* Subtle inner glow on hover */}
                <div className="absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity duration-300 rounded-2xl"
                  style={{ background: `radial-gradient(ellipse at 50% 0%, ${card.accentHex}18 0%, transparent 70%)` }} />

                {/* Keyboard shortcut badge */}
                <div className={`absolute top-4 right-4 font-mono text-2xl font-extrabold ${card.numClass} select-none leading-none`}>
                  {card.num}
                </div>

                <div className="p-6 pt-7">
                  {/* Icon */}
                  <motion.div
                    className={`w-12 h-12 rounded-xl ${card.iconBg} flex items-center justify-center ${card.iconColor} mb-4`}
                    whileHover={{ scale: 1.12, rotate: -4 }}
                    transition={{ type: 'spring', stiffness: 300, damping: 18 }}
                  >
                    {card.icon}
                  </motion.div>

                  {/* Title */}
                  <h3 className={`text-sm font-semibold text-zinc-100 mb-2 transition-colors duration-200 ${card.titleHover}`}>
                    {card.title}
                  </h3>

                  {/* Description */}
                  <p className="text-[11px] text-zinc-500 leading-relaxed mb-4">
                    {card.desc}
                  </p>

                  {/* Capability tags */}
                  <div className="flex flex-wrap gap-1.5 mb-5">
                    {card.tags.map(tag => (
                      <span key={tag}
                        className={`px-2 py-0.5 text-[9.5px] rounded-full border font-medium ${card.tagClass}`}>
                        {tag}
                      </span>
                    ))}
                  </div>

                  {/* CTA row */}
                  <div className="flex items-center gap-1.5 text-[11px] text-zinc-600 group-hover:text-zinc-300 transition-colors duration-200">
                    <span>Open</span>
                    <motion.span
                      animate={pickedMode ? {} : { x: [0, 4, 0] }}
                      transition={{ duration: 1.6, repeat: Infinity, ease: 'easeInOut', repeatDelay: 0.4 }}
                    >
                      →
                    </motion.span>
                    <span className="ml-auto text-[9px] text-zinc-700 font-mono group-hover:text-zinc-500">
                      Press {card.num}
                    </span>
                  </div>
                </div>
              </motion.button>
            )
          })}
        </section>

        {/* ── Recent runs ── */}
        <section className="max-w-4xl mx-auto w-full mt-10 mb-14 px-8">
          {history.length > 0 && (
            <>
              <div className="flex items-center gap-2 mb-3">
                <Clock size={12} className="text-zinc-600" />
                <h2 className="text-[10px] font-bold text-zinc-600 uppercase tracking-[0.16em]">
                  Recent Runs
                </h2>
                <span className="ml-auto text-[10px] text-zinc-700">{history.length} run{history.length !== 1 ? 's' : ''}</span>
              </div>
              <div className="grid grid-cols-2 gap-2">
                {history.slice(0, 6).map((h, i) => (
                  <HistoryRow key={h.id} item={h} delay={i * 0.05} />
                ))}
              </div>
            </>
          )}
          {history.length === 0 && (
            <p className="text-xs text-zinc-700 text-center py-4">
              No runs yet — pick a mode above to get started.
            </p>
          )}
        </section>

      </div>
    </div>
  )
}
