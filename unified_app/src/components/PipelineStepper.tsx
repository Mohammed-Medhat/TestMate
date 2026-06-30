import { X, Loader } from 'lucide-react'
import { motion, AnimatePresence } from 'framer-motion'
import type { Mode } from '../types'
import { MODE_COLORS } from '../theme'


export type StageStatus = 'pending' | 'active' | 'done' | 'error'

export interface Stage {
  id: string
  label: string
  status: StageStatus
}

// ── Per-mode stage templates ─────────────────────────────────────────────
// Combined stages expand based on quality_mode — base always the same,
// balanced adds gap analysis, best adds gap-fill too.
const COMBINED_STAGES_FAST: Stage[] = [
  { id: 'srs',      label: 'SRS Extract',    status: 'pending' },
  { id: 'readme',   label: 'README Extract', status: 'pending' },
  { id: 'discover', label: 'Discover Files', status: 'pending' },
  { id: 'partb',    label: 'Test Gen',       status: 'pending' },
  { id: 'repair',   label: 'Bug Fix',        status: 'pending' },
  { id: 'done',     label: 'Done',           status: 'pending' },
]
const COMBINED_STAGES_BALANCED: Stage[] = [
  ...COMBINED_STAGES_FAST.filter(s => s.id !== 'done'),
  { id: 'gap_analysis', label: 'Gap Analysis', status: 'pending' },
  { id: 'done',         label: 'Done',         status: 'pending' },
]
const COMBINED_STAGES_BEST: Stage[] = [
  ...COMBINED_STAGES_BALANCED.filter(s => s.id !== 'done'),
  { id: 'gap_fill', label: 'Gap Fill',   status: 'pending' },
  { id: 'repair',   label: 'Bug Repair', status: 'pending' },
  { id: 'done',     label: 'Done',       status: 'pending' },
]

const STAGES_BY_MODE: Record<Mode, Stage[]> = {
  combined: COMBINED_STAGES_FAST,   // default; caller overrides via getInitialStages
  parta: [
    { id: 'read',     label: 'Read Doc',     status: 'pending' },
    { id: 'segment',  label: 'Segment',      status: 'pending' },
    { id: 'align',    label: 'Align',        status: 'pending' },
    { id: 'scenarios',label: 'Scenarios',    status: 'pending' },
    { id: 'done',     label: 'Done',         status: 'pending' },
  ],
  partb: [
    { id: 'discover', label: 'Discover Files', status: 'pending' },
    { id: 'model',    label: 'Load Model',     status: 'pending' },
    { id: 'gen',      label: 'Generate Tests', status: 'pending' },
    { id: 'eval',     label: 'Evaluate',       status: 'pending' },
    { id: 'done',     label: 'Done',           status: 'pending' },
  ],
  partc: [
    { id: 'discover', label: 'Discover Files', status: 'pending' },
    { id: 'tests',    label: 'Run Tests',      status: 'pending' },
    { id: 'sbfl',     label: 'SBFL Localize',  status: 'pending' },
    { id: 'repair',   label: 'LLM Repair',     status: 'pending' },
    { id: 'verify',   label: 'Re-test',        status: 'pending' },
    { id: 'done',     label: 'Done',           status: 'pending' },
  ],
}

export function getInitialStages(mode: Mode, qualityMode?: string): Stage[] {
  if (mode === 'combined') {
    const template = qualityMode === 'best'     ? COMBINED_STAGES_BEST
                   : qualityMode === 'balanced' ? COMBINED_STAGES_BALANCED
                   : COMBINED_STAGES_FAST
    return template.map(s => ({ ...s }))
  }
  return STAGES_BY_MODE[mode].map(s => ({ ...s }))
}

// ── Stage detector: parse a log line and return which stage to activate ──
export function detectStageFromLog(mode: Mode, message: string): string | null {
  const m = message.toLowerCase()

  if (mode === 'combined') {
    if (m.includes('phase 1') || m.includes('srs extraction'))     return 'srs'
    if (m.includes('readme extraction'))                            return 'readme'
    if (m.includes('discover') || m.includes('project-level'))     return 'discover'
    if (m.includes('phase 2') || m.includes('test gen') || m.includes('autonomous'))
                                                                    return 'partb'
    if (m.includes('phase 3') || m.includes('gap analysis') || m.includes('coverage gap'))
                                                                    return 'gap_analysis'
    if (m.includes('phase 4') || m.includes('gap-fill') || m.includes('gap fill'))
                                                                    return 'gap_fill'
    if (m.includes('auto-repair') || m.includes('🔧 auto-repair') || m.includes('bug repair starting'))
                                                                    return 'repair'
    if (m.includes('phase 2 complete') || m.includes('✅ pipeline') || m.includes('repair complete'))
                                                                    return 'done'
  }
  if (mode === 'parta') {
    if (m.includes('processing') || m.includes('reading pdf') || m.includes('reading docx')) return 'read'
    if (m.includes('segment'))     return 'segment'
    if (m.includes('align'))       return 'align'
    if (m.includes('scenario') || m.includes('extracting readme') || m.includes('generating')) return 'scenarios'
    if (m.includes('done') || m.includes('✅')) return 'done'
  }
  if (mode === 'partb') {
    if (m.includes('discover') || m.includes('scanning') || m.includes('found'))      return 'discover'
    if (m.includes('loading model') || m.includes('loading qwen'))                    return 'model'
    if (m.includes('generating') || m.includes('autonomous') || m.includes('test gen')) return 'gen'
    if (m.includes('evaluat') || m.includes('coverage') || m.includes('pytest'))      return 'eval'
    if (m.includes('complete') || m.includes('✅'))                                    return 'done'
  }
  if (mode === 'partc') {
    if (m.includes('discover') || m.includes('workspace'))                    return 'discover'
    if (m.includes('step 1') || m.includes('running pytest') || m.includes('🧪')) return 'tests'
    if (m.includes('step 2') || m.includes('sbfl') || m.includes('🔍'))       return 'sbfl'
    if (m.includes('step 4') || m.includes('calling testmate') || m.includes('🤖')) return 'repair'
    if (m.includes('step 6') || m.includes('re-running') || m.includes('verify') || m.includes('🔁')) return 'verify'
    if (m.includes('success') || m.includes('🎉') || m.includes('apr') && m.includes('succeed'))      return 'done'
  }
  return null
}

// ── Promote stages: when stage X activates, all prior stages → done ─────
// The active stage shows as 'active' (spinner); all stages before it → 'done'.
// When 'done' fires, any skipped pending stages (e.g. repair if no bugs) → 'done'.
export function advanceStages(stages: Stage[], activeId: string): Stage[] {
  let hit = false
  return stages.map(s => {
    if (s.id === activeId) {
      hit = true
      return { ...s, status: s.id === 'done' ? 'done' : 'active' }
    }
    if (!hit) return { ...s, status: 'done' }  // stages before active → done
    return { ...s, status: 'pending' }          // stages after active → remain pending
  })
}

// ── Component ────────────────────────────────────────────────────────────
interface Props {
  stages: Stage[]
  mode?: Mode
  currentFile?: string
  progress?: { current: number; total: number }
  elapsedSec?: number
}

export default function PipelineStepper({ stages, mode = 'combined', currentFile, progress, elapsedSec }: Props) {
  const hex = MODE_COLORS[mode].hex
  const showProgress = progress && progress.total > 0
  return (
    <div className="bg-zinc-900 border-b border-zinc-800 px-4 py-2 flex items-center gap-2 overflow-x-auto shrink-0">
      {/* Stage chain */}
      <div className="flex items-center gap-1 flex-1">
        {stages.map((s, i) => (
          <div key={s.id} className="flex items-center gap-1">
            {i > 0 && (
              <motion.span
                animate={{ color: stages[i-1].status === 'done' ? 'rgba(16,185,129,0.6)' : '#3f3f46' }}
                transition={{ duration: 0.4 }}
                className="text-xs mx-1"
              >→</motion.span>
            )}
            <motion.div
              layout
              animate={{
                backgroundColor: s.status === 'done'   ? `${hex}1a`
                               : s.status === 'active' ? `${hex}33`
                               : s.status === 'error'  ? 'rgba(239,68,68,0.15)'
                               : 'transparent',
                color: s.status === 'done'   ? hex
                     : s.status === 'active' ? '#ffffff'
                     : s.status === 'error'  ? '#f87171'
                     : '#52525b',
              }}
              transition={{ duration: 0.35 }}
              className={`flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[11px] font-medium
                ${s.status === 'active' ? 'stage-active' : ''}`}
              style={s.status === 'active' ? { '--pulse-color': `${hex}73` } as React.CSSProperties : undefined}
            >
              <AnimatePresence mode="wait">
                {s.status === 'done' && (
                  <motion.svg key="check" width={11} height={11} viewBox="0 0 11 11"
                    initial={{ opacity: 0, scale: 0.5 }}
                    animate={{ opacity: 1, scale: 1 }}
                    exit={{ opacity: 0 }}
                    transition={{ duration: 0.25 }}
                  >
                    <polyline points="2,6 4.5,8.5 9,3" fill="none" stroke={hex} strokeWidth={1.8}
                      strokeLinecap="round" strokeLinejoin="round"
                      strokeDasharray={20} strokeDashoffset={0}
                      className="check-draw" />
                  </motion.svg>
                )}
                {s.status === 'active' && (
                  <motion.span key="spin" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
                    <Loader size={11} className="animate-spin" />
                  </motion.span>
                )}
                {s.status === 'error' && (
                  <motion.span key="x" initial={{ opacity: 0, scale: 0.5 }} animate={{ opacity: 1, scale: 1 }} exit={{ opacity: 0 }}>
                    <X size={11} className="text-red-400" />
                  </motion.span>
                )}
                {s.status === 'pending' && (
                  <motion.span key="dot" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
                    <span className="block w-[11px] h-[11px] rounded-full border border-zinc-700" />
                  </motion.span>
                )}
              </AnimatePresence>
              {s.label}
              {s.status === 'active' && (
                <span className="inline-flex items-center ml-0.5">
                  <span className="thinking-dot" /><span className="thinking-dot" /><span className="thinking-dot" />
                </span>
              )}
            </motion.div>
          </div>
        ))}
      </div>

      {/* Right side: progress + file + elapsed */}
      {(showProgress || currentFile || elapsedSec !== undefined) && (
        <div className="flex items-center gap-3 shrink-0 text-[11px] text-zinc-500">
          {currentFile && (
            <span className="font-mono text-emerald-400/80 truncate max-w-[200px]" title={currentFile}>
              {currentFile}
            </span>
          )}
          {showProgress && (
            <div className="flex items-center gap-2">
              <div className="w-24 h-1 bg-zinc-800 rounded-full overflow-hidden">
                <motion.div
                  className="h-full rounded-full"
                  style={{ background: hex }}
                  animate={{ width: `${Math.min(100, (progress!.current / progress!.total) * 100)}%` }}
                  transition={{ duration: 0.5, ease: 'easeOut' }}
                />
              </div>
              <span>{progress!.current}/{progress!.total}</span>
            </div>
          )}
          {elapsedSec !== undefined && elapsedSec > 0 && (
            <span>⏱ {formatElapsed(elapsedSec)}</span>
          )}
        </div>
      )}
    </div>
  )
}

function formatElapsed(sec: number): string {
  if (sec < 60) return `${Math.round(sec)}s`
  const m = Math.floor(sec / 60), s = Math.round(sec % 60)
  return `${m}m ${s}s`
}
