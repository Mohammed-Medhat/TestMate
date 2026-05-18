import { X, Zap, Scale, Gem } from 'lucide-react'
import { useState } from 'react'
import type { RunSettings, QualityMode } from '../types'

interface ModalProps {
  onClose: () => void
  settings: RunSettings
  onSave: (s: RunSettings) => void
}

function Toggle({ label, desc, checked, onChange, color = 'emerald' }: {
  label: string; desc: string; checked: boolean; onChange: () => void; color?: string
}) {
  const bg = checked
    ? color === 'violet' ? 'bg-violet-500'
    : color === 'amber'  ? 'bg-amber-500'
    : 'bg-emerald-500'
    : 'bg-zinc-700'
  return (
    <div className="flex items-center justify-between p-3 bg-zinc-800/50 rounded-lg">
      <div>
        <p className="text-sm font-medium text-zinc-200">{label}</p>
        <p className="text-xs text-zinc-500">{desc}</p>
      </div>
      <button onClick={onChange} className={`relative w-11 h-6 rounded-full transition-colors ${bg}`}>
        <span className={`absolute top-1 w-4 h-4 bg-white rounded-full transition-transform ${checked ? 'left-6' : 'left-1'}`} />
      </button>
    </div>
  )
}

const QUALITY_TIERS: { id: QualityMode; icon: React.ReactNode; label: string; time: string; desc: string; color: string }[] = [
  { id: 'fast',     icon: <Zap size={14} />,   label: 'Fast',         time: '~2-3 min/file', desc: 'LoRA + plan. Quick iteration.',                color: 'zinc'    },
  { id: 'balanced', icon: <Scale size={14} />, label: 'Balanced',     time: '~4-5 min/file', desc: 'LoRA + plan + SRS gap analysis.',               color: 'emerald' },
  { id: 'best',     icon: <Gem size={14} />,   label: 'Best Quality', time: '~6-8 min/file', desc: 'LoRA + plan + gap analysis + gap-fill pass.',   color: 'amber'   },
]

export default function ProjectSettingsModal({ onClose, settings, onSave }: ModalProps) {
  const [docker,      setDocker]      = useState(settings.docker)
  const [deepScan,    setDeepScan]    = useState(settings.deepScan)
  const [intense,     setIntense]     = useState(settings.intense)
  const [hitl,        setHitl]        = useState(settings.hitl)
  const [planMode,    setPlanMode]    = useState(settings.planMode)
  const [retries,     setRetries]     = useState(settings.maxRetries)
  const [useBaseOnly, setUseBaseOnly] = useState(settings.useBaseOnly ?? false)
  const [qualityMode, setQualityMode] = useState<QualityMode>(settings.qualityMode ?? 'fast')

  const handleSave = () => {
    onSave({ docker, deepScan, maxRetries: retries, hitl, intense, planMode, useBaseOnly, qualityMode })
    onClose()
  }

  return (
    <div className="fixed inset-0 bg-black/60 flex items-center justify-center z-50" onClick={onClose}>
      <div
        className="bg-zinc-900 rounded-xl border border-zinc-700 w-[480px] max-h-[90vh] overflow-hidden shadow-2xl animate-modal-in"
        onClick={e => e.stopPropagation()}
      >
        <div className="flex items-center justify-between p-4 border-b border-zinc-800">
          <h2 className="text-lg font-semibold text-zinc-100">Run Settings</h2>
          <button onClick={onClose} className="p-1 hover:bg-zinc-800 rounded transition-colors">
            <X size={20} className="text-zinc-400" />
          </button>
        </div>

        <div className="p-6 space-y-6 overflow-y-auto max-h-[calc(90vh-120px)]">

          {/* Quality Mode Selector */}
          <div className="space-y-2">
            <h3 className="text-sm font-medium text-zinc-300">Quality Mode</h3>
            <p className="text-xs text-zinc-500">Controls how many pipeline stages run per file</p>
            <div className="space-y-2">
              {QUALITY_TIERS.map(tier => {
                const isActive = qualityMode === tier.id
                const border = tier.color === 'emerald' ? 'border-emerald-500/50 bg-emerald-500/5'
                             : tier.color === 'amber'   ? 'border-amber-500/50  bg-amber-500/5'
                             : 'border-zinc-700 bg-zinc-800/30'
                return (
                  <button key={tier.id} onClick={() => setQualityMode(tier.id)}
                    className={`w-full text-left p-3 rounded-lg border transition-all ${isActive ? border : 'border-zinc-800 hover:border-zinc-700'}`}>
                    <div className="flex items-center gap-2">
                      <span className={isActive
                        ? tier.color === 'emerald' ? 'text-emerald-400'
                        : tier.color === 'amber'   ? 'text-amber-400'
                        : 'text-zinc-300' : 'text-zinc-500'}>
                        {tier.icon}
                      </span>
                      <span className={`text-sm font-medium ${isActive ? 'text-zinc-100' : 'text-zinc-400'}`}>{tier.label}</span>
                      <span className="ml-auto text-[10px] text-zinc-500">{tier.time}</span>
                      {isActive && (
                        <span className={`w-2 h-2 rounded-full ${
                          tier.color === 'emerald' ? 'bg-emerald-400'
                          : tier.color === 'amber' ? 'bg-amber-400'
                          : 'bg-zinc-400'}`} />
                      )}
                    </div>
                    <p className="text-[11px] text-zinc-500 mt-1 pl-6">{tier.desc}</p>
                  </button>
                )
              })}
            </div>
          </div>

          {/* Execution */}
          <div className="space-y-3">
            <h3 className="text-sm font-medium text-zinc-300">Execution</h3>
            <Toggle label="Docker Isolation" desc="Run tests in isolated containers (requires Docker)" checked={docker} onChange={() => setDocker(!docker)} />
            <Toggle label="Deep Scan" desc="Enable comprehensive vulnerability scanning" checked={deepScan} onChange={() => setDeepScan(!deepScan)} />
            <Toggle label="Intense Mode" desc="Multiple iterations per target (slower, better coverage)" checked={intense} onChange={() => setIntense(!intense)} />
            <Toggle label="Human-in-the-Loop" desc="Pause for manual review before running tests" checked={hitl} onChange={() => setHitl(!hitl)} />
          </div>

          {/* Generation */}
          <div className="space-y-3">
            <h3 className="text-sm font-medium text-zinc-300">Generation</h3>
            <Toggle label="Plan Mode" desc="Generate a test plan before code — review & edit it" checked={planMode} onChange={() => setPlanMode(!planMode)} color="violet" />
            <Toggle
              label="Base Model Only (no LoRA)"
              desc="Use raw Qwen2.5-Coder-7B (4-bit) without the value_reasoning LoRA. Useful for A/B comparison vs fine-tuned model."
              checked={useBaseOnly}
              onChange={() => setUseBaseOnly(!useBaseOnly)}
              color="amber"
            />
            <div className="p-3 bg-zinc-800/50 rounded-lg">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm text-zinc-400">Max Retries per Target</span>
                <span className="text-sm font-medium text-emerald-400">{retries}</span>
              </div>
              <input type="range" min="1" max="5" value={retries}
                onChange={e => setRetries(Number(e.target.value))}
                className="w-full h-2 bg-zinc-700 rounded-lg appearance-none cursor-pointer accent-emerald-500" />
              <div className="flex justify-between text-xs text-zinc-500 mt-1">
                <span>1 (fast)</span><span>5 (thorough)</span>
              </div>
            </div>
          </div>

          {/* Active summary */}
          <div className="p-3 bg-emerald-500/5 border border-emerald-500/20 rounded-lg">
            <h3 className="text-xs font-semibold text-emerald-400 uppercase tracking-wider mb-2">Active Configuration</h3>
            <div className="flex flex-wrap gap-2">
              {docker   && <span className="px-2 py-0.5 text-xs rounded-full bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">Docker</span>}
              {deepScan && <span className="px-2 py-0.5 text-xs rounded-full bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">Deep Scan</span>}
              {intense  && <span className="px-2 py-0.5 text-xs rounded-full bg-amber-500/20 text-amber-400 border border-amber-500/30">Intense</span>}
              {hitl     && <span className="px-2 py-0.5 text-xs rounded-full bg-indigo-500/20 text-indigo-400 border border-indigo-500/30">HITL</span>}
              {planMode     && <span className="px-2 py-0.5 text-xs rounded-full bg-violet-500/20 text-violet-400 border border-violet-500/30">Plan Mode</span>}
              {useBaseOnly  && <span className="px-2 py-0.5 text-xs rounded-full bg-amber-500/20 text-amber-400 border border-amber-500/30">Base Only</span>}
              <span className={`px-2 py-0.5 text-xs rounded-full border ${
                qualityMode === 'best'     ? 'bg-amber-500/20 text-amber-400 border-amber-500/30'
                : qualityMode === 'balanced' ? 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30'
                : 'bg-zinc-700 text-zinc-400 border-zinc-600'
              }`}>
                {qualityMode === 'best' ? '💎 Best' : qualityMode === 'balanced' ? '⚖️ Balanced' : '⚡ Fast'}
              </span>
              <span className="px-2 py-0.5 text-xs rounded-full bg-zinc-700 text-zinc-400 border border-zinc-600">{retries} retries</span>
              {!docker && !deepScan && !intense && !hitl && !planMode && (
                <span className="px-2 py-0.5 text-xs rounded-full bg-zinc-700 text-zinc-400 border border-zinc-600">Default</span>
              )}
            </div>
          </div>
        </div>

        <div className="p-4 border-t border-zinc-800 flex justify-end gap-3">
          <button onClick={onClose} className="px-4 py-2 text-sm font-medium text-zinc-300 hover:text-white bg-zinc-800 hover:bg-zinc-700 rounded-lg transition-colors">
            Cancel
          </button>
          <button onClick={handleSave} className="px-4 py-2 text-sm font-medium text-white bg-emerald-500 hover:bg-emerald-600 rounded-lg transition-colors">
            Save Changes
          </button>
        </div>
      </div>
    </div>
  )
}
