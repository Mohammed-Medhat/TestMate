import React, { useState } from 'react'
import { X, Settings, ChevronDown } from 'lucide-react'

function Toggle({ label, checked, onChange }) {
  return (
    <div className="flex items-center justify-between">
      <span className="text-sm text-zinc-300">{label}</span>
      <div className="flex items-center gap-2.5">
        <button
          onClick={() => onChange(!checked)}
          className={`relative inline-flex h-5 w-9 flex-none items-center rounded-full transition-colors duration-200 ${
            checked ? 'bg-emerald-500' : 'bg-zinc-700'
          }`}
        >
          <span
            className={`inline-block h-3.5 w-3.5 transform rounded-full bg-white shadow-sm transition-transform duration-200 ${
              checked ? 'translate-x-[18px]' : 'translate-x-[3px]'
            }`}
          />
        </button>
        <span
          className={`w-12 text-right text-xs font-medium ${
            checked ? 'text-emerald-400' : 'text-zinc-500'
          }`}
        >
          {checked ? 'Active' : 'Inactive'}
        </span>
      </div>
    </div>
  )
}

const LLM_MODELS = ['Local Llama 3', 'GPT-4', 'Claude 3', 'Gemini Pro']
const FRAMEWORKS = ['PyTest', 'Jest']
const SEVERITIES = ['Critical', 'High', 'Medium', 'Low']

export default function SettingsModal({ onClose }) {
  const [docker, setDocker] = useState(true)
  const [deepScan, setDeepScan] = useState(true)
  const [model, setModel] = useState('Local Llama 3')
  const [framework, setFramework] = useState('PyTest')
  const [coverage, setCoverage] = useState(80)
  const [severities, setSeverities] = useState(['Critical', 'High', 'Medium'])

  function toggleSeverity(sev) {
    setSeverities((prev) =>
      prev.includes(sev) ? prev.filter((s) => s !== sev) : [...prev, sev]
    )
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-zinc-950/75 backdrop-blur-sm">
      <div className="relative w-[500px] overflow-hidden rounded-2xl border border-zinc-700 bg-zinc-900 shadow-2xl">
        {/* Header */}
        <div className="flex items-center justify-between border-b border-zinc-800 px-5 py-4">
          <div className="flex items-center gap-2">
            <Settings size={15} className="text-emerald-400" />
            <h2 className="text-sm font-semibold text-zinc-100">Project Settings</h2>
          </div>
          <button
            onClick={onClose}
            className="rounded-lg p-1 text-zinc-500 transition-colors hover:bg-zinc-800 hover:text-zinc-300"
          >
            <X size={16} />
          </button>
        </div>

        {/* Body */}
        <div className="flex flex-col gap-5 p-5">
          {/* Toggles */}
          <section className="flex flex-col gap-3 rounded-xl border border-zinc-800 bg-zinc-950/40 p-4">
            <p className="text-[10px] font-semibold uppercase tracking-widest text-zinc-500">
              Isolation &amp; Scanning
            </p>
            <Toggle label="Docker Isolation" checked={docker} onChange={setDocker} />
            <div className="border-t border-zinc-800/60" />
            <Toggle label="Deep Scan" checked={deepScan} onChange={setDeepScan} />
          </section>

          {/* LLM Model */}
          <section className="flex flex-col gap-2">
            <label className="text-[10px] font-semibold uppercase tracking-widest text-zinc-500">
              LLM Model Selection
            </label>
            <div className="relative">
              <select
                value={model}
                onChange={(e) => setModel(e.target.value)}
                className="w-full appearance-none rounded-lg border border-zinc-700 bg-zinc-800 py-2 pl-3 pr-8 text-sm text-zinc-200 outline-none transition-colors focus:border-emerald-500"
              >
                {LLM_MODELS.map((m) => (
                  <option key={m} value={m}>
                    {m}
                  </option>
                ))}
              </select>
              <ChevronDown
                size={14}
                className="pointer-events-none absolute right-2.5 top-1/2 -translate-y-1/2 text-zinc-500"
              />
            </div>
          </section>

          {/* Test Framework */}
          <section className="flex flex-col gap-2">
            <label className="text-[10px] font-semibold uppercase tracking-widest text-zinc-500">
              Test Framework
            </label>
            <div className="flex gap-5">
              {FRAMEWORKS.map((f) => (
                <label
                  key={f}
                  className="flex cursor-pointer items-center gap-2"
                >
                  <input
                    type="radio"
                    name="framework"
                    value={f}
                    checked={framework === f}
                    onChange={() => setFramework(f)}
                    className="h-3.5 w-3.5 accent-emerald-500"
                  />
                  <span className="text-sm text-zinc-300">{f}</span>
                  {framework === f && (
                    <span className="text-xs font-bold text-emerald-400">✓</span>
                  )}
                </label>
              ))}
            </div>
          </section>

          {/* Coverage Threshold */}
          <section className="flex flex-col gap-2">
            <div className="flex items-center justify-between">
              <label className="text-[10px] font-semibold uppercase tracking-widest text-zinc-500">
                Coverage Threshold
              </label>
              <span className="font-mono text-sm font-bold text-emerald-400">{coverage}%</span>
            </div>
            <input
              type="range"
              min="0"
              max="100"
              value={coverage}
              onChange={(e) => setCoverage(Number(e.target.value))}
              className="w-full"
            />
            <div className="flex justify-between text-[9px] text-zinc-600">
              <span>0%</span>
              <span>50%</span>
              <span>100%</span>
            </div>
          </section>

          {/* Bug Severity */}
          <section className="flex flex-col gap-2">
            <label className="text-[10px] font-semibold uppercase tracking-widest text-zinc-500">
              Bug Severity Threshold
            </label>
            <div className="flex flex-wrap gap-4">
              {SEVERITIES.map((sev) => (
                <label key={sev} className="flex cursor-pointer items-center gap-2">
                  <input
                    type="checkbox"
                    checked={severities.includes(sev)}
                    onChange={() => toggleSeverity(sev)}
                    className="h-3.5 w-3.5 accent-emerald-500"
                  />
                  <span className="text-sm text-zinc-300">{sev}</span>
                </label>
              ))}
            </div>
          </section>
        </div>

        {/* Footer */}
        <div className="flex items-center justify-end gap-2 border-t border-zinc-800 px-5 py-3">
          <button
            onClick={onClose}
            className="rounded-lg px-4 py-2 text-sm text-zinc-400 transition-colors hover:bg-zinc-800 hover:text-zinc-200"
          >
            Cancel
          </button>
          <button
            onClick={onClose}
            className="rounded-lg bg-emerald-500 px-4 py-2 text-sm font-semibold text-white transition-colors hover:bg-emerald-400 active:bg-emerald-600"
          >
            Save Settings
          </button>
        </div>
      </div>
    </div>
  )
}
