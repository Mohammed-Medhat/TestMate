import React from 'react'
import { ChevronRight, Settings } from 'lucide-react'

const TABS = [
  { id: 'upload', label: 'Upload SRS' },
  { id: 'requirements', label: 'Extracted Requirements' },
  { id: 'testcases', label: 'Generated Test Cases' },
  { id: 'readme', label: 'README Extractor' },
]

export default function Breadcrumb({ activeTab, onTabChange, onSettingsClick }) {
  return (
    <div className="flex flex-col bg-zinc-900">
      {/* Top bar: path + settings */}
      <div className="flex items-center justify-between gap-4 border-b border-zinc-800/60 px-4 py-2">
        <div className="flex min-w-0 items-center gap-1 text-xs text-zinc-500">
          <span className="cursor-pointer transition-colors hover:text-zinc-300">testmate</span>
          <ChevronRight size={11} className="flex-none" />
          <span className="cursor-pointer transition-colors hover:text-zinc-300">pipeline</span>
          <ChevronRight size={11} className="flex-none" />
          <span className="font-mono text-zinc-200">upload_srs</span>
        </div>

        <button
          onClick={onSettingsClick}
          className="flex flex-none items-center gap-1.5 rounded-md border border-zinc-700 px-2.5 py-1.5 text-xs text-zinc-400 transition-colors hover:border-zinc-600 hover:bg-zinc-800 hover:text-zinc-200"
        >
          <Settings size={11} />
          Settings
        </button>
      </div>

      {/* Tab bar */}
      <div className="flex items-center gap-0 px-4">
        {TABS.map((tab) => (
          <button
            key={tab.id}
            onClick={() => onTabChange(tab.id)}
            className={`border-b-2 px-4 py-2 text-xs font-medium transition-colors ${
              activeTab === tab.id
                ? 'border-emerald-400 text-emerald-400'
                : 'border-transparent text-zinc-500 hover:text-zinc-300'
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>
    </div>
  )
}
