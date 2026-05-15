import React, { useState } from 'react'
import { Home, Settings, User, FolderTree, ChevronRight } from 'lucide-react'
import FileTree from './FileTree'

const NAV_ITEMS = [
  { id: 'explorer', icon: FolderTree, label: 'Explorer' },
  { id: 'home', icon: Home, label: 'Home' },
  { id: 'profile', icon: User, label: 'Profile' },
  { id: 'settings', icon: Settings, label: 'Settings' },
]

export default function Sidebar({ onSettingsClick }) {
  const [activeNav, setActiveNav] = useState('explorer')

  function handleNavClick(id) {
    if (id === 'settings') {
      onSettingsClick()
      return
    }
    setActiveNav((prev) => (prev === id ? null : id))
  }

  return (
    <div className="flex h-full flex-none">
      {/* ── Narrow icon rail ── */}
      <div className="flex w-14 flex-col items-center gap-1 border-r border-zinc-800 bg-zinc-950 py-3">
        {NAV_ITEMS.map(({ id, icon: Icon, label }) => (
          <button
            key={id}
            onClick={() => handleNavClick(id)}
            title={label}
            className={`flex h-10 w-10 items-center justify-center rounded-lg transition-all duration-150 ${
              activeNav === id
                ? 'bg-emerald-500/15 text-emerald-400 ring-1 ring-emerald-500/30'
                : 'text-zinc-500 hover:bg-zinc-800 hover:text-zinc-300'
            }`}
          >
            <Icon size={19} strokeWidth={1.75} />
          </button>
        ))}
      </div>

      {/* ── File explorer panel ── */}
      {activeNav === 'explorer' && (
        <div className="flex w-56 flex-col border-r border-zinc-800 bg-zinc-900">
          <div className="flex items-center gap-1.5 border-b border-zinc-800 px-3 py-2.5">
            <ChevronRight size={11} className="text-zinc-600" />
            <span className="text-[10px] font-semibold uppercase tracking-widest text-zinc-500">
              Projects
            </span>
          </div>
          <div className="flex-1 overflow-y-auto py-1">
            <FileTree />
          </div>
        </div>
      )}
    </div>
  )
}
