import React, { useState } from 'react'
import { Eye, AlertTriangle, X } from 'lucide-react'

const BUGS = [
  {
    id: 'ZD-001',
    severity: 'Critical',
    description: 'Cross-Site Scripting (XSS) in Login',
    detail: 'Unsanitized user input is rendered directly into the DOM in the login form, allowing injection of arbitrary scripts.',
  },
  {
    id: 'ZD-002',
    severity: 'High',
    description: 'SQL Injection in User Query',
    detail: 'The user search endpoint concatenates raw input into a SQL query string without parameterization.',
  },
  {
    id: 'ZD-003',
    severity: 'Medium',
    description: 'Insecure Direct Object Reference',
    detail: 'Object IDs are exposed in URLs without authorization checks, allowing horizontal privilege escalation.',
  },
  {
    id: 'ZD-004',
    severity: 'Critical',
    description: 'Remote Code Execution in Upload',
    detail: 'The file upload endpoint does not validate MIME type server-side, allowing upload and execution of server-side scripts.',
  },
  {
    id: 'ZD-005',
    severity: 'Low',
    description: 'Information Disclosure in Headers',
    detail: 'HTTP response headers expose server version and framework details that aid fingerprinting.',
  },
]

const SEVERITY = {
  Critical: { badge: 'bg-red-600 text-white', dot: 'bg-red-500' },
  High: { badge: 'bg-orange-500 text-white', dot: 'bg-orange-400' },
  Medium: { badge: 'bg-yellow-500 text-black', dot: 'bg-yellow-400' },
  Low: { badge: 'bg-blue-500 text-white', dot: 'bg-blue-400' },
}

function DetailPanel({ bug, onClose }) {
  return (
    <div className="mt-4 rounded-lg border border-zinc-700 bg-zinc-900 p-4">
      <div className="mb-3 flex items-start justify-between gap-4">
        <div>
          <span className="font-mono text-xs text-zinc-500">{bug.id}</span>
          <h3 className="mt-0.5 text-sm font-semibold text-zinc-100">{bug.description}</h3>
        </div>
        <button
          onClick={onClose}
          className="mt-0.5 flex-none rounded p-0.5 text-zinc-500 transition-colors hover:bg-zinc-800 hover:text-zinc-300"
        >
          <X size={14} />
        </button>
      </div>
      <p className="text-xs leading-relaxed text-zinc-400">{bug.detail}</p>
      <div className="mt-3 flex gap-2">
        <button className="rounded border border-emerald-500/40 bg-emerald-500/10 px-3 py-1.5 text-xs font-medium text-emerald-400 transition-colors hover:bg-emerald-500/20">
          Generate Fix
        </button>
        <button className="rounded border border-zinc-700 px-3 py-1.5 text-xs font-medium text-zinc-400 transition-colors hover:bg-zinc-800">
          Suppress
        </button>
      </div>
    </div>
  )
}

export default function BugTable() {
  const [expanded, setExpanded] = useState(null)

  function toggle(id) {
    setExpanded((prev) => (prev === id ? null : id))
  }

  return (
    <div className="flex flex-col gap-4">
      {/* Header row */}
      <div className="flex items-center justify-between">
        <h2 className="text-sm font-semibold text-zinc-200">Detected Bugs</h2>
        <span className="flex items-center gap-1.5 rounded-full bg-red-600 px-2.5 py-0.5 text-xs font-semibold text-white">
          <AlertTriangle size={11} />5 Issues
        </span>
      </div>

      {/* Table */}
      <div className="overflow-hidden rounded-lg border border-zinc-800">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-zinc-800 bg-zinc-900">
              <th className="py-2.5 pl-4 pr-3 text-left text-[10px] font-semibold uppercase tracking-wider text-zinc-500">
                ID
              </th>
              <th className="px-3 py-2.5 text-left text-[10px] font-semibold uppercase tracking-wider text-zinc-500">
                Severity
              </th>
              <th className="px-3 py-2.5 text-left text-[10px] font-semibold uppercase tracking-wider text-zinc-500">
                Description
              </th>
              <th className="py-2.5 pl-3 pr-4 text-right text-[10px] font-semibold uppercase tracking-wider text-zinc-500">
                Actions
              </th>
            </tr>
          </thead>
          <tbody className="divide-y divide-zinc-800/50 bg-zinc-950">
            {BUGS.map((bug) => {
              const { badge } = SEVERITY[bug.severity]
              const isOpen = expanded === bug.id
              return (
                <tr
                  key={bug.id}
                  className={`transition-colors ${isOpen ? 'bg-zinc-800/30' : 'hover:bg-zinc-900/60'}`}
                >
                  <td className="py-3 pl-4 pr-3 font-mono text-xs text-zinc-500">{bug.id}</td>
                  <td className="px-3 py-3">
                    <span className={`inline-block rounded px-2 py-0.5 text-[10px] font-bold ${badge}`}>
                      {bug.severity}
                    </span>
                  </td>
                  <td className="px-3 py-3 text-sm text-zinc-200">{bug.description}</td>
                  <td className="py-3 pl-3 pr-4 text-right">
                    <button
                      onClick={() => toggle(bug.id)}
                      className={`inline-flex items-center gap-1.5 rounded border px-2.5 py-1 text-xs font-medium transition-colors ${
                        isOpen
                          ? 'border-emerald-500 bg-emerald-500/20 text-emerald-300'
                          : 'border-emerald-500/40 bg-emerald-500/10 text-emerald-400 hover:border-emerald-400 hover:bg-emerald-500/20'
                      }`}
                    >
                      <Eye size={11} />
                      {isOpen ? 'Hide' : 'View Details'}
                    </button>
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>

      {/* Expanded detail panel */}
      {expanded && (
        <DetailPanel
          bug={BUGS.find((b) => b.id === expanded)}
          onClose={() => setExpanded(null)}
        />
      )}
    </div>
  )
}
