import React from 'react'
import { FileCheck } from 'lucide-react'

const REQUIREMENTS = [
  {
    num: 1,
    id: 'REQ-001',
    type: 'Functional',
    text: 'The system shall allow users to upload SRS documents in PDF or DOCX format for automated processing.',
    confidence: 96,
  },
  {
    num: 2,
    id: 'REQ-002',
    type: 'Non-Functional',
    text: 'The system shall process uploaded documents and return extracted requirements within 30 seconds under normal load.',
    confidence: 88,
  },
  {
    num: 3,
    id: 'REQ-003',
    type: 'Functional',
    text: 'The system shall automatically extract and segment individual requirements from the uploaded SRS document.',
    confidence: 93,
  },
  {
    num: 4,
    id: 'REQ-004',
    type: 'Functional',
    text: 'The system shall generate a minimum of one test case for each identified functional requirement.',
    confidence: 91,
  },
  {
    num: 5,
    id: 'REQ-005',
    type: 'Non-Functional',
    text: 'The system shall maintain a confidence score for each extracted requirement based on NLP model analysis.',
    confidence: 84,
  },
  {
    num: 6,
    id: 'REQ-006',
    type: 'Functional',
    text: 'The system shall allow users to export all generated test cases in both JSON and CSV file formats.',
    confidence: 97,
  },
]

const NON_FUNC_KEYWORDS = [
  'performance', 'security', 'response time', 'latency', 'availability',
  'reliability', 'scalability', 'usability', 'maintainability', 'throughput',
  'concurrent', 'load', 'speed', 'efficiency', 'capacity',
]

function inferType(text) {
  const lower = text.toLowerCase()
  return NON_FUNC_KEYWORDS.some((kw) => lower.includes(kw)) ? 'Non-Functional' : 'Functional'
}

function TypeBadge({ type }) {
  if (type === 'Functional') {
    return (
      <span className="inline-block rounded px-2 py-0.5 text-[10px] font-bold bg-emerald-500/15 text-emerald-400 border border-emerald-500/30">
        Functional
      </span>
    )
  }
  return (
    <span className="inline-block rounded px-2 py-0.5 text-[10px] font-bold bg-blue-500/15 text-blue-400 border border-blue-500/30">
      Non-Functional
    </span>
  )
}

function ConfidenceBar({ value }) {
  return (
    <div className="flex items-center gap-2 min-w-[100px]">
      <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-zinc-800">
        <div
          className="h-full rounded-full bg-emerald-500 transition-all"
          style={{ width: `${value}%` }}
        />
      </div>
      <span className="w-8 text-right font-mono text-xs text-zinc-400">{value}%</span>
    </div>
  )
}

export default function RequirementsTab({ requirements }) {
  const rows = requirements
    ? requirements.rows.map((item, i) => ({
        num: i + 1,
        id: `REQ-${String(i + 1).padStart(3, '0')}`,
        type: inferType(item.text),
        text: item.text,
        confidence: Math.round(item.score * 100),
      }))
    : REQUIREMENTS

  const totalCount = requirements ? requirements.total : 47

  return (
    <div className="flex flex-col gap-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h2 className="text-sm font-semibold text-zinc-200">Extracted Requirements</h2>
        <span className="flex items-center gap-1.5 rounded-full bg-emerald-500 px-2.5 py-0.5 text-xs font-semibold text-white">
          <FileCheck size={11} />
          {totalCount} Requirements Extracted
        </span>
      </div>

      {/* Empty state */}
      {requirements && rows.length === 0 && (
        <p className="py-8 text-center text-xs text-zinc-500">
          No requirements matched the aligned dataset for this document.
        </p>
      )}

      {/* Table */}
      {rows.length > 0 && (
        <div className="overflow-hidden rounded-lg border border-zinc-800">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-zinc-800 bg-zinc-900">
                <th className="py-2.5 pl-4 pr-2 text-left text-[10px] font-semibold uppercase tracking-wider text-zinc-500 w-8">
                  #
                </th>
                <th className="px-3 py-2.5 text-left text-[10px] font-semibold uppercase tracking-wider text-zinc-500 w-24">
                  Req ID
                </th>
                <th className="px-3 py-2.5 text-left text-[10px] font-semibold uppercase tracking-wider text-zinc-500 w-36">
                  Type
                </th>
                <th className="px-3 py-2.5 text-left text-[10px] font-semibold uppercase tracking-wider text-zinc-500">
                  Requirement Text
                </th>
                <th className="px-3 py-2.5 pr-4 text-left text-[10px] font-semibold uppercase tracking-wider text-zinc-500 w-40">
                  Confidence
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-zinc-800/50 bg-zinc-950">
              {rows.map((req) => (
                <tr key={req.id} className="transition-colors hover:bg-zinc-900/60">
                  <td className="py-3 pl-4 pr-2 font-mono text-xs text-zinc-600">{req.num}</td>
                  <td className="px-3 py-3 font-mono text-xs font-semibold text-zinc-400">{req.id}</td>
                  <td className="px-3 py-3">
                    <TypeBadge type={req.type} />
                  </td>
                  <td className="px-3 py-3 text-xs leading-relaxed text-zinc-300">{req.text}</td>
                  <td className="px-3 py-3 pr-4">
                    <ConfidenceBar value={req.confidence} />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <p className="text-xs text-zinc-600">
        Showing {rows.length} of {totalCount} extracted requirements
        {!requirements && ' — run the full pipeline to view all.'}
      </p>
    </div>
  )
}
