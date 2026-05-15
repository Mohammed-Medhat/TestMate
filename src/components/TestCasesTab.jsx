import React from 'react'
import { FlaskConical, Download } from 'lucide-react'

const TEST_CASES = [
  {
    id: 'TC-001',
    reqId: 'REQ-001',
    description: 'Verify that a valid PDF file can be uploaded through the drag-and-drop interface',
    expected: 'File is accepted and displayed in the file card with correct name and size',
    status: 'Pass',
  },
  {
    id: 'TC-002',
    reqId: 'REQ-001',
    description: 'Verify that unsupported file formats are rejected at upload time',
    expected: 'Error message displayed indicating unsupported file type; file not processed',
    status: 'Pass',
  },
  {
    id: 'TC-003',
    reqId: 'REQ-003',
    description: 'Verify that the pipeline correctly segments sentences from an uploaded DOCX document',
    expected: 'Sentences extracted and listed with correct segmentation boundaries',
    status: 'Pass',
  },
  {
    id: 'TC-004',
    reqId: 'REQ-004',
    description: 'Verify that at least one test case is generated for every functional requirement identified',
    expected: 'Each functional requirement has one or more associated test cases in the output',
    status: 'Pending',
  },
  {
    id: 'TC-005',
    reqId: 'REQ-006',
    description: 'Verify that generated test cases can be exported as a valid JSON file with correct schema',
    expected: 'JSON file downloaded containing all test case fields with correct data types',
    status: 'Pending',
  },
]

function StatusBadge({ status }) {
  if (status === 'Pass') {
    return (
      <span className="inline-block rounded px-2 py-0.5 text-[10px] font-bold bg-emerald-500/15 text-emerald-400 border border-emerald-500/30">
        Pass
      </span>
    )
  }
  return (
    <span className="inline-block rounded px-2 py-0.5 text-[10px] font-bold bg-zinc-800 text-zinc-400 border border-zinc-700">
      Pending
    </span>
  )
}

function exportData(format) {
  if (format === 'json') {
    const blob = new Blob([JSON.stringify(TEST_CASES, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'test_cases.json'
    a.click()
    URL.revokeObjectURL(url)
  } else {
    const header = 'Test ID,Linked Requirement,Description,Expected Output,Status\n'
    const rows = TEST_CASES.map(
      (tc) =>
        `"${tc.id}","${tc.reqId}","${tc.description.replace(/"/g, '""')}","${tc.expected.replace(/"/g, '""')}","${tc.status}"`
    ).join('\n')
    const blob = new Blob([header + rows], { type: 'text/csv' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'test_cases.csv'
    a.click()
    URL.revokeObjectURL(url)
  }
}

export default function TestCasesTab() {
  return (
    <div className="flex flex-col gap-4">
      {/* Header */}
      <div className="flex items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <h2 className="text-sm font-semibold text-zinc-200">Generated Test Cases</h2>
          <span className="flex items-center gap-1.5 rounded-full bg-emerald-500 px-2.5 py-0.5 text-xs font-semibold text-white">
            <FlaskConical size={11} />
            23 Test Cases Generated
          </span>
        </div>

        {/* Export buttons */}
        <div className="flex items-center gap-2">
          <button
            onClick={() => exportData('json')}
            className="flex items-center gap-1.5 rounded-lg border border-emerald-500/40 px-3 py-1.5 text-xs font-medium text-emerald-400 transition-colors hover:bg-emerald-500/10 hover:border-emerald-400"
          >
            <Download size={12} />
            Export JSON
          </button>
          <button
            onClick={() => exportData('csv')}
            className="flex items-center gap-1.5 rounded-lg border border-emerald-500/40 px-3 py-1.5 text-xs font-medium text-emerald-400 transition-colors hover:bg-emerald-500/10 hover:border-emerald-400"
          >
            <Download size={12} />
            Export CSV
          </button>
        </div>
      </div>

      {/* Table */}
      <div className="overflow-hidden rounded-lg border border-zinc-800">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-zinc-800 bg-zinc-900">
              <th className="py-2.5 pl-4 pr-3 text-left text-[10px] font-semibold uppercase tracking-wider text-zinc-500 w-20">
                Test ID
              </th>
              <th className="px-3 py-2.5 text-left text-[10px] font-semibold uppercase tracking-wider text-zinc-500 w-28">
                Linked Req
              </th>
              <th className="px-3 py-2.5 text-left text-[10px] font-semibold uppercase tracking-wider text-zinc-500">
                Test Case Description
              </th>
              <th className="px-3 py-2.5 text-left text-[10px] font-semibold uppercase tracking-wider text-zinc-500">
                Expected Output
              </th>
              <th className="px-3 py-2.5 pr-4 text-left text-[10px] font-semibold uppercase tracking-wider text-zinc-500 w-20">
                Status
              </th>
            </tr>
          </thead>
          <tbody className="divide-y divide-zinc-800/50 bg-zinc-950">
            {TEST_CASES.map((tc) => (
              <tr key={tc.id} className="transition-colors hover:bg-zinc-900/60">
                <td className="py-3 pl-4 pr-3 font-mono text-xs font-semibold text-zinc-400">{tc.id}</td>
                <td className="px-3 py-3 font-mono text-xs text-zinc-500">{tc.reqId}</td>
                <td className="px-3 py-3 text-xs leading-relaxed text-zinc-300">{tc.description}</td>
                <td className="px-3 py-3 text-xs leading-relaxed text-zinc-400">{tc.expected}</td>
                <td className="px-3 py-3 pr-4">
                  <StatusBadge status={tc.status} />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <p className="text-xs text-zinc-600">
        Showing 5 of 23 generated test cases — run the full pipeline to generate all cases.
      </p>
    </div>
  )
}
