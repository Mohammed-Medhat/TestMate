import { useState } from 'react'
import { Copy, Check } from 'lucide-react'

interface Props {
  code: string
  filename?: string
}

export default function CodeViewer({ code, filename = 'test_*.py' }: Props) {
  const [copied, setCopied] = useState(false)

  const copy = () => {
    navigator.clipboard.writeText(code)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  if (!code)
    return <div className="flex items-center justify-center h-32 text-zinc-500 text-sm">No test code generated yet</div>

  const lines = code.split('\n')

  return (
    <div className="flex flex-col h-full">
      {/* Toolbar */}
      <div className="flex items-center justify-between px-4 py-2 bg-zinc-900 border-b border-zinc-800 shrink-0">
        <span className="font-mono text-xs text-emerald-400">{filename}</span>
        <button
          onClick={copy}
          className="flex items-center gap-1.5 text-xs text-zinc-400 hover:text-zinc-200 transition-colors"
        >
          {copied ? <Check size={12} className="text-emerald-400" /> : <Copy size={12} />}
          {copied ? 'Copied!' : 'Copy'}
        </button>
      </div>
      {/* Code */}
      <div className="flex-1 overflow-auto font-mono text-[11px] leading-relaxed">
        {lines.map((line, i) => (
          <div key={i} className="flex hover:bg-zinc-900/50 group">
            <span className="w-10 shrink-0 text-right pr-3 text-zinc-600 select-none border-r border-zinc-800/50 group-hover:text-zinc-500">
              {i + 1}
            </span>
            <span className="pl-4 text-zinc-300 whitespace-pre">{line}</span>
          </div>
        ))}
      </div>
    </div>
  )
}
