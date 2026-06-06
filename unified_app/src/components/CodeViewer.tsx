import { useState } from 'react'
import { Copy, Check, Download } from 'lucide-react'
import SyntaxHighlighter from 'react-syntax-highlighter'
import { atomOneDark } from 'react-syntax-highlighter/dist/esm/styles/hljs'

interface Props {
  code: string
  filename?: string
  language?: string
}

export default function CodeViewer({ code, filename = 'test_*.py', language }: Props) {
  const [copied, setCopied]       = useState(false)
  const [downloaded, setDownloaded] = useState(false)

  const detectedLang = language
    ?? (filename.endsWith('.py') ? 'python'
      : filename.endsWith('.js') || filename.endsWith('.ts') ? 'typescript'
      : 'text')

  const copy = () => {
    navigator.clipboard.writeText(code)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  const download = () => {
    const blob = new Blob([code], { type: 'text/plain' })
    const url  = URL.createObjectURL(blob)
    const a    = document.createElement('a')
    a.href = url; a.download = filename.split('/').pop() ?? 'test.py'
    a.click(); URL.revokeObjectURL(url)
    setDownloaded(true)
    setTimeout(() => setDownloaded(false), 2000)
  }

  if (!code)
    return <div className="flex items-center justify-center h-32 text-zinc-500 text-sm">No code generated yet</div>

  return (
    <div className="flex flex-col h-full">
      {/* Toolbar */}
      <div className="flex items-center justify-between px-4 py-2 bg-zinc-900 border-b border-zinc-800 shrink-0">
        <span className="font-mono text-xs text-emerald-400">{filename}</span>
        <div className="flex items-center gap-2">
          <button onClick={copy}
            className="flex items-center gap-1.5 text-xs text-zinc-400 hover:text-zinc-200 transition-colors focus-ring rounded px-1.5 py-0.5">
            {copied ? <Check size={12} className="text-emerald-400" /> : <Copy size={12} />}
            {copied ? 'Copied!' : 'Copy'}
          </button>
          <button onClick={download}
            className="flex items-center gap-1.5 text-xs text-zinc-400 hover:text-zinc-200 transition-colors focus-ring rounded px-1.5 py-0.5">
            {downloaded ? <Check size={12} className="text-emerald-400" /> : <Download size={12} />}
            {downloaded ? 'Saved!' : 'Save'}
          </button>
        </div>
      </div>
      {/* Highlighted code */}
      <div className="flex-1 overflow-auto text-[11px]">
        <SyntaxHighlighter
          language={detectedLang}
          style={atomOneDark}
          showLineNumbers
          lineNumberStyle={{ color: '#52525b', minWidth: '2.8em', paddingRight: '12px', borderRight: '1px solid #27272a', marginRight: '12px' }}
          customStyle={{ margin: 0, padding: '8px 0', background: '#09090b', height: '100%', fontSize: '11px', lineHeight: '1.6' }}
          codeTagProps={{ style: { fontFamily: "'JetBrains Mono', 'Fira Code', 'Cascadia Code', monospace" } }}
        >
          {code}
        </SyntaxHighlighter>
      </div>
    </div>
  )
}
