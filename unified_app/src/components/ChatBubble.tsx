import { useState, useRef, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { MessageSquare, Send, X, FileCode, Sparkles, Loader2 } from 'lucide-react'
import type { ChatMessage } from '../types'

interface TestFile {
  file?: string
  test_file?: string
  test_code?: string
  test_path?: string
  target?: string
}

interface Props {
  files: TestFile[]
  threads: Record<string, ChatMessage[]>
  busy: boolean
  status: string
  /** Send a message about file at `idx`. Returns once the request is dispatched. */
  onSend: (idx: number, message: string) => void
}

const EXAMPLES = [
  'Add a test for the empty-input case',
  'Remove the network mocks — use real objects',
  'Rewrite the parametrized test to cover negatives',
]

export function fileKey(f: TestFile): string {
  return f.test_path || f.target || f.test_file || f.file || 'test'
}

export default function ChatBubble({ files, threads, busy, status, onSend }: Props) {
  const [open, setOpen] = useState(false)
  const [idx, setIdx] = useState(0)
  const [input, setInput] = useState('')
  const scrollRef = useRef<HTMLDivElement>(null)

  const cur = files[Math.min(idx, files.length - 1)]
  const thread = cur ? (threads[fileKey(cur)] ?? []) : []

  // Auto-scroll to the latest message
  useEffect(() => {
    if (scrollRef.current) scrollRef.current.scrollTop = scrollRef.current.scrollHeight
  }, [thread.length, status, open])

  if (!files.length) return null

  const send = (text: string) => {
    const msg = text.trim()
    if (!msg || busy) return
    onSend(Math.min(idx, files.length - 1), msg)
    setInput('')
  }

  return (
    <>
      {/* Collapsed floating button */}
      <AnimatePresence>
        {!open && (
          <motion.button
            key="bubble-btn"
            initial={{ scale: 0, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            exit={{ scale: 0, opacity: 0 }}
            transition={{ type: 'spring', stiffness: 380, damping: 26 }}
            onClick={() => setOpen(true)}
            className="fixed bottom-5 right-5 z-50 w-14 h-14 rounded-full bg-violet-500 hover:bg-violet-600 shadow-lg shadow-violet-500/30 flex items-center justify-center text-white transition-colors"
            title="Refine tests with chat"
          >
            <MessageSquare size={22} />
          </motion.button>
        )}
      </AnimatePresence>

      {/* Expanded floating card */}
      <AnimatePresence>
        {open && (
          <motion.div
            key="bubble-card"
            initial={{ scale: 0.9, opacity: 0, y: 16 }}
            animate={{ scale: 1, opacity: 1, y: 0 }}
            exit={{ scale: 0.9, opacity: 0, y: 16 }}
            transition={{ type: 'spring', stiffness: 320, damping: 28 }}
            className="fixed bottom-5 right-5 z-50 w-[380px] h-[560px] max-h-[80vh] flex flex-col bg-zinc-900 border border-violet-500/30 rounded-2xl shadow-2xl shadow-black/50 overflow-hidden"
          >
            {/* Header */}
            <div className="p-3 bg-violet-500/10 border-b border-violet-500/20 flex items-center gap-2 shrink-0">
              <Sparkles size={16} className="text-violet-400" />
              <div className="min-w-0">
                <p className="text-sm font-semibold text-zinc-100 leading-tight">Refine tests</p>
                <p className="text-[10px] text-zinc-500">Tell me what to change — I edit & re-run the file</p>
              </div>
              <button onClick={() => setOpen(false)} className="ml-auto text-zinc-500 hover:text-zinc-200 transition-colors">
                <X size={18} />
              </button>
            </div>

            {/* File selector */}
            <div className="px-3 py-2 border-b border-zinc-800 bg-zinc-900/60 flex items-center gap-2 shrink-0">
              <FileCode size={13} className="text-violet-400 shrink-0" />
              <select
                value={Math.min(idx, files.length - 1)}
                onChange={e => setIdx(Number(e.target.value))}
                className="flex-1 min-w-0 bg-zinc-950 text-zinc-300 text-xs rounded-md border border-zinc-800 px-2 py-1 outline-none focus:border-violet-500/40"
              >
                {files.map((f, i) => (
                  <option key={i} value={i}>
                    {f.test_file ?? f.file ?? `test_${i}.py`}
                  </option>
                ))}
              </select>
            </div>

            {/* Messages */}
            <div ref={scrollRef} className="flex-1 overflow-y-auto p-3 space-y-3">
              {thread.length === 0 && (
                <div className="space-y-2">
                  <p className="text-xs text-zinc-500">Try one of these:</p>
                  {EXAMPLES.map((ex, i) => (
                    <button key={i} onClick={() => send(ex)} disabled={busy}
                      className="block w-full text-left text-xs text-zinc-300 bg-zinc-800/60 hover:bg-zinc-800 disabled:opacity-50 rounded-lg px-3 py-2 border border-zinc-800 transition-colors">
                      {ex}
                    </button>
                  ))}
                </div>
              )}
              {thread.map((m, i) => (
                <div key={i} className={`flex ${m.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                  <div className={`max-w-[85%] rounded-2xl px-3 py-2 text-xs leading-relaxed whitespace-pre-wrap ${
                    m.role === 'user'
                      ? 'bg-violet-500 text-white rounded-br-sm'
                      : m.ok === false
                        ? 'bg-red-500/10 text-red-300 border border-red-500/20 rounded-bl-sm'
                        : 'bg-zinc-800 text-zinc-200 rounded-bl-sm'
                  }`}>
                    {m.pending
                      ? <span className="flex items-center gap-1.5 text-zinc-400"><Loader2 size={12} className="animate-spin" />{status || 'Thinking…'}</span>
                      : m.content}
                  </div>
                </div>
              ))}
            </div>

            {/* Input */}
            <div className="p-3 border-t border-zinc-800 bg-zinc-900/60 shrink-0">
              <div className="flex items-end gap-2">
                <textarea
                  value={input}
                  onChange={e => setInput(e.target.value)}
                  onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(input) } }}
                  placeholder={busy ? 'Working…' : 'Describe a change to this test file…'}
                  disabled={busy}
                  rows={2}
                  className="flex-1 bg-zinc-950 text-zinc-200 text-xs rounded-lg border border-zinc-800 px-2.5 py-2 outline-none resize-none placeholder:text-zinc-600 focus:border-violet-500/40 disabled:opacity-60"
                />
                <button onClick={() => send(input)} disabled={busy || !input.trim()}
                  className="w-9 h-9 shrink-0 rounded-lg bg-violet-500 hover:bg-violet-600 disabled:opacity-40 flex items-center justify-center text-white transition-colors">
                  {busy ? <Loader2 size={15} className="animate-spin" /> : <Send size={15} />}
                </button>
              </div>
              <p className="mt-1.5 text-[10px] text-zinc-600">Edits are validated by re-running the test — a failing edit is discarded.</p>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  )
}
