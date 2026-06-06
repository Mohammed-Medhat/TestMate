import { X, Link2, FileText, Bot, Wrench, Keyboard, Info } from 'lucide-react'
import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'

const TABS = [
  { id: 'overview',  label: 'Overview',      icon: <Info size={13} />,     color: 'text-zinc-400' },
  { id: 'combined',  label: 'Full Pipeline', icon: <Link2 size={13} />,    color: 'text-emerald-400' },
  { id: 'parta',     label: 'Part A',        icon: <FileText size={13} />, color: 'text-blue-400' },
  { id: 'partb',     label: 'Part B',        icon: <Bot size={13} />,      color: 'text-violet-400' },
  { id: 'partc',     label: 'Bug Fixer',     icon: <Wrench size={13} />,   color: 'text-amber-400' },
  { id: 'shortcuts', label: 'Shortcuts',     icon: <Keyboard size={13} />, color: 'text-zinc-400' },
]

function Kbd({ children }: { children: React.ReactNode }) {
  return (
    <kbd className="px-1.5 py-0.5 text-[10px] font-mono rounded border border-zinc-600 bg-zinc-800 text-zinc-300">
      {children}
    </kbd>
  )
}

function Step({ n, children }: { n: number; children: React.ReactNode }) {
  return (
    <div className="flex gap-3">
      <span className="flex-shrink-0 w-5 h-5 rounded-full bg-zinc-800 border border-zinc-700 text-[10px] font-bold text-zinc-400 flex items-center justify-center mt-0.5">
        {n}
      </span>
      <p className="text-[12px] text-zinc-400 leading-relaxed">{children}</p>
    </div>
  )
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="space-y-2">
      <h4 className="text-[10px] font-bold text-zinc-500 uppercase tracking-[0.14em]">{title}</h4>
      <div className="space-y-2">{children}</div>
    </div>
  )
}

function Tip({ children }: { children: React.ReactNode }) {
  return (
    <p className="text-[11px] text-zinc-500 pl-3 border-l-2 border-zinc-700 leading-relaxed">
      {children}
    </p>
  )
}

const CONTENT: Record<string, React.ReactNode> = {
  overview: (
    <div className="space-y-5">
      <p className="text-[12px] text-zinc-300 leading-relaxed">
        TestMate is an AI-driven test generation and bug repair platform. It uses a fine-tuned
        Qwen2.5-Coder-7B model with a 3-layer RAG agent to automatically write, run, and self-correct
        Python test suites for your codebase.
      </p>

      <Section title="Four Modes">
        <div className="space-y-2">
          {[
            { color: 'bg-emerald-500', label: 'Full Pipeline', desc: 'End-to-end: SRS requirements → test generation (best starting point)' },
            { color: 'bg-blue-500',    label: 'Part A',         desc: 'Extract and classify requirements from SRS documents or README files' },
            { color: 'bg-violet-500',  label: 'Part B',         desc: 'Standalone AI test generator for any Python repo on GitHub' },
            { color: 'bg-amber-500',   label: 'Bug Fixer',      desc: 'SBFL fault localization + LLM automated program repair' },
          ].map(m => (
            <div key={m.label} className="flex items-start gap-2.5">
              <span className={`flex-shrink-0 w-2 h-2 rounded-full mt-1.5 ${m.color}`} />
              <div>
                <span className="text-[12px] font-semibold text-zinc-200">{m.label} — </span>
                <span className="text-[12px] text-zinc-400">{m.desc}</span>
              </div>
            </div>
          ))}
        </div>
      </Section>

      <Section title="Quick Start">
        <Step n={1}>Click any mode card on the home screen (or press <Kbd>1</Kbd>–<Kbd>4</Kbd>)</Step>
        <Step n={2}>Fill in the inputs on the left sidebar</Step>
        <Step n={3}>Hit <Kbd>Run</Kbd> or press <Kbd>Ctrl</Kbd>+<Kbd>Enter</Kbd></Step>
        <Step n={4}>Watch the pipeline stepper and live logs — results appear in the centre tabs</Step>
      </Section>

      <Section title="Settings">
        <Tip>Click the ⚙ gear icon (top-right on landing, or sidebar) to open Run Settings. There you can choose Quality Mode, enable Plan Mode, Docker isolation, and more.</Tip>
      </Section>
    </div>
  ),

  combined: (
    <div className="space-y-5">
      <p className="text-[12px] text-zinc-300 leading-relaxed">
        Full Pipeline chains Part A → Part B into a single automated run. Provide an SRS document
        and/or README, then point it at a GitHub repo and it does the rest.
      </p>

      <Section title="Workflow">
        <Step n={1}>
          <strong className="text-zinc-300">SRS File (optional)</strong> — upload a PDF or DOCX requirements document.
          The pipeline extracts labelled requirements and routes the most relevant ones to each source file.
        </Step>
        <Step n={2}>
          <strong className="text-zinc-300">README (optional)</strong> — paste the project README to extract
          feature descriptions and generate scenario-level test ideas.
        </Step>
        <Step n={3}>
          <strong className="text-zinc-300">Repo URL</strong> — enter a GitHub URL and click <em>Discover Files</em>
          to scan for testable Python files. Select the ones you want tested.
        </Step>
        <Step n={4}>
          Click <strong className="text-zinc-300">Run Full Pipeline</strong>. The stepper shows:
          SRS Extract → README Extract → Discover → Test Gen (→ Gap Analysis → Gap Fill if quality mode allows).
        </Step>
        <Step n={5}>
          Results appear in real time: <em>Requirements</em>, <em>Test Code</em>, <em>Coverage</em>, and <em>Bugs</em> tabs.
        </Step>
      </Section>

      <Section title="Quality Modes (Settings)">
        <Tip><strong className="text-zinc-300">Fast</strong> — SRS + README extraction then test generation. ~2-3 min/file.</Tip>
        <Tip><strong className="text-zinc-300">Balanced</strong> — adds SRS coverage gap analysis after test gen. ~4-5 min/file.</Tip>
        <Tip><strong className="text-zinc-300">Best</strong> — also runs a gap-fill pass to cover any missing requirements. ~6-8 min/file.</Tip>
      </Section>

      <Section title="Auto-Repair">
        <Tip>Enable <em>Auto-Repair Confirmed Bugs</em> in Settings to automatically invoke the Bug Fixer on any files where tests find real bugs. Patched files are saved and a git branch is created if the folder has a <code className="font-mono text-[10px] bg-zinc-800 px-1 rounded">.git</code> directory.</Tip>
      </Section>
    </div>
  ),

  parta: (
    <div className="space-y-5">
      <p className="text-[12px] text-zinc-300 leading-relaxed">
        Part A extracts and classifies requirements from SRS documents or README files using
        spaCy NLP and fuzzy matching against the PURE dataset.
      </p>

      <Section title="SRS Mode">
        <Step n={1}>Select <strong className="text-zinc-300">SRS Document</strong> from the sub-mode toggle.</Step>
        <Step n={2}>Click <em>Upload SRS File</em> and pick a PDF or DOCX.</Step>
        <Step n={3}>Adjust the <strong className="text-zinc-300">Fuzzy Threshold</strong> slider if needed
          (higher = stricter matching, lower = more permissive).</Step>
        <Step n={4}>Click <strong className="text-zinc-300">Extract Requirements</strong>.</Step>
        <Step n={5}>Results appear in the <em>Requirements</em> tab — sentences labelled 1 are requirements,
          0 are non-requirements.</Step>
      </Section>

      <Section title="README Mode">
        <Step n={1}>Select <strong className="text-zinc-300">README / Docs</strong> from the sub-mode toggle.</Step>
        <Step n={2}>Paste your project README text in the text area.</Step>
        <Step n={3}>Optionally enter a project name and describe problems, expected behaviour, and edge cases.</Step>
        <Step n={4}>Click <strong className="text-zinc-300">Extract from README</strong>.</Step>
        <Step n={5}>Results appear in the <em>Scenarios</em> tab — LLM-generated test scenarios from the README.</Step>
      </Section>

      <Section title="Tips">
        <Tip>Use <em>Both</em> sub-mode to run SRS and README extraction in one go.</Tip>
        <Tip>The threshold slider controls how strictly sentences must match the PURE requirement dataset. 0.85 is a good default.</Tip>
      </Section>
    </div>
  ),

  partb: (
    <div className="space-y-5">
      <p className="text-[12px] text-zinc-300 leading-relaxed">
        Part B is a self-correcting AI test generator. It discovers testable Python files in a
        GitHub repo, generates pytest suites using Qwen2.5-Coder-7B + LoRA, runs them, feeds
        errors back to the model, and repeats until tests pass (up to 15 iterations).
      </p>

      <Section title="Workflow">
        <Step n={1}>Paste a <strong className="text-zinc-300">GitHub repo URL</strong> (e.g. <code className="font-mono text-[10px] bg-zinc-800 px-1 rounded">https://github.com/user/repo</code>) and optionally a branch name.</Step>
        <Step n={2}>Click <strong className="text-zinc-300">Discover Files</strong> — the backend clones the repo and finds all testable Python files.</Step>
        <Step n={3}>Check or uncheck files. Use <em>Select All / None</em> to bulk-toggle.</Step>
        <Step n={4}>Click <strong className="text-zinc-300">Generate Tests</strong> (or <Kbd>Ctrl</Kbd>+<Kbd>Enter</Kbd>).</Step>
        <Step n={5}>Watch the live log. Each file goes through the autonomous loop: AST parse → generate → run → fix → repeat.</Step>
        <Step n={6}>Results appear per-file in real time under <em>Test Code</em>, <em>Coverage</em>, and <em>Bugs</em> tabs.</Step>
      </Section>

      <Section title="Key Settings">
        <Tip><strong className="text-zinc-300">Plan Mode</strong> — generates a test plan first; you can review and edit it before code is written (HITL must also be on).</Tip>
        <Tip><strong className="text-zinc-300">HITL Review</strong> — pauses before each file so you can approve or edit the generated tests.</Tip>
        <Tip><strong className="text-zinc-300">Intense Mode</strong> — runs multiple generation iterations per file for better coverage (slower).</Tip>
        <Tip><strong className="text-zinc-300">Base Model Only</strong> — disables the LoRA adapter. Useful for A/B comparison vs the fine-tuned model.</Tip>
      </Section>
    </div>
  ),

  partc: (
    <div className="space-y-5">
      <p className="text-[12px] text-zinc-300 leading-relaxed">
        Bug Fixer (APR) uses Ochiai SBFL to localise faults in a buggy source file,
        then asks the LLM to generate a patch. It re-runs the tests after each patch
        until they pass or the attempt limit is reached.
      </p>

      <Section title="Workflow">
        <Step n={1}>Enter a GitHub repo URL and click <strong className="text-zinc-300">Discover Files</strong>.</Step>
        <Step n={2}>Under <strong className="text-zinc-300">Source File (buggy)</strong>, pick the Python file that contains the bug.</Step>
        <Step n={3}>Under <strong className="text-zinc-300">Test File (failing)</strong>, pick the pytest file that exposes the bug.</Step>
        <Step n={4}>Click <strong className="text-zinc-300">Run Bug Repair</strong>.</Step>
        <Step n={5}>The pipeline runs the tests, collects pass/fail coverage, ranks suspicious lines by Ochiai score, and asks the LLM to patch them.</Step>
        <Step n={6}>Results appear in <em>SBFL</em> (suspicious lines with scores), <em>Patches</em> (each attempt's diff), and <em>Logs</em>.</Step>
      </Section>

      <Section title="Reading the Results">
        <Tip><strong className="text-zinc-300">SBFL tab</strong> — lines ranked by suspiciousness score (1.0 = most suspicious). Green = covered by passing tests only, red = covered by at least one failing test.</Tip>
        <Tip><strong className="text-zinc-300">Patches tab</strong> — each repair attempt shows the unified diff. A green header means the patch made tests pass.</Tip>
        <Tip>If the max attempts are reached without success, try increasing <em>Max Retries</em> in Settings, or manually inspect the SBFL ranking to guide the LLM.</Tip>
      </Section>

      <Section title="Auto-Repair from Full Pipeline">
        <Tip>You don't need to run Part C manually if you use Full Pipeline with <em>Auto-Repair Confirmed Bugs</em> enabled. Part B will flag real bugs and Part C will patch them automatically.</Tip>
      </Section>
    </div>
  ),

  shortcuts: (
    <div className="space-y-5">
      <Section title="Global">
        <div className="space-y-2">
          {[
            { keys: ['?'], desc: 'Open this help panel' },
            { keys: ['Esc'], desc: 'Close any open modal' },
          ].map(s => (
            <div key={s.keys.join('')} className="flex items-center justify-between py-1.5 border-b border-zinc-800/60">
              <span className="text-[12px] text-zinc-400">{s.desc}</span>
              <div className="flex gap-1">{s.keys.map(k => <Kbd key={k}>{k}</Kbd>)}</div>
            </div>
          ))}
        </div>
      </Section>

      <Section title="Workspace">
        <div className="space-y-2">
          {[
            { keys: ['Ctrl', 'Enter'], desc: 'Start / trigger the current run' },
            { keys: ['1'], desc: 'Switch to Full Pipeline mode' },
            { keys: ['2'], desc: 'Switch to Part A mode' },
            { keys: ['3'], desc: 'Switch to Part B mode' },
            { keys: ['4'], desc: 'Switch to Bug Fixer mode' },
          ].map(s => (
            <div key={s.keys.join('')} className="flex items-center justify-between py-1.5 border-b border-zinc-800/60">
              <span className="text-[12px] text-zinc-400">{s.desc}</span>
              <div className="flex items-center gap-1">
                {s.keys.map((k, i) => (
                  <span key={k} className="flex items-center gap-1">
                    {i > 0 && <span className="text-zinc-600 text-[10px]">+</span>}
                    <Kbd>{k}</Kbd>
                  </span>
                ))}
              </div>
            </div>
          ))}
        </div>
      </Section>

      <Section title="Landing Page">
        <div className="space-y-2">
          {[
            { keys: ['1', '2', '3', '4'], desc: 'Jump directly into a mode' },
          ].map(s => (
            <div key={s.keys.join('')} className="flex items-center justify-between py-1.5 border-b border-zinc-800/60">
              <span className="text-[12px] text-zinc-400">{s.desc}</span>
              <div className="flex items-center gap-1">
                {s.keys.map((k, i) => (
                  <span key={k} className="flex items-center gap-1">
                    {i > 0 && <span className="text-zinc-600 text-[10px]">/</span>}
                    <Kbd>{k}</Kbd>
                  </span>
                ))}
              </div>
            </div>
          ))}
        </div>
      </Section>

      <div className="p-3 bg-zinc-800/40 rounded-lg border border-zinc-700/50">
        <p className="text-[11px] text-zinc-500">
          Tip: keyboard shortcuts <Kbd>1</Kbd>–<Kbd>4</Kbd> only fire when the focus is on the document body (not inside an input field).
        </p>
      </div>
    </div>
  ),
}

interface Props {
  onClose: () => void
}

export default function HelpModal({ onClose }: Props) {
  const [activeTab, setActiveTab] = useState('overview')

  return (
    <div className="fixed inset-0 bg-black/60 flex items-center justify-center z-50" onClick={onClose}>
      <div
        className="bg-zinc-900 rounded-xl border border-zinc-700 w-[620px] max-h-[88vh] overflow-hidden shadow-2xl animate-modal-in flex flex-col"
        onClick={e => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-5 py-3.5 border-b border-zinc-800 shrink-0">
          <div className="flex items-center gap-2">
            <div className="w-6 h-6 rounded-md bg-zinc-800 border border-zinc-700 flex items-center justify-center">
              <span className="text-[11px] font-bold text-zinc-300">?</span>
            </div>
            <h2 className="text-sm font-semibold text-zinc-100">TestMate Help</h2>
            <span className="text-[10px] text-zinc-600 font-mono ml-1">/ manual</span>
          </div>
          <button onClick={onClose} className="p-1 hover:bg-zinc-800 rounded transition-colors">
            <X size={18} className="text-zinc-400" />
          </button>
        </div>

        <div className="flex flex-1 min-h-0">
          {/* Tab rail */}
          <div className="w-36 shrink-0 border-r border-zinc-800 py-3 flex flex-col gap-0.5 px-2">
            {TABS.map(tab => {
              const isActive = activeTab === tab.id
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`flex items-center gap-2 px-2.5 py-2 rounded-lg text-left transition-colors text-[12px] font-medium
                    ${isActive
                      ? 'bg-zinc-800 text-zinc-100'
                      : 'text-zinc-500 hover:text-zinc-300 hover:bg-zinc-800/50'
                    }`}
                >
                  <span className={isActive ? tab.color : 'text-zinc-600'}>{tab.icon}</span>
                  {tab.label}
                </button>
              )
            })}
          </div>

          {/* Content */}
          <div className="flex-1 overflow-y-auto">
            <AnimatePresence mode="wait">
              <motion.div
                key={activeTab}
                initial={{ opacity: 0, y: 6 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -4 }}
                transition={{ duration: 0.18 }}
                className="p-5"
              >
                {CONTENT[activeTab]}
              </motion.div>
            </AnimatePresence>
          </div>
        </div>

        {/* Footer */}
        <div className="px-5 py-3 border-t border-zinc-800 flex items-center justify-between shrink-0">
          <p className="text-[10px] text-zinc-600">
            Press <Kbd>?</Kbd> anytime to open this help panel
          </p>
          <button
            onClick={onClose}
            className="px-3 py-1.5 text-xs font-medium text-zinc-300 bg-zinc-800 hover:bg-zinc-700 rounded-lg transition-colors"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  )
}
