import React, { useState } from 'react'
import {
  ChevronRight,
  ChevronDown,
  Folder,
  FolderOpen,
  FileCode,
  FileText,
  File,
} from 'lucide-react'

const TREE = {
  name: 'srs2testcases',
  type: 'folder',
  defaultOpen: true,
  children: [
    { name: 'run_pipeline.py', type: 'file', ext: 'py' },
    { name: 'requirements.txt', type: 'file', ext: 'txt' },
    {
      name: 'src',
      type: 'folder',
      children: [
        {
          name: 'preprocessing',
          type: 'folder',
          children: [
            { name: 'align_pure_datasets.py', type: 'file', ext: 'py' },
            { name: 'segment_sentences.py', type: 'file', ext: 'py' },
            { name: 'reader_pdf.py', type: 'file', ext: 'py' },
            { name: 'reader_docx.py', type: 'file', ext: 'py' },
          ],
        },
        { name: 'models', type: 'folder', children: [] },
        { name: 'generation', type: 'folder', children: [] },
        { name: 'validation', type: 'folder', children: [] },
      ],
    },
    {
      name: 'scripts',
      type: 'folder',
      children: [
        { name: 'align_all_datasets.py', type: 'file', ext: 'py' },
        { name: 'test_alignment.py', type: 'file', ext: 'py' },
        { name: 'verify_alignment.py', type: 'file', ext: 'py' },
        { name: 'test_pipeline.py', type: 'file', ext: 'py' },
      ],
    },
    {
      name: 'data',
      type: 'folder',
      children: [
        { name: 'raw', type: 'folder', children: [] },
        { name: 'processed', type: 'folder', children: [] },
        { name: 'aligned', type: 'folder', children: [] },
      ],
    },
    { name: 'results', type: 'folder', children: [] },
  ],
}

function fileIcon(ext) {
  switch (ext) {
    case 'py':
      return <FileCode size={13} className="flex-none text-emerald-400" />
    case 'json':
      return <FileCode size={13} className="flex-none text-yellow-400" />
    case 'txt':
      return <FileText size={13} className="flex-none text-zinc-400" />
    default:
      return <File size={13} className="flex-none text-zinc-500" />
  }
}

function TreeNode({ node, depth, onSelect, selected }) {
  const [open, setOpen] = useState(node.defaultOpen ?? depth < 1)
  const indent = depth * 10

  if (node.type === 'folder') {
    return (
      <div>
        <button
          onClick={() => setOpen((o) => !o)}
          style={{ paddingLeft: 8 + indent }}
          className="flex w-full items-center gap-1 py-[3px] pr-2 text-left transition-colors hover:bg-zinc-800/60"
        >
          <span className="flex-none text-zinc-600">
            {open ? <ChevronDown size={11} /> : <ChevronRight size={11} />}
          </span>
          {open ? (
            <FolderOpen size={13} className="flex-none text-emerald-400/80" />
          ) : (
            <Folder size={13} className="flex-none text-emerald-500/60" />
          )}
          <span className="truncate font-mono text-[11px] text-zinc-300">{node.name}</span>
        </button>
        {open && node.children?.length > 0 && (
          <div>
            {node.children.map((child, i) => (
              <TreeNode
                key={i}
                node={child}
                depth={depth + 1}
                onSelect={onSelect}
                selected={selected}
              />
            ))}
          </div>
        )}
      </div>
    )
  }

  const isActive = selected === node.name
  return (
    <button
      onClick={() => onSelect(node.name)}
      style={{ paddingLeft: 8 + indent }}
      className={`flex w-full items-center gap-1.5 py-[3px] pr-2 text-left transition-colors ${
        isActive
          ? 'bg-emerald-500/10 text-emerald-400'
          : 'text-zinc-400 hover:bg-zinc-800/60 hover:text-zinc-200'
      }`}
    >
      {fileIcon(node.ext)}
      <span className="truncate font-mono text-[11px]">{node.name}</span>
    </button>
  )
}

export default function FileTree({ onFileSelect }) {
  const [selected, setSelected] = useState(null)

  function handleSelect(name) {
    setSelected(name)
    onFileSelect?.(name)
  }

  return <TreeNode node={TREE} depth={0} onSelect={handleSelect} selected={selected} />
}
