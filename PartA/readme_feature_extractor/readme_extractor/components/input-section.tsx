"use client"

import { useState, useRef } from "react"
import { motion } from "framer-motion"
import { Github, Upload, Wand2, FileText, X, MessageSquare, AlertCircle } from "lucide-react"
import { Button } from "@/components/ui/button"

interface InputSectionProps {
  onSubmit: (input: string, type: "url" | "file", description?: string, problem?: string) => void
  isLoading: boolean
}

export function InputSection({ onSubmit, isLoading }: InputSectionProps) {
  const [repoUrl, setRepoUrl] = useState("")
  const [userDescription, setUserDescription] = useState("")
  const [specificProblem, setSpecificProblem] = useState("")
  const [dragActive, setDragActive] = useState(false)
  const [droppedFile, setDroppedFile] = useState<File | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleDrag = (e: React.DragEvent) => { /* ... نفس الكود بتاعك ... */
    e.preventDefault()
    e.stopPropagation()
    if (e.type === "dragenter" || e.type === "dragover") setDragActive(true)
    else if (e.type === "dragleave") setDragActive(false)
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setDragActive(false)
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0]
      if (file.name.toLowerCase().endsWith(".md")) {
        setDroppedFile(file)
        setRepoUrl("")
      }
    }
  }

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setDroppedFile(e.target.files[0])
      setRepoUrl("")
    }
  }

  const handleSubmit = async () => {
    if (droppedFile) {
      const content = await droppedFile.text()
      onSubmit(content, "file", userDescription, specificProblem)
    } else if (repoUrl.trim() || userDescription.trim()) {
      onSubmit(repoUrl.trim(), "url", userDescription, specificProblem)
    }
  }

  return (
    <section className="px-4 py-12">
      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5 }} className="max-w-3xl mx-auto">
        <div className="glass-card rounded-2xl p-8 neon-border">
          
          {/* URL Input */}
          <div className="relative mb-6">
            <div className="absolute inset-y-0 left-4 flex items-center pointer-events-none">
              <Github className="w-5 h-5 text-purple-400" />
            </div>
            <input
              type="text"
              placeholder="https://github.com/username/repository"
              value={repoUrl}
              onChange={(e) => { setRepoUrl(e.target.value); if (e.target.value) setDroppedFile(null) }}
              disabled={isLoading}
              className="w-full pl-12 pr-4 py-4 bg-background/50 rounded-xl border border-purple-500/30 text-foreground placeholder-muted-foreground focus:outline-none focus:border-purple-500 focus:ring-2 focus:ring-purple-500/20 transition-all disabled:opacity-50"
            />
          </div>

          {/* User Inputs (New) */}
          <div className="grid md:grid-cols-2 gap-4 mb-6">
            <div className="relative">
              <div className="absolute top-4 left-4"><MessageSquare className="w-5 h-5 text-purple-400" /></div>
              <textarea
                placeholder="Custom Description (Optional)"
                value={userDescription}
                onChange={(e) => setUserDescription(e.target.value)}
                disabled={isLoading}
                className="w-full h-24 pl-12 pr-4 py-4 bg-background/50 rounded-xl border border-purple-500/30 text-foreground placeholder-muted-foreground focus:outline-none focus:border-purple-500 focus:ring-2 focus:ring-purple-500/20 transition-all resize-none"
              />
            </div>
            <div className="relative">
              <div className="absolute top-4 left-4"><AlertCircle className="w-5 h-5 text-pink-400" /></div>
              <textarea
                placeholder="Specific problem to test (Optional)"
                value={specificProblem}
                onChange={(e) => setSpecificProblem(e.target.value)}
                disabled={isLoading}
                className="w-full h-24 pl-12 pr-4 py-4 bg-background/50 rounded-xl border border-pink-500/30 text-foreground placeholder-muted-foreground focus:outline-none focus:border-pink-500 focus:ring-2 focus:ring-pink-500/20 transition-all resize-none"
              />
            </div>
          </div>

          {/* Submit Button */}
          <motion.div className="mt-8" whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.98 }}>
            <Button
              onClick={handleSubmit}
              disabled={isLoading || (!repoUrl.trim() && !droppedFile && !userDescription.trim())}
              className="w-full py-6 text-lg font-semibold bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-500 hover:to-pink-500 text-white border-0 rounded-xl pulse-glow disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isLoading ? "Processing..." : <><Wand2 className="w-5 h-5 mr-2" /> Start Extraction</>}
            </Button>
          </motion.div>
        </div>
      </motion.div>
    </section>
  )
}