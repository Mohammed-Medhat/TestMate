"use client"

import { motion } from "framer-motion"
import { Download, Brush, Brain, Database, Check } from "lucide-react"

interface Step {
  id: number
  label: string
  icon: React.ElementType
}

const steps: Step[] = [
  { id: 1, label: "Fetching Repo", icon: Download },
  { id: 2, label: "Cleaning Markdown", icon: Brush },
  { id: 3, label: "LLM Processing", icon: Brain },
  { id: 4, label: "Saving Results", icon: Database },
]

interface ProgressTimelineProps {
  currentStep: number
  isComplete: boolean
}

export function ProgressTimeline({ currentStep, isComplete }: ProgressTimelineProps) {
  if (currentStep === 0 && !isComplete) return null

  return (
    <section className="px-4 py-12">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="max-w-4xl mx-auto"
      >
        <div className="glass-card rounded-2xl p-8">
          <h3 className="text-lg font-semibold text-foreground mb-8 text-center">
            Extraction Pipeline
          </h3>

          {/* Desktop: Horizontal Timeline */}
          <div className="hidden md:flex items-center justify-between">
            {steps.map((step, index) => {
              const Icon = step.icon
              const isActive = currentStep === step.id
              const isCompleted = isComplete || currentStep > step.id

              return (
                <div key={step.id} className="flex items-center flex-1">
                  <div className="flex flex-col items-center">
                    <motion.div
                      initial={{ scale: 0 }}
                      animate={{ scale: 1 }}
                      transition={{ delay: index * 0.1 }}
                      className={`relative w-14 h-14 rounded-full flex items-center justify-center transition-all duration-300 ${
                        isCompleted
                          ? "bg-gradient-to-r from-purple-600 to-pink-600"
                          : isActive
                            ? "bg-gradient-to-r from-purple-600/50 to-pink-600/50 pulse-glow"
                            : "bg-muted border border-purple-500/30"
                      }`}
                    >
                      {isCompleted ? (
                        <Check className="w-6 h-6 text-white" />
                      ) : (
                        <Icon className={`w-6 h-6 ${isActive ? "text-white" : "text-muted-foreground"}`} />
                      )}

                      {isActive && !isComplete && (
                        <motion.div
                          className="absolute inset-0 rounded-full border-2 border-purple-500"
                          animate={{ scale: [1, 1.2, 1], opacity: [1, 0, 1] }}
                          transition={{ duration: 1.5, repeat: Infinity }}
                        />
                      )}
                    </motion.div>

                    <span
                      className={`mt-3 text-sm font-medium transition-colors ${
                        isActive || isCompleted ? "text-foreground" : "text-muted-foreground"
                      }`}
                    >
                      {step.label}
                    </span>
                  </div>

                  {index < steps.length - 1 && (
                    <div className="flex-1 h-1 mx-4 rounded-full bg-muted overflow-hidden">
                      <motion.div
                        className="h-full bg-gradient-to-r from-purple-600 to-pink-600"
                        initial={{ width: "0%" }}
                        animate={{
                          width: isCompleted || currentStep > step.id ? "100%" : "0%",
                        }}
                        transition={{ duration: 0.5 }}
                      />
                    </div>
                  )}
                </div>
              )
            })}
          </div>

          {/* Mobile: Vertical Timeline */}
          <div className="md:hidden space-y-6">
            {steps.map((step, index) => {
              const Icon = step.icon
              const isActive = currentStep === step.id
              const isCompleted = isComplete || currentStep > step.id

              return (
                <motion.div
                  key={step.id}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: index * 0.1 }}
                  className="flex items-center gap-4"
                >
                  <div
                    className={`relative w-12 h-12 rounded-full flex items-center justify-center shrink-0 transition-all duration-300 ${
                      isCompleted
                        ? "bg-gradient-to-r from-purple-600 to-pink-600"
                        : isActive
                          ? "bg-gradient-to-r from-purple-600/50 to-pink-600/50 pulse-glow"
                          : "bg-muted border border-purple-500/30"
                    }`}
                  >
                    {isCompleted ? (
                      <Check className="w-5 h-5 text-white" />
                    ) : (
                      <Icon className={`w-5 h-5 ${isActive ? "text-white" : "text-muted-foreground"}`} />
                    )}
                  </div>

                  <div className="flex-1">
                    <span
                      className={`font-medium transition-colors ${
                        isActive || isCompleted ? "text-foreground" : "text-muted-foreground"
                      }`}
                    >
                      {step.label}
                    </span>
                    {isActive && !isComplete && (
                      <div className="w-full h-1 mt-2 rounded-full bg-muted overflow-hidden">
                        <motion.div
                          className="h-full bg-gradient-to-r from-purple-600 to-pink-600"
                          animate={{ x: ["-100%", "100%"] }}
                          transition={{ duration: 1.5, repeat: Infinity, ease: "easeInOut" }}
                        />
                      </div>
                    )}
                  </div>
                </motion.div>
              )
            })}
          </div>
        </div>
      </motion.div>
    </section>
  )
}
