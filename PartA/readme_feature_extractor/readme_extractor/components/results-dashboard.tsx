"use client"

import { useState, useEffect } from "react"
import { motion, AnimatePresence } from "framer-motion"
import { Button } from "@/components/ui/button"

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000"

interface DisplayFeature {
  name: string
  description: string
  key: string
}

interface Scenario {
  name: string
  type: string
  description: string
  expected_result: string;
  priority: string;
}

/**
 * Converts backend ProjectFeatures object into an editable list for the UI.
 */
function featuresToDisplayList(features: any): DisplayFeature[] {
  if (!features) return []
  
  const labelMap: Record<string, string> = {
    project_name: "Project Name",
    description: "Project Description",
    tech_stack: "Tech Stack",
    installation_commands: "Installation Steps",
    has_tests: "Testing Infrastructure",
    license_type: "License Info"
  }

  return Object.entries(labelMap).map(([key, label]) => {
    const val = features[key]
    let display = ""
    if (Array.isArray(val)) display = val.join(", ")
    else if (typeof val === "boolean") display = val ? "Detected" : "Not Found"
    else display = String(val || "")

    return { key, name: label, description: display }
  });
}

export function ResultsDashboard({ data, onReset }: { data: any, onReset: () => void }) {
  const [editableFeatures, setEditableFeatures] = useState<DisplayFeature[]>([])
  const [editableScenarios, setEditableScenarios] = useState<Scenario[]>([])
  const [isSaving, setIsSaving] = useState(false)

  useEffect(() => {
    if (data) {
      setEditableFeatures(featuresToDisplayList(data.features))
      setEditableScenarios(data.test_scenarios || [])
    }
  }, [data])

  const handleCommit = async () => {
    setIsSaving(true)
    try {
      // Prepare payload to match the save_feedback endpoint in extraction.py
      const payload = {
        extraction_id: data.extraction_id,
        project_name: editableFeatures.find(f => f.key === "project_name")?.description,
        edited_features: editableFeatures,
        edited_scenarios: editableScenarios,
        timestamp: new Date().toISOString()
      }

      const res = await fetch(`${API_BASE_URL}/api/v1/save_feedback`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      })

      if (res.ok) alert("Successfully saved to MongoDB and SQLite!")
      else alert("Failed to save. Check server logs.")
    } catch (err) {
      alert("Error connecting to server.")
    } finally {
      setIsSaving(false)
    }
  }

  return (
    <section className="max-w-6xl mx-auto p-6 space-y-12 pb-20">
      <motion.div 
        initial={{ opacity: 0 }} 
        animate={{ opacity: 1 }}
        className="flex flex-col md:flex-row justify-between items-center gap-6 border-b border-purple-500/20 pb-8"
      >
        <div className="text-center md:text-left">
          <h2 className="text-4xl font-bold neon-text mb-2">Analysis Results</h2>
          <p className="text-muted-foreground">Verify and refine the AI-generated test scenarios.</p>
        </div>
        <div className="flex gap-4">
          <Button onClick={onReset} variant="outline" className="rounded-xl px-6 border-purple-500/30">
            Discard
          </Button>
          <Button 
            onClick={handleCommit} 
            disabled={isSaving}
            className="bg-gradient-to-r from-purple-600 to-pink-600 text-white px-8 rounded-xl font-bold shadow-lg"
          >
            {isSaving ? "Saving..." : "Commit Changes"}
          </Button>
        </div>
      </motion.div>

      {/* Features Grid */}
      <div className="grid md:grid-cols-2 gap-6">
        {editableFeatures.map((feature, idx) => (
          <div key={idx} className="glass-card p-6 rounded-2xl border border-purple-500/10">
            <label className="text-[10px] font-bold text-purple-400 uppercase tracking-widest block mb-2">
              {feature.name}
            </label>
            <textarea
              value={feature.description}
              onChange={(e) => {
                const updated = [...editableFeatures]
                updated[idx].description = e.target.value
                setEditableFeatures(updated)
              }}
              className="w-full bg-transparent border-none text-foreground text-sm resize-none focus:outline-none focus:ring-1 focus:ring-purple-500/30 rounded p-1"
              rows={2}
            />
          </div>
        ))}
      </div>

      {/* Scenarios List */}
      <div className="space-y-6">
        <h3 className="text-2xl font-bold text-pink-500 flex items-center gap-3">
          <span className="w-1.5 h-6 bg-pink-500 rounded-full" />
          Test Scenarios
        </h3>
        
        <div className="space-y-4">
          {editableScenarios.map((scenario, idx) => (
            <div key={idx} className="glass-card p-6 rounded-2xl border border-pink-500/10">
              <div className="grid md:grid-cols-3 gap-6">
                <div className="space-y-4">
                  <input
                    value={scenario.name}
                    onChange={(e) => {
                      const updated = [...editableScenarios];
                      updated[idx].name = e.target.value;
                      setEditableScenarios(updated);
                    }}
                    className="text-lg font-bold bg-transparent border-b border-white/10 w-full outline-none focus:border-pink-500"
                  />
                  <div className="flex gap-2">
                    <span className="text-[10px] bg-pink-500/20 text-pink-300 px-2 py-1 rounded uppercase">
                      {scenario.type}
                    </span>
                    <span className="text-[10px] bg-purple-500/20 text-purple-300 px-2 py-1 rounded uppercase">
                      {scenario.priority}
                    </span>
                  </div>
                </div>
                <div className="md:col-span-2 space-y-4">
                  <textarea
                    value={scenario.description}
                    onChange={(e) => {
                      const updated = [...editableScenarios];
                      updated[idx].description = e.target.value;
                      setEditableScenarios(updated);
                    }}
                    className="w-full bg-black/20 p-3 rounded-xl text-sm outline-none focus:ring-1 focus:ring-pink-500/30"
                    rows={2}
                  />
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}