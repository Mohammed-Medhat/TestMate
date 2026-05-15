"use client"

import { useState, useCallback } from "react"
import { HeroSection } from "@/components/hero-section"
import { InputSection } from "@/components/input-section"
import { ProgressTimeline } from "@/components/progress-timeline"
import { ResultsDashboard } from "@/components/results-dashboard"

// Determine API URL (Localhost for development)
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000"

interface TestScenario {
  name: string;
  type: string;
  description: string;
  expected_result: string;
  priority: string;
}

interface ResultsData {
  extraction_id?: number
  features: any
  confidence: any
  low_confidence_fields: string[]
  test_scenarios: TestScenario[]
}

/**
 * Main function to communicate with the FastAPI backend.
 * Uses the /api/v1/scenarios/generate_scenarios endpoint.
 */
async function extractFromAPI(
  input: string,
  type: "url" | "file",
  description: string,
  problem: string,
  onStepChange: (step: number) => void
): Promise<ResultsData> {
  onStepChange(1) // Step 1: Starting/Fetching

  try {
    const payload = {
      repo_name: type === "url" ? input : "Uploaded File",
      user_input: {
        description: type === "file" ? input : (description || ""),
        problems: problem || "",
        expected: "", // Can be expanded in UI later
        edge_cases: ""
      }
    }

    // Call the newly organized backend endpoint
    const response = await fetch(`${API_BASE_URL}/api/v1/scenarios/generate_scenarios`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    })

    onStepChange(2) // Step 2: Processing (Backend is running LLM)
    
    if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || "Failed to extract features");
    }

    onStepChange(3) // Step 3: LLM Finishing
    const data = await response.json()
    
    onStepChange(4) // Step 4: Finalizing/Saving
    return {
      extraction_id: data.extraction_id,
      features: data.features || {},
      confidence: data.confidence || {},
      low_confidence_fields: data.low_confidence_fields || [],
      test_scenarios: data.test_scenarios || []
    }
  } catch (error) {
    console.error("API Error:", error);
    throw error
  }
}

export default function HomePage() {
  const [isLoading, setIsLoading] = useState(false)
  const [currentStep, setCurrentStep] = useState(0)
  const [isComplete, setIsComplete] = useState(false)
  const [results, setResults] = useState<ResultsData | null>(null)
  const [error, setError] = useState<string | null>(null)

  const handleSubmit = useCallback(async (input: string, type: "url" | "file", desc?: string, prob?: string) => {
    setIsLoading(true)
    setCurrentStep(0)
    setIsComplete(false)
    setResults(null)
    setError(null)

    try {
      const data = await extractFromAPI(input, type, desc || "", prob || "", setCurrentStep)
      setResults(data)
      setIsComplete(true)
      setCurrentStep(4)
    } catch (err) {
      setError(err instanceof Error ? err.message : "Connection failed. Is the FastAPI server running?")
    } finally {
      setIsLoading(false)
    }
  }, [])

  const handleReset = () => {
    setIsLoading(false)
    setCurrentStep(0)
    setIsComplete(false)
    setResults(null)
    setError(null)
  }

  return (
    <main className="min-h-screen bg-background grid-bg relative overflow-hidden">
      {/* Background Neon Effects */}
      <div className="fixed inset-0 bg-gradient-to-b from-purple-900/10 via-transparent to-pink-900/10 pointer-events-none" />
      
      <div className="relative z-10">
        <HeroSection />
        
        {!results && (
          <InputSection onSubmit={handleSubmit} isLoading={isLoading} />
        )}

        {error && (
          <div className="max-w-xl mx-auto mt-4 p-4 glass border border-red-500/50 rounded-xl text-red-400 text-center">
            {error}
          </div>
        )}

        <ProgressTimeline currentStep={currentStep} isComplete={isComplete} />

        {results && (
          <ResultsDashboard data={results} onReset={handleReset} />
        )}
      </div>
    </main>
  )
}