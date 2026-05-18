export type View = 'landing' | 'workspace'
export type Mode = 'combined' | 'parta' | 'partb'
export type PartAMode = 'srs' | 'readme' | 'both'
export type RunStatus = 'idle' | 'running' | 'done' | 'error'
export type ActiveTab = 'bugs' | 'coverage' | 'mutations' | 'requirements' | 'scenarios' | 'testcode' | 'features'

// ── Matching PartB/gui exactly ──────────────────────────────────────────────
export interface FileInfo {
  path: string
  abs_path: string
  import_path: string | null
  lines: number
  functions: number
  classes: number
  func_names: string[]
  class_names: string[]
}

export interface ReviewRequest {
  test_code: string
  test_file: string
  num_tests: number
  target_file: string
  file_index: number
  file_total: number
}

export interface PlanRequest {
  plan: string
  target: string
  test_file: string
  method_name: string
  class_name: string
}

export type QualityMode = 'fast' | 'balanced' | 'best'

export interface RunSettings {
  docker: boolean
  deepScan: boolean
  maxRetries: number
  hitl: boolean
  intense: boolean
  planMode: boolean
  useBaseOnly: boolean   // skip value_reasoning LoRA — raw 4-bit Qwen base
  qualityMode: QualityMode  // fast=LoRA+plan | balanced=+gap-analysis | best=+gap-fill
}

// ── History ─────────────────────────────────────────────────────────────────
export interface HistoryItem {
  id: string
  mode: Mode
  timestamp: string
  summary: string
  status: 'success' | 'error'
}

// ── Part A ──────────────────────────────────────────────────────────────────
export interface Requirement {
  text: string
  label: number    // 1=req, 0=non-req, -1=unknown
  match_type: string
  score: number
  page?: number
  sentence_id?: string
}

export interface Scenario {
  name: string
  type: 'positive' | 'negative' | 'edge' | 'boundary'
  description: string
  expected_result: string
  priority: 'high' | 'medium' | 'low'
}

export interface PartAStats {
  total: number
  requirements: number
  non_requirements: number
  unlabeled: number
}

// ── Progress ─────────────────────────────────────────────────────────────────
export interface Progress {
  current: number
  total: number
  file: string
}
