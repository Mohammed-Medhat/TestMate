# TestMate — AI-Powered Test Generation, Bug Detection & Automated Repair

## Overview

**TestMate** is an end-to-end AI system that automates the full software quality lifecycle:
requirements extraction → test generation → bug detection → automated program repair.

It runs as an **Electron desktop app** (React + TypeScript frontend, FastAPI backend) built around a fine-tuned **Qwen2.5-Coder-7B** student model trained via multi-teacher knowledge distillation.

---

## Architecture — Three Parts

```
┌─────────────────────────────────────────────────────────────────────┐
│  Part A  — Requirement Extraction                                    │
│  SRS (PDF/DOCX) or README → labeled requirements + test scenarios   │
│  Pure NLP (spaCy + difflib) for SRS  │  LLM-based for README        │
└──────────────────────────┬──────────────────────────────────────────┘
                           │ requirements (label=1)
┌──────────────────────────▼──────────────────────────────────────────┐
│  Part B  — Self-Correcting Test Generation  (3-Layer RAG)           │
│  Qwen2.5-Coder-7B + LoRA  │  AST → LLM → test → error → retry      │
│                                                                     │
│  NEW: Bug Exposure Score (BES)  — 5-dim quality gate                │
│    1. Spec coverage   (30%) — docstring examples as assertions       │
│    2. Input variety   (25%) — case / sign / type characteristic mix  │
│    3. Boundary cov.   (20%) — AST-verified boundary args            │
│    4. Assertion qual. (15%) — exact ==, raises, distinct targets     │
│    5. Mutation pot.   (10%) — heuristic mutation-killing signal      │
│                                                                     │
│  NEW: Layer 1 — docstring extractor (deterministic spec tests)      │
│  NEW: Layer 2 — boundary synthesizer (type-aware edge cases)        │
│  NEW: Pre-pass triage — real bug vs stale test disambiguation        │
└──────────────────────────┬──────────────────────────────────────────┘
                           │ confirmed/suspected bugs → bug_reports.jsonl
┌──────────────────────────▼──────────────────────────────────────────┐
│  Part C  — Automated Program Repair  (APR)                          │
│  SBFL fault localisation (Ochiai) → Qwen2.5-Coder-7B + LoRA        │
│  AST-safe patching → workspace isolation → git branch               │
└─────────────────────────────────────────────────────────────────────┘
```

### Unified Desktop App

```
unified_app/
├── src/                     # React + TypeScript (Vite + Tailwind)
│   ├── App.tsx              # Root — 4 modes: Combined | PartA | PartB | PartC
│   ├── components/          # Landing, LeftSidebar, MainContent, RightSidebar…
│   └── types.ts             # Shared TypeScript types (incl. BES + prepass types)
├── server.py                # FastAPI :8080 — all routes, SSE streaming
├── orchestrator.py          # Combined pipeline: A→B→C with quality modes
├── model_lifecycle.py       # GPU context managers + multi-adapter swap (partb/partc)
├── requirement_matcher.py   # Hybrid keyword + semantic matcher
└── coverage_analyzer.py     # SRS coverage gap analysis
```

---

## Key Features

### Part A
- **SRS Pipeline** — PDF/DOCX → spaCy sentence segmentation → PURE dataset fuzzy alignment → heuristic fallback labeler
- **README Extractor** — cleans README → LLM feature extraction → LLM test scenario generation (subprocess-isolated for VRAM safety)

### Part B (Test Generation)
- **3-Layer RAG** — Layer 1: FAISS external docs · Layer 2: BM25 + semantic code graph · Layer 3: Qwen2.5-Coder-7B + LoRA
- **Self-correcting loop** — generate → run → capture error → regenerate (max 15 iterations)
- **Bug Exposure Score (BES)** — replaces old 3-dim composite; detects tests that look good but miss bugs
- **Layer 1 amplifier** — docstring extractor guarantees spec examples are always present in the test file
- **Pre-pass triage** — before generating, checks if existing tests already fail; uses LLM to confirm "two independent failures = real bug" and routes directly to PartC
- **Auto-priming** — scans repo for passing tests and injects as style examples
- **Testable filter** — skips `__init__`, empty files, and existing test files
- **Mutation testing** — mutmut integration with bug report generation

### Part C (Automated Program Repair)
- **SBFL fault localisation** — Ochiai score per line using coverage spectrum
- **Workspace isolation** — source never modified in place; all repair in `PartC/workspaces/<id>/`
- **AST-safe patching** — replaces only matching functions; full-file fallback for syntax errors
- **Multi-adapter GPU sharing** — PartB and PartC adapters loaded simultaneously via PEFT named adapters; cheap hot-swap between them
- **Git branch creation** — successful patches committed to `testmate-repair-<timestamp>` branch
- **Verdict update** — PartC success/failure promotes or demotes PartB's bug confidence score

### Unified App GUI
- **4 modes** — Full Pipeline (A→B→C) · Part A Only · Part B Only · Bug Fixer (PartC standalone)
- **SSE streaming** — all long jobs return `job_id` immediately; frontend polls `/stream?job_id=…`
- **Stale Tests tab** — shows pre-pass results: existing tests found, real bugs confirmed, stale tests auto-updated
- **Auto-Repair toggle** — opt-in setting; when enabled, confirmed bugs automatically trigger PartC repair
- **Quality Modes** — Fast (A+B) · Balanced (+gap analysis) · Best (+gap fill)

---

## Bug Exposure Score (BES) — New Scoring System

The old scorer (3 dimensions, weighted by literal counts) gave PartB's weak tests 80/100, accepting tests that missed every bug. BES focuses on **what makes a test catch bugs**:

| | Old scorer | New BES |
|---|---|---|
| Weak test (all lowercase inputs) | **80/100 → ACCEPTED** | **36.5/100 → REJECTED** |
| Good test (docstring examples + variety) | 100/100 | **82.5/100 → ACCEPTED** |
| Bugs detected | 0/5 | 3/5 |
| Test functions generated | 1 | 13 |

The BES gate forces up to 3 retries with specific hints (`"Add uppercase input — docstring says case-insensitive"`), then falls back to deterministic Layer 1 docstring tests.

---

## Project Structure

```
TestMate/
├── PartA/
│   ├── srs_pipeline/           # SRS → labeled requirements
│   │   ├── srs_api.py          # execute_part_a_srs()
│   │   └── run_pipeline.py     # CLI
│   └── readme_extractor/       # README → features + scenarios
│       ├── readme_api.py       # execute_part_a_readme()
│       └── extractor_subprocess.py  # VRAM-isolated subprocess wrapper
│
├── PartB/
│   ├── testgen/
│   │   ├── main.py             # autonomous_loop() — core generation + BES gate
│   │   ├── testgen_api.py      # execute_part_b() — clean API
│   │   ├── bes_scorer.py       # NEW: 5-dim Bug Exposure Score
│   │   ├── docstring_extractor.py  # NEW: Layer 1 spec tests
│   │   ├── boundary_synthesizer.py # NEW: Layer 2 boundary tests
│   │   ├── bug_detector.py     # NEW: Oracle 4 + triage confidence scoring
│   │   ├── bug_to_partc.py     # NEW: PartB→PartC bridge + adapter swap
│   │   ├── existing_test_runner.py # NEW: pre-pass triage
│   │   ├── rag_store.py        # SQLite RAG memory (BES-rescored)
│   │   ├── api_server.py       # Standalone FastAPI :8000
│   │   └── pipeline_test/      # Test scenario: buggy_calculator.py
│   ├── models/
│   │   └── graphrag_lora/final/    # PartB LoRA adapter
│   └── value_reasoning_model/      # Second LoRA (value reasoning)
│
├── PartC/
│   ├── api/
│   │   └── partc_api.py        # execute_part_c() — clean API
│   ├── core/
│   │   ├── control_loop.py     # repair_loop() — SBFL → patch → verify
│   │   ├── run_and_collect.py  # per-test coverage (Ochiai spectrum)
│   │   ├── sbfl_localiser.py   # Ochiai scoring
│   │   ├── model_runner.py     # Qwen inference (accepts pre-loaded model)
│   │   └── inference.py        # 4-bit BitsAndBytes loader
│   ├── models/adapter/         # PartC LoRA adapter weights
│   └── training/               # QLoRA fine-tuning scripts
│
└── unified_app/
    ├── src/                    # React + TypeScript (4-mode workspace)
    ├── server.py               # FastAPI :8080 + SSE streaming
    ├── orchestrator.py         # 5-phase combined pipeline
    ├── model_lifecycle.py      # GPU lifecycle + multi-adapter session
    └── package.json            # npm: dev | build:win
```

---

## Running

### Unified Desktop App (recommended)

```bash
cd unified_app
npm install
npm run dev           # Electron + Vite hot-reload
python server.py      # or run backend standalone on :8080
```

### Part B standalone (test generation)

```bash
cd PartB
pip install -r testgen/requirements_docker.txt
python testgen/api_server.py          # FastAPI on :8000
python testgen/main.py --target path/to/code.py  # single file
```

### Part C standalone (bug repair)

```bash
cd PartC
python web/app.py                     # Flask UI on :5000
python core/control_loop.py           # CLI mode
```

### Part A standalone (requirement extraction)

```bash
cd PartA/srs_pipeline
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python run_pipeline.py --input doc.pdf --output out.json
```

---

## Pipeline Test

A self-contained test scenario is included in `PartB/testgen/pipeline_test/`:

```
pipeline_test/
├── buggy_calculator.py          # 5 intentional logic bugs
├── test_buggy_calculator.py     # human-written tests (13 functions)
├── verify_calculator.py         # comprehensive verification suite
└── run_pipeline_test.py         # automated smoke test (6 checks)
```

Run the smoke test (no GPU needed — mocks the model):

```bash
cd PartB/testgen
python pipeline_test/run_pipeline_test.py
```

Expected output: `6/6 tests passed` — verifies Oracle 4, BES scoring, bug grouping, workspace isolation, verdict transitions.

---

## Critical VRAM Constraint

**PartA and PartB/C models cannot coexist in GPU memory.**

- `model_lifecycle.py` provides context managers that load → yield → delete+flush
- README extractor runs in a **subprocess** (`extractor_subprocess.py`) because BitsAndBytes 4-bit models on Windows don't release VRAM cleanly with `del+empty_cache` in the same process
- PartB + PartC share the same Qwen2.5-Coder-7B base; PEFT named adapters (`"partb"`, `"partc"`) allow a cheap hot-swap (~10ms) instead of a full model reload

---

## References

- **KGCompass** (arXiv 2025) — multi-hop graph traversal for code navigation
- **RAGFix** (NSF/IEEE 2025) — external knowledge retrieval for bug fixing
- **RAG Traceability** (MCSE 2025) — requirement-code linking
- **Knowledge Distillation** (arXiv 2025) — efficient training from teacher models
- Knowledge Distillation (Chen et al., 2024)
- Automated Program Repair (Zhang et al., 2023; Meng et al., 2024)
- Retrieval-Augmented Generation (Gao et al., 2023; Zhao et al., 2024)
