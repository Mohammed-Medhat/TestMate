# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**TestMate** is an AI-driven system that automates software test generation and bug fixing using LLMs. It integrates multi-teacher knowledge distillation and Retrieval-Augmented Generation (RAG) to create a lightweight, efficient student model.

The repo has three top-level components:

- **PartA** — Requirement extraction from SRS documents (PDF/DOCX) and README files
- **PartB** — Self-correcting test generation using a 3-layer RAG agent
- **unified_app** — Electron desktop app (React + TypeScript) that chains PartA → PartB with a FastAPI backend

## Architecture

### unified_app (Electron desktop)

The primary user-facing product. It bundles PartA and PartB into a single desktop app.

```
unified_app/
├── src/                    # React + TypeScript frontend (Vite + Tailwind)
│   ├── App.tsx             # Root component (tab routing: Combined | PartA | PartB)
│   ├── components/         # Landing, LeftSidebar, MainContent, RightSidebar, etc.
│   └── types.ts            # Shared TypeScript types
├── server.py               # FastAPI backend on port 8080 (SSE streaming, all routes)
├── orchestrator.py         # Combined pipeline: Phase 1 (PartA) → Phase 2 (PartB) → Phase 3/4
├── model_lifecycle.py      # GPU context managers: part_a_model(), part_b_model()
├── requirement_matcher.py  # Hybrid keyword+semantic matcher — routes SRS reqs to files
├── coverage_analyzer.py    # SRS coverage gap analysis (balanced/best quality modes)
├── electron.vite.config.ts # Electron-Vite build config
└── package.json            # npm scripts: dev, build, build:win
```

**Running the unified app:**
```bash
cd unified_app
npm install
npm run dev          # development (Electron + Vite hot-reload)
npm run build:win    # package as Windows installer
python server.py     # or run the Python backend standalone on port 8080
```

**API routes (port 8080):**
- `POST /api/combined/run` — Full pipeline (PartA → PartB), SSE-streamed via job_id
- `GET  /api/combined/stream?job_id=<id>` — SSE event stream for combined runs
- `POST /api/parta/srs/run` — SRS extraction only (multipart/form-data, upload PDF/DOCX)
- `POST /api/parta/readme/run` — README extraction only (JSON body)
- `POST /api/partb/discover` — Clone/scan repo, return list of testable files
- `POST /api/partb/run` — Test generation only (SSE-streamed)
- `GET  /api/partb/stream?job_id=<id>` — SSE event stream for PartB-only runs
- `GET  /api/history` — Recent run records

### Combined Pipeline (orchestrator.py)

The combined pipeline runs in 4 phases (not all phases run in all quality modes):

1. **Phase 1 – PartA** (once per project): SRS pipeline runs in-process; README extractor runs in a **subprocess** to free VRAM before PartB loads.
2. **Phase 2 – PartB** (per file): Each target file gets the most relevant requirements via hybrid match, then tests are generated.
3. **Phase 3 – SRS coverage** (balanced/best only): Checks which requirements have no test coverage.
4. **Phase 4 – Gap-fill** (best only): Re-runs generation targeting uncovered requirements.

**Quality modes:** `fast` (phases 1–2) | `balanced` (phases 1–3) | `best` (phases 1–4)

### Critical VRAM constraint

Part A's LLM and Part B's Qwen model **cannot coexist in GPU memory**. The `model_lifecycle.py` module provides context managers (`part_a_model()`, `part_b_model()`) that load, yield, then delete+flush the model. The README extractor specifically runs in a subprocess (`extractor_subprocess.py`) because BitsAndBytes 4-bit models on Windows don't release VRAM cleanly with `del+empty_cache` in the same process.

### PartA: Two sub-components

**1. SRS Pipeline** (`PartA/srs_pipeline/`):
- No GPU required — pure NLP (spaCy + difflib)
- Pipeline: extract text (PDF/DOCX) → segment sentences → fuzzy-match against PURE dataset → heuristic fallback labeler
- Entry points: `run_pipeline.py` (CLI), `srs_api.py` (programmatic API)
- Default aligned dataset: `data/aligned/pure_train_aligned.json`

```bash
cd PartA/srs_pipeline
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python scripts/align_all_datasets.py          # build aligned PURE dataset
python run_pipeline.py --input doc.pdf --output out.json \
  --aligned-dataset data/aligned/pure_train_aligned.json
```

**2. README Extractor** (`PartA/readme_extractor/`):
- GPU required — runs an HF LLM (loaded via `model_loader.py`)
- Pipeline: clean README → extract project features (LLM) → generate test scenarios (LLM)
- Entry points: `readme_api.py` (programmatic API), `extractor_subprocess.py` (subprocess wrapper)

```bash
cd PartA/readme_extractor
python download_model.py          # download HF model weights
python src/main.py                # standalone run
```

### PartB: 3-Layer RAG for Test Generation

```
PartB/
├── agent.py                # SWE-bench agent orchestrator (Layer 1+2+3)
├── layer1_docs/            # RAGFix: FAISS-based external doc search
├── layer2_code/            # KGCompass: BM25 + semantic code-graph navigation
├── testgen/
│   ├── main.py             # autonomous_loop() — AST → LLM → test → error → retry
│   ├── testgen_api.py      # execute_part_b() — clean API used by unified server
│   ├── api_server.py       # Standalone FastAPI on port 8000 (HTML GUI, HITL review)
│   ├── rag_store.py        # SQLite RAG memory (good tests, failure patterns)
│   ├── existing_test_scanner.py   # Auto-priming: scan repo for style examples
│   ├── testable_filter.py         # Skip empty/__init__/test files before generation
│   ├── langchain/          # LangChain-based agent wrappers
│   ├── training/           # Kaggle training + evaluation scripts (T4 GPU)
│   └── generated_tests/    # Output: test_<name>.py files
├── models/
│   └── graphrag_lora/final/    # LoRA adapter (applied on top of Qwen2.5-Coder-7B)
└── value_reasoning_model/      # Second LoRA adapter
```

**PartB standalone commands:**
```bash
cd PartB
pip install -r testgen/requirements_docker.txt

python testgen/api_server.py              # start standalone backend (port 8000)
python testgen/main.py --target path/to/code.py   # single file generation
pytest PartB/test_graphrag.py -v
pytest PartB/test_context_assembler.py -v

# Kaggle training
python testgen/training/kaggle_train_testgen.py   # outputs to models/graphrag_lora/final/
python testgen/training/kaggle_eval_testgen.py
```

**Self-correcting test generation loop (`main.py → autonomous_loop()`):**
1. Parse target file AST (CFG paths, raises, callees, docstrings)
2. Optionally run plan-then-generate (plan_mode=True)
3. Generate pytest with Qwen2.5-Coder-7B + LoRA (max 15 iterations)
4. Execute tests, capture errors, feed back to LLM
5. Store good tests in SQLite RAG memory; tag bad patterns for learning

## Key Architectural Decisions

- **Subprocess isolation for VRAM**: README extractor runs in a child process so GPU memory is fully freed before PartB loads its model.
- **Modular API surface**: Each component has a clean `execute_*()` function (`execute_part_a_srs`, `execute_part_a_readme`, `execute_part_b`). The Electron server calls these; they each handle their own model loading if no pre-loaded model is passed.
- **Auto-priming**: Before generation, the unified pipeline scans the target repo for existing passing tests and injects them as style examples.
- **Testable filter**: `testable_filter.py` skips empty files, `__init__.py`, and already-existing test files before invoking the (expensive) LLM loop.
- **SSE streaming**: All long-running jobs return a `job_id` immediately; the frontend polls `/stream?job_id=…` for Server-Sent Events, including per-file `file_result` events so UI updates incrementally.

## PartA → PartB Integration

- PartA's labeled requirements (`label=1`) are passed into PartB's `autonomous_loop()` as `srs_requirements`.
- `requirement_matcher.py` does hybrid keyword + semantic matching to pick the top-k most relevant requirements for each target file (avoids flooding LLM context with all project requirements).
- SRS coverage is measured post-generation by checking whether requirement keywords appear in the test code.
