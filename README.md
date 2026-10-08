# TestMate: Local AI Test Generation, Bug Detection & Automated Repair

TestMate is a fully local, offline system that reads a codebase, generates `pytest` suites, finds real bugs, and proposes verified fixes. It runs a fine-tuned **Qwen2.5-Coder-7B-Instruct** (4-bit) on your own GPU. No code ever leaves your machine and no external API is called.

Graduation project, Faculty of Computer and Information Sciences, Ain Shams University (2025–2026).

## Results at a glance

| Task | Benchmark | Result |
|---|---|---|
| Test generation (coverage) | TestEval, 210 LeetCode programs | **93.8%** line coverage, **97.6%** pass rate (base model: 78.9% / 79.8%) |
| Test generation (usable suites) | TestEval | Passing suites for **205 / 210** programs (base: 168 / 210) |
| Test correctness | HumanEval (164), test generation | **60.4%** pass@1 vs 33.5% for the base model (+26.9 pp) |
| Automated repair | HumanEvalFix (164 bugs) | **91.46%** repaired vs 67.68% baseline |
| RAG on real framework code | testgenevallite (N=30, directional) | Pass@1 **13.3%** vs 6.7% with RAG off |

Every number traces to a result JSON in this repo (see [Reproducing the results](#reproducing-the-results)). Known limitations are listed [at the end](#limitations).

## Table of contents

1. [Overview](#overview)
2. [Results](#results)
3. [Architecture](#architecture)
4. [Features](#features)
5. [Model and training](#model-and-training)
6. [Project structure](#project-structure)
7. [Running](#running)
8. [Reproducing the results](#reproducing-the-results)
9. [Limitations](#limitations)
10. [References](#references)

## Overview

TestMate automates the software quality loop in three parts:

| Part | What it does |
|---|---|
| **A: Requirement extraction** | Turns an SRS (PDF/DOCX) or a README into labeled requirements and test scenarios. |
| **B: Test generation** | Generates pytest suites with RAG, a self-correcting generate → run → fix loop, and a quality gate. |
| **C: Automated program repair** | Localizes faults with SBFL (Ochiai) and patches them with the fine-tuned model, verified by the tests. |

The full pipeline runs as an **Electron desktop app** (React + TypeScript frontend, FastAPI backend), and each part can also run standalone.

The central finding of the project: **coverage alone is a misleading metric for LLM-generated tests.** The same system scores 93.8% or 42.5% coverage depending only on its objective, and high coverage can coexist with wrong assertions. TestMate therefore evaluates three axes:

| Axis | Metric | Question it answers |
|---|---|---|
| Coverage | line / branch % | Did the tests execute the code? |
| Correctness | pass@1 % | Are the assertions actually right? |
| Bug detection | mutation kill % | Would the tests catch a real injected bug? |

## Results

### Test generation: TestEval (210 LeetCode programs, suite mode)

Model: Qwen2.5-Coder-7B-Instruct (4-bit NF4) + LoRA trained on decontaminated MBPP only. Baseline: the same base model without the adapter.

| System | Line cov % | Pass rate % | Mutation kill % | Programs with passing tests |
|---|---|---|---|---|
| **TestMate** | **93.8** | **97.6** | 81.5 | 205 / 210 |
| Base (no LoRA) | 78.9 | 79.8 | 89.8 | 168 / 210 |

- Coverage of 93.8% is at the top of the open-source 7B models reported in the TestEval paper.
- Correctness is **+18 points** higher and **37 more programs** get a usable suite.
- Bug detection is a trade-off, not a clean sweep: the base model has a higher per-file mutation kill rate (89.8% vs 81.5%) but only over the 168 programs it handles. TestMate produces bug-catching tests on about 22% more programs.
- Coverage was computed with a re-implementation of the official TestEval protocol, not the official harness.

### Coverage vs correctness: the same system, two objectives

| TestEval mode | Line cov % | Pass@1 % |
|---|---|---|
| Suite mode (optimizes coverage) | 93.8 | n/a (reported as pass rate above) |
| Quality mode (optimizes per-target correctness) | 42.5 | 16.7 |

The spread between 42.5% and 93.8% "coverage" for one system is the evidence that coverage alone should not be trusted.

### Sanity check: HumanEval (164 problems, test generation)

| System | Pass@1 % | Line cov % | Valid line cov % |
|---|---|---|---|
| **TestMate** | **60.4** | 67.6 | 92.6 |
| Base (no LoRA) | 33.5 | 34.1 | 63.5 |

This is test generation on HumanEval problems, not the code-generation leaderboard task. It is used only to confirm that the pipeline helps consistently with TestEval.

### RAG on real framework code: testgenevallite (matched N=30)

| System | Pass@1 % | Line cov % | graphRAG hit | vectorRAG hit |
|---|---|---|---|---|
| **TestMate** (RAG on) | **13.3** | 11.7 | 27% | 57% |
| No RAG | 6.7 | 14.7 | 0% | 0% |

Retrieval doubles correctness on real Django/SymPy-style code but gives no coverage lift (these files are hard: 35–69 targets each with deep dependencies). With N=30 (about 4 vs 2 passing files) this is **directional**, not statistically firm.

### Automated repair: HumanEvalFix (164 bugs)

![Repair rate comparison](docs/repair_rate_comparison.png)

| Mode | Bugs fixed | Repair rate | Avg time / bug | Avg attempts |
|---|---|---|---|---|
| `base_paper` (untuned base model) | 111 / 164 | 67.68% | 23.3 s | 1.00 |
| `finetuned_zero` (fine-tuned, single attempt) | 138 / 164 | 84.15% | 31.1 s | 1.00 |
| `finetuned_full` (fine-tuned + full repair loop) | 150 / 164 | **91.46%** | 44.0 s | 1.25 |

- Fine-tuning alone adds **+16.5 pp** (67.68% → 84.15%).
- The retry loop adds a further **+7.3 pp** (84.15% → 91.46%): 140 bugs were fixed on the first attempt, 7 on the second, and 3 on the third or later.
- The cost is time: the full pipeline takes about 1.9× longer per bug than the baseline.

### Bug Exposure Score (BES) example

BES is a five-dimension quality gate that rejects tests that look good (high coverage) but would miss bugs. On the bundled `buggy_calculator.py` scenario (5 intentional logic bugs):

| | Old 3-dimension scorer | **BES** |
|---|---|---|
| Weak test (all lowercase inputs) | 80/100, accepted | 36.5/100, **rejected** |
| Good test (docstring examples + variety) | 100/100 | 82.5/100, accepted |
| Bugs detected | 0 / 5 | 3 / 5 |
| Test functions generated | 1 | 13 |

This is one illustrative scenario, not a benchmark. The BES gate forces up to 3 retries with specific hints (for example "add uppercase input, the docstring says case-insensitive") and then falls back to deterministic docstring tests.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│  Part A: Requirement Extraction                                      │
│  SRS (PDF/DOCX) or README → labeled requirements + test scenarios   │
│  Pure NLP (spaCy + difflib) for SRS  │  LLM-based for README        │
└──────────────────────────┬──────────────────────────────────────────┘
                           │ requirements (label=1)
┌──────────────────────────▼──────────────────────────────────────────┐
│  Part B: Self-Correcting Test Generation                             │
│  Qwen2.5-Coder-7B + LoRA  │  AST → LLM → test → error → retry       │
│  RAG: call-graph (BM25 + semantic) + vector store + SQLite memory   │
│                                                                      │
│  Bug Exposure Score (BES): 5-dimension quality gate                 │
│    1. Spec coverage   (30%): docstring examples as assertions        │
│    2. Input variety   (25%): case / sign / type characteristic mix   │
│    3. Boundary cov.   (20%): AST-verified boundary args              │
│    4. Assertion qual. (15%): exact ==, raises, distinct targets      │
│    5. Mutation pot.   (10%): heuristic mutation-killing signal       │
│                                                                      │
│  Docstring extractor (deterministic spec tests)                     │
│  Boundary synthesizer (type-aware edge cases)                       │
│  Pre-pass triage: real bug vs stale test disambiguation             │
└──────────────────────────┬──────────────────────────────────────────┘
                           │ confirmed/suspected bugs → bug_reports.jsonl
┌──────────────────────────▼──────────────────────────────────────────┐
│  Part C: Automated Program Repair (APR)                              │
│  SBFL fault localisation (Ochiai) → Qwen2.5-Coder-7B + LoRA         │
│  AST-safe patching → workspace isolation → git branch               │
└─────────────────────────────────────────────────────────────────────┘
```

### Part B generation loop

For each target file:

1. **Parse**: AST analysis (control-flow paths, raises, callees, docstrings).
2. **Filter**: skip empty files, `__init__`, and existing test files.
3. **Prime**: scan the repo for existing passing tests and use them as style examples.
4. **Retrieve**: call-graph (BM25 + semantic) and vector context, an optional external-docs FAISS layer, and SQLite memory of good tests and failure patterns.
5. **Generate**: prompt the local model for pytest source.
6. **Validate**: syntax, import, and symbol checks.
7. **Execute**: run pytest with coverage and capture the traceback.
8. **Self-correct**: feed the error back and regenerate (up to about 15 iterations).
9. **Score**: BES quality gate.
10. **Store**: save good tests and bad patterns to memory.
11. **Mutate** (optional): mutation testing for bug detection.

Two execution modes: **quality mode** (default, per-target generation with the BES gate, optimizes correctness) and **suite mode** (`--suite`, one comprehensive suite per program plus one coverage-feedback round, optimizes coverage, the TestEval protocol).

### Unified desktop app

```
unified_app/
├── src/                     # React + TypeScript (Vite + Tailwind)
│   ├── App.tsx              # Root: 4 modes (Combined | PartA | PartB | PartC)
│   ├── components/          # Landing, LeftSidebar, MainContent, RightSidebar…
│   └── types.ts             # Shared TypeScript types
├── server.py                # FastAPI :8080, all routes, SSE streaming
├── orchestrator.py          # Combined pipeline A→B→C with quality modes
├── model_lifecycle.py       # GPU context managers + multi-adapter swap (partb/partc)
├── requirement_matcher.py   # Hybrid keyword + semantic matcher
└── coverage_analyzer.py     # SRS coverage gap analysis
```

## Features

### Part A: Requirement extraction
- **SRS pipeline**: PDF/DOCX → spaCy sentence segmentation → PURE dataset fuzzy alignment → heuristic fallback labeler.
- **README extractor**: cleans the README → LLM feature extraction → LLM test-scenario generation (run in a subprocess for VRAM safety).

### Part B: Test generation
- **RAG**: call-graph retrieval (BM25 + semantic), vector retrieval, optional external-docs layer (FAISS), and SQLite memory.
- **Self-correcting loop**: generate → run → capture error → regenerate.
- **Bug Exposure Score (BES)**: detects tests that look good but miss bugs.
- **Docstring extractor**: guarantees spec examples are always present in the test file.
- **Boundary synthesizer**: type-aware edge cases.
- **Pre-pass triage**: checks whether existing tests already fail, and uses the LLM to confirm "two independent failures = real bug" before routing to Part C.
- **Auto-priming**: injects passing tests from the repo as style examples.
- **Mutation testing**: `mutmut` integration with bug-report generation.
- **Chat refinement**: the model stays warm after a run so you can ask for follow-up edits to a generated test file. It is edits-only by design: any message that does not produce a valid test block is answered with a fixed string instead of free-form model text.

### Part C: Automated program repair
- **SBFL fault localisation**: Ochiai score per line from a coverage spectrum.
- **Workspace isolation**: source is never modified in place; repairs happen in `PartC/workspaces/<id>/`.
- **AST-safe patching**: replaces only matching functions, with a full-file fallback for syntax errors.
- **Multi-adapter GPU sharing**: Part B and Part C adapters are loaded together via PEFT named adapters for a cheap hot-swap (about 10 ms) instead of a full model reload.
- **Git branch creation**: successful patches are committed to a `testmate-repair-<timestamp>` branch.
- **Verdict update**: a Part C success or failure promotes or demotes Part B's bug-confidence score.

### Desktop app
- **4 modes**: Full Pipeline (A→B→C), Part A only, Part B only, Bug Fixer (Part C standalone).
- **SSE streaming**: long jobs return a `job_id` immediately and the frontend streams `/stream?job_id=…`.
- **Stale Tests tab**: shows pre-pass results (existing tests found, real bugs confirmed, stale tests auto-updated).
- **Auto-Repair toggle**: opt-in; confirmed bugs automatically trigger Part C repair.
- **Quality modes**: Fast (A+B), Balanced (+ gap analysis), Best (+ gap fill).

## Model and training

| | |
|---|---|
| Base model | `Qwen/Qwen2.5-Coder-7B-Instruct`, 4-bit NF4 (bitsandbytes) |
| Adapter | LoRA, r=16, alpha=32, targets `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| Part B training data | MBPP only, decontaminated against HumanEval and TestEval; 970 samples, 2 epochs, QLoRA |
| Final training loss | 0.259 |
| Hardware for training | Kaggle T4 (QLoRA on a 7B needs more than 8 GB of VRAM) |

**Contamination control:** the adapter is trained on MBPP only, with every example that overlaps the evaluation sets removed (`decontaminate_mbpp.py`). The baseline for every comparison is the same base model without the adapter.

An earlier adapter trained on SWE-bench + Magicoder + MBPP, on the non-Instruct base, is kept only as an automatic fallback. The clean adapter (`models/graphrag_lora_clean/final`) is preferred everywhere.

## Project structure

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
│   ├── ast_parser_complete.py  # AST → CFG paths, raises, callees
│   ├── layers/                 # RAG layers (docs FAISS, call graph + vector)
│   ├── models/
│   │   ├── graphrag_lora_clean/final/  # preferred adapter (MBPP-only, decontaminated)
│   │   └── graphrag_lora/final/        # old adapter (fallback)
│   ├── value_reasoning_model/  # second LoRA (value reasoning)
│   ├── eval_lite/              # testgenevallite files + cached TestEval data
│   └── testgen/
│       ├── main.py             # autonomous_loop(): core generation + BES gate
│       ├── testgen_api.py      # execute_part_b() and refine_test_file()
│       ├── api_server.py       # standalone FastAPI :8000
│       ├── bes_scorer.py       # 5-dimension Bug Exposure Score
│       ├── docstring_extractor.py
│       ├── boundary_synthesizer.py
│       ├── bug_detector.py     # mutation helpers + triage confidence
│       ├── bug_to_partc.py     # Part B → Part C bridge + adapter swap
│       ├── existing_test_scanner.py / existing_test_runner.py
│       ├── rag_store.py        # SQLite RAG memory
│       ├── quality_gates.py / symbol_validator.py
│       └── training/           # training + evaluation harness
│           ├── decontaminate_mbpp.py
│           ├── kaggle_train_clean.py   # QLoRA training
│           ├── _ablation_common.py     # evaluation engine
│           ├── run_comparisons.py      # variant sweeps + comparison tables
│           ├── make_paper_tables.py    # results JSON → tables
│           └── results/                # every run output (JSON)
│
├── PartC/
│   ├── api/partc_api.py        # execute_part_c()
│   ├── core/
│   │   ├── control_loop.py     # repair_loop(): SBFL → patch → verify
│   │   ├── run_and_collect.py  # per-test coverage (Ochiai spectrum)
│   │   ├── sbfl_localiser.py   # Ochiai scoring
│   │   ├── model_runner.py     # Qwen inference
│   │   └── inference.py        # 4-bit BitsAndBytes loader
│   ├── models/adapter/         # Part C LoRA adapter weights
│   └── training/               # QLoRA fine-tuning scripts
│
├── unified_app/                # desktop app (see above)
├── docs/
│   └── repair_rate_comparison.png
├── EVAL_RUNBOOK.md
└── README.md
```

## Running

Requirements: Python 3.11+, Node.js (for the desktop app), and a GPU with enough VRAM for a 4-bit 7B model (an 8 GB GPU was used during development). Developed and tested on Windows.

### Unified desktop app (recommended)

```bash
cd unified_app
npm install
npm run dev           # Electron + Vite hot-reload
python server.py      # or run the backend standalone on :8080
```

### Part B standalone (test generation)

```bash
cd PartB
pip install -r testgen/requirements_docker.txt
python testgen/api_server.py                     # FastAPI on :8000
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

### Smoke test (no GPU needed)

A self-contained scenario with 5 intentional bugs lives in `PartB/testgen/pipeline_test/`. The model is mocked.

```bash
cd PartB/testgen
python pipeline_test/run_pipeline_test.py
```

Expected output: `6/6 tests passed`, covering the bug oracle, BES scoring, bug grouping, workspace isolation, and verdict transitions.

### GPU memory constraint

Part A's model and Part B/C's Qwen cannot share GPU memory. `model_lifecycle.py` loads a model, yields, then deletes and flushes it. The README extractor runs in a **subprocess** because 4-bit bitsandbytes models on Windows do not release VRAM cleanly in the same process. Part B and Part C share one base model and swap named PEFT adapters (`partb`, `partc`).

## Reproducing the results

All commands run from `PartB/testgen/training/`.

```bash
# Headline: TestEval coverage + correctness + bug detection
python run_comparisons.py --dataset testeval --variants testmate no_lora --suite --mutation --sample 0

# Quality-mode exhibit (the coverage-vs-correctness gap)
python run_comparisons.py --dataset testeval --variants testmate --mutation --sample 0

# HumanEval sanity check
python run_comparisons.py --dataset humaneval --variants testmate no_lora --sample 0

# RAG lift on real code (local only: needs version-matched venvs)
python setup_version_venvs.py
python run_comparisons.py --dataset testgenevallite --variants testmate no_rag --sample 0 --mutation

# Stitch result tables
python make_paper_tables.py
```

**Retraining the adapter** (Kaggle, GPU T4, internet on): run `decontaminate_mbpp.py`, then `kaggle_train_clean.py`, then place the adapter at `PartB/models/graphrag_lora_clean/final/`.

**Part C repair results:** `compare_evals.py` merges `eval_20260411_221134.json` (fine-tuned modes) and `eval_20260413_140126.json` (baseline) into `evaluation_comparison.csv` and `docs/repair_rate_comparison.png`.

The comparison variants in `run_comparisons.py` isolate each component:

| Variant | What it isolates |
|---|---|
| `testmate` | Full stack: RAG + self-correction + LoRA |
| `no_rag` | LoRA only; total RAG value = `testmate` − `no_rag` |
| `graph_only` | Only the call-graph layer |
| `no_graph` | Docs + vector + memory, no graph |
| `no_lora` | Full RAG and loop on the base model; LoRA/system value |

## Limitations

- Single-run numbers at temperature 0.3; no multi-seed significance testing.
- The TestEval comparison uses a faithful re-implementation of the protocol, not the official harness.
- The testgenevallite RAG result is directional (N=30).
- The LoRA-vs-base contrasts are system-level comparisons (full TestMate vs base, vs no-RAG), not isolated single-component ablations.
- HumanEvalFix is function-level repair. It does not measure repair of multi-file bugs in large codebases.
- The external-docs RAG layer can degrade (a non-fatal query error is logged); the call-graph, vector, and memory layers keep working.
- TestMate is not claiming coverage state of the art or a comparison with GPT-4-class models. A local 4-bit 7B model sits near the top of the open 7B field on a metric that is close to saturated.

## References

- TestEval: Wang et al., NAACL Findings 2025.
- HumanEval: Chen et al., 2021. MBPP: Austin et al., 2021. HumanEvalFix (HumanEvalPack): Muennighoff et al., 2023.
- Qwen2.5-Coder: Hui et al., 2024. LoRA: Hu et al., 2021. QLoRA: Dettmers et al., 2023.
- KGCompass (arXiv 2025): multi-hop graph traversal for code navigation.
- RAGFix (NSF/IEEE 2025): external knowledge retrieval for bug fixing.
- RAG Traceability (MCSE 2025): requirement-to-code linking.
- Automated program repair: Zhang et al., 2023; Meng et al., 2024. Retrieval-augmented generation: Gao et al., 2023; Zhao et al., 2024.

## Author

Mohammed Medhat Abdelaleem. [LinkedIn](https://www.linkedin.com/in/mohammedmedhat-1646b3281) · [GitHub](https://github.com/Mohammed-Medhat)
