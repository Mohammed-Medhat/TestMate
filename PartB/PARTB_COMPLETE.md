# TestMate — Part B: Complete Engineering & Reproduction Reference

> **Purpose of this document.** A single, self-contained reference that lets a
> developer (or an AI agent) understand *everything* about Part B and rebuild it
> from scratch — architecture, every module, the model, the training pipeline,
> the evaluation harness, the datasets, the design decisions and the bugs that
> shaped it, and exact reproduction commands. If you only want the numbers and
> the conclusion, read `PARTB_RESULTS.md` instead. For paper-claim wording read
> `PAPER_README.md`.

---

## 1. What Part B is

**TestMate Part B** is a **self-correcting unit-test generator**. Given a Python
source file, it produces a `pytest` test suite using a local LLM
(**Qwen2.5-Coder-7B-Instruct**, 4-bit, + a LoRA adapter), grounded by a
multi-layer **RAG** system and refined by an **execute → read error → retry**
self-correction loop. It runs **fully local — no external API**.

**Design goal:** maximize *useful* tests, where "useful" is measured on three
axes the literature usually collapses into one:

| Axis | Metric | Question it answers |
|------|--------|---------------------|
| Coverage | line / branch % | Did the test inputs *execute* the code? |
| Correctness | pass@1 % | Are the *assertions correct* (tests actually pass)? |
| Bug-detection | mutation kill % | Would the test *catch a real injected bug*? |

The central empirical finding: **coverage alone is misleading** — a suite can hit
~94 % coverage while a large fraction of its assertions are wrong. Part B
measures all three axes and reports the trade-offs honestly.

---

## 2. End-to-end pipeline (the autonomous loop)

`testgen/main.py :: autonomous_loop()` is the heart. Per target file:

```
1. PARSE        AST analysis → CFG paths, raises, callees, docstrings, targets
                (ast_parser_complete.py, docstring_extractor.py)
2. FILTER       skip empty / __init__ / existing-test files (testable_filter.py)
3. PRIME        scan repo for existing passing tests as style examples
                (existing_test_scanner.py, existing_test_runner.py)
4. RETRIEVE     3-layer RAG context:
                  L1 docs   — FAISS over external docs (layers / layer1)
                  L2 code   — call-graph (BM25 + semantic) + vector store (layer2)
                  memory    — SQLite store of good tests + failure patterns (rag_store.py)
5. GENERATE     prompt Qwen2.5-Coder-7B + LoRA → pytest source
                (optional plan-then-generate when plan_mode=True)
6. VALIDATE     syntax + import + symbol checks (symbol_validator.py, quality_gates.py)
7. EXECUTE      run pytest (+ coverage); capture pass/fail + traceback
8. SELF-CORRECT feed the error back to the LLM, regenerate (max ~15 iters)
9. SCORE        BES (Bug Exposure Score) quality gate (bes_scorer.py)
10. STORE       good tests + bad patterns → SQLite RAG memory (rag_store.py)
11. (opt) MUTATE mutation testing for bug-detection (bug_detector.py / harness)
```

Two execution modes:
- **Quality mode** (default): per-target generation, BES gate, self-correction —
  optimizes *correctness*.
- **Suite mode** (`--suite`): one comprehensive multi-case suite per program, no
  BES gate, one coverage-feedback round — optimizes *coverage* (the TestEval
  protocol). Implemented in the harness as `generate_suite_mode()`.

---

## 3. Directory map (what each thing is)

```
PartB/
├── ast_parser_complete.py        # AST → CFG paths, raises, callees, complexity
├── layers/                       # 3-layer RAG implementations (docs FAISS, code graph+vector)
├── value_reasoning/              # value-reasoning model wrappers
├── value_reasoning_model/        # second LoRA adapter (value reasoning)
├── models/
│   ├── graphrag_lora/            # OLD adapter (fallback)
│   └── graphrag_lora_clean/      # CLEAN adapter (preferred) — MBPP-only, decontaminated
│       └── final/                # ← inference loads this
├── eval_lite/                    # testgenevallite eval files + cached TestEval jsonl
├── desktop/ outputs/ repos/ tests/
└── testgen/
    ├── main.py                   # autonomous_loop() — the core generator
    ├── testgen_api.py            # execute_part_b() — clean API used by unified server
    ├── api_server.py             # standalone FastAPI (port 8000) + HITL review GUI
    ├── bes_scorer.py             # Bug Exposure Score quality gate
    ├── docstring_extractor.py    # docstring spec examples → tests
    ├── existing_test_scanner.py  # auto-prime: find passing tests as style examples
    ├── existing_test_runner.py   # run discovered tests
    ├── testable_filter.py        # skip empty/__init__/test files before the LLM
    ├── rag_store.py              # SQLite RAG memory (good tests, failure patterns)
    ├── quality_gates.py          # syntax/import/quality validation
    ├── symbol_validator.py       # symbol-resolution / hallucination checks
    ├── docker_runner.py          # optional Docker isolation (version-matched exec)
    ├── bug_detector.py           # mutation / bug-detection helpers
    ├── intense_mode.py           # high-effort generation mode
    ├── requirements_docker.txt   # runtime deps
    ├── Dockerfile                # container for version-matched execution
    ├── generated_tests/          # output: test_<name>.py
    └── training/                 # === training + evaluation (see §6, §7) ===
        ├── _ablation_common.py   # THE eval harness (configs, run_ablation, mutation, suite, venv routing)
        ├── run_comparisons.py    # variant sweep + comparison tables (--suite --mutation --force --fresh)
        ├── make_paper_tables.py  # stitch results JSON → paper_tables/*.md,*.csv
        ├── setup_version_venvs.py# build per-framework-version venvs (testgenevallite)
        ├── kaggle_train_clean.py # CLEAN QLoRA training (MBPP only, decontaminated)
        ├── decontaminate_mbpp.py # remove MBPP↔HumanEval/TestEval overlap
        ├── local_ablation_{testeval,humaneval,testmate}.py  # local entrypoints
        ├── results/              # all run outputs (compare_<dataset>{,_suite}/*.json)
        └── paper_tables/         # generated tables (humaneval.md, rag_lift.md, …)
```

---

## 4. The model

- **Base:** `Qwen/Qwen2.5-Coder-7B-Instruct`, loaded **4-bit NF4** (BitsAndBytes).
- **Adapter:** LoRA, **r=16, alpha=32**, target `q/k/v/o_proj`, applied on top of
  the base at inference.
- **Two adapters exist:**
  - `models/graphrag_lora_clean/final` — **preferred**. Clean retrain: trained on
    **MBPP only**, **decontaminated** against HumanEval/TestEval, on the
    **Instruct** base (fixes an earlier base-mismatch bug). 970 samples, 2 epochs.
  - `models/graphrag_lora/final` — old adapter, kept as automatic fallback.
- All entrypoints auto-prefer the clean adapter and fall back to the old one, so
  downstream code "just works" once the clean adapter is in place.
- **Local snapshot for zero-download inference:**
  `D:\donwloader\Qwen2.5-Coder-7B-Instruct` (harness sets `HF_HUB_OFFLINE=1`).

---

## 5. VRAM constraint (Part A ↔ Part B)

Part A's LLM and Part B's Qwen **cannot coexist in GPU memory**. The unified app's
`model_lifecycle.py` provides context managers that load → yield → delete+flush.
The README extractor runs in a **subprocess** because 4-bit BitsAndBytes models on
Windows don't release VRAM cleanly with `del + empty_cache` in-process. (Detail
lives in `unified_app/`, but it constrains how Part B is invoked in the combined
pipeline.)

---

## 6. Training pipeline (how to reproduce the adapter)

**Where:** Kaggle (QLoRA on a 7B needs >8 GB; T4 works). Internet ON.

1. `decontaminate_mbpp.py` — builds an MBPP training set with all examples that
   overlap HumanEval/TestEval removed (prevents train/test contamination).
2. `kaggle_train_clean.py` — QLoRA (4-bit NF4) on the decontaminated MBPP, on the
   **Instruct** base. Outputs an adapter.
3. Download the adapter → place at `PartB/models/graphrag_lora_clean/final/`.

That's the only step that requires Kaggle. Everything downstream is local.

---

## 7. Evaluation harness (the core of the results)

### 7.1 `_ablation_common.py` — the engine
Owns model loading and the whole eval. Key pieces:

- **`AblationConfig`** — toggles for every component: `enable_layer1_docs`,
  `enable_layer2_graph`, `enable_layer2_vector`, `enable_rag_memory`,
  `enable_self_correction_loop`, `enable_lora`, `enable_plan_mode`, plus
  `dataset`, `suite_mode`, `model_id`, `lora_path`, `sample_size`,
  `max_file_seconds`, `max_retries`, `resume`.
- **`run_ablation(cfg, output_path)`** — runs generation+evaluation over a
  dataset, writing per-file metrics + a `summary`. Checkpoints to
  `<name>.partial.json` after each file (resume-safe).
- **Dataset routing:** `load_testeval_dataset()` (fetches & caches
  `leetcode-py.jsonl` from the TestEval GitHub), `load_humaneval_dataset()`
  (HF `openai_humaneval`), else the testgenevallite loader (local `eval_lite/`).
- **Suite mode:** `generate_suite_mode()`, `_suite_prompt()`,
  `_suite_coverage_probe()` — comprehensive suite + one coverage round, no BES.
- **Mutation:** `run_mutation_pass(in_json, out_json)` and `_run_mutmut_on_pair()`
  — run mutation testing on files that have **passing** tests, producing
  `mean_mutation_score`. Source is supplied from the dataset map (not disk).
- **Version venvs (testgenevallite):** `_CLUSTER_MAP`, `_cluster_name_for()`,
  `_select_cluster_python()`, `_version_matches()` route each real-framework file
  to a venv with the matching dependency version (e.g. `django42`). Falls back to
  host Python if the venv is missing (which degrades testgenevallite numbers).
- **Identity restoration (testgenevallite):** synthetic targets are written back
  to their real module path (e.g. `django/forms/formsets.py`) via a `sys.modules`
  conftest so imports resolve like the real package.

### 7.2 `run_comparisons.py` — variant sweep
Runs a set of variants and prints/writes a comparison table.

| variant | isolates |
|---|---|
| `testmate` | full stack: all RAG layers + self-correction + LoRA |
| `no_rag` | LoRA only, no RAG, no loop → total RAG value = testmate − no_rag |
| `graph_only` | only the call-graph layer |
| `no_graph` | docs + vector + memory, no graph |
| `no_lora` | full RAG + loop on the **base** model → LoRA/system value |

Flags: `--dataset {testeval,humaneval,testgenevallite}`, `--sample N` (0=all),
`--suite`, `--mutation`, `--force` (re-run completed), `--fresh` (wipe checkpoints
— use after editing generation code), `--variants ...`.
Skip-if-complete: a finished variant (has `summary`, enough files) is reused.

### 7.3 `make_paper_tables.py` — stitching
Reads `results/compare_*/` and emits `paper_tables/`:
`humaneval.md`, `rag_lift.md` (testgenevallite), `quality_vs_coverage.md`
(three-axis), `testeval_vs_paper.md` (vs the paper's 7B models — competitor
numbers in `_TESTEVAL_PAPER`), `SUMMARY.md`. Prefers `<variant>_with_mut.json`
so mutation scores are folded in.

---

## 8. Datasets

| Dataset | What | Source | Role |
|---|---|---|---|
| **MBPP** (decontaminated) | basic Python tasks | HF, filtered | **training only** |
| **TestEval** (210 LeetCode) | coverage-focused test-gen benchmark | Wang et al. 2024, GitHub `LLM4SoftwareTesting/TestEval`, cached `eval_lite/testeval_leetcode-py.jsonl` | headline coverage + correctness + mutation |
| **HumanEval** (164) | standard | HF `openai_humaneval` | internal sanity (system vs base) |
| **testgenevallite** (~80–101 real framework files) | Django/SymPy/etc. internals | local `eval_lite/` | RAG-lift on real code (Δ = testmate − no_rag) |

**Contamination control:** trained on MBPP only, decontaminated vs the eval sets;
the `no_lora`/base comparison uses the *same* model without the adapter.

---

## 9. Key design decisions & the bugs that shaped them (history)

These are the non-obvious lessons — preserve them, they cost real debugging time.

1. **Clean retrain for contamination + base-mismatch.** The old adapter trained
   on SWE-bench+Magicoder+MBPP and on the *base* (non-Instruct) model, while
   inference used *Instruct*. Fixed by the MBPP-only, decontaminated, Instruct-base
   retrain (`graphrag_lora_clean`).
2. **6-tuple unpack regression.** An early return in
   `_run_production_autonomous_loop` returned 5 values while the caller unpacked
   6 → 0 % scores on 3 Kaggle runs. Fixed to `return "", 0, 0, 0, [], {}`.
3. **evaluate flat-layout bug.** Used flat `{mid}.py` instead of the
   identity-restoration layout → 0 % scoring. Fixed to use `_make_identity_conftest`
   when `original_code_file` is present.
4. **`test_output_dir` bug.** `autonomous_loop` wrote the test to
   `generated_tests/` when `import_path` was set, but the ablation looked next to
   the tempdir source → empty `test_code`. Added a `test_output_dir` param.
5. **Windows Unicode crash.** `open(test_file, "w")` used cp1252 and crashed on
   model-generated `→`. All 9 sites now use `encoding="utf-8"`.
6. **Version-prefix collision.** `"1.12".startswith("1.1")` was True → wrong venv
   cluster. Fixed with `_version_matches` (dot-boundary check).
7. **Mutation 0 % bug (most impactful).** `_run_mutmut_on_pair` looked for source
   as a disk file that doesn't exist for synthetic TestEval → 0.0 for all files.
   Fixed by building an id→source map in `run_mutation_pass` and passing
   `source_code` through. Revalidated to real ~74–82 %.
8. **Suite mode added** to fix the coverage gap (54 % → ~94 % on TestEval) — and
   in doing so exposed the coverage-vs-correctness gap that became the paper's hook.
9. **O5 validator** setting `_last_category="import_error"` cut retries early —
   made diagnostic-only.
10. **skip-if-complete** logic: `--sample 0` (None) treated any summarized file as
    "complete" regardless of count; this can wrongly skip a variant. Use `--force`
    /`--fresh`, or run a single variant, to control it (this is why the matched-N
    testgenevallite comparison needs care — see `PARTB_RESULTS.md`).

---

## 10. Full reproduction (turnkey)

```powershell
cd D:\TestMate\TestMate\PartB\testgen\training
set PYTHONIOENCODING=utf-8

# headline: TestEval coverage + correctness + bug-detection
python run_comparisons.py --dataset testeval --variants testmate no_lora --suite --mutation --sample 0

# quality-mode exhibit (coverage-vs-correctness gap)
python run_comparisons.py --dataset testeval --variants testmate --mutation --sample 0

# sanity: HumanEval (or on Kaggle — see below)
python run_comparisons.py --dataset humaneval --variants testmate no_lora --sample 0

# RAG-lift on real code (LOCAL only — version venvs)
python setup_version_venvs.py                 # one-time, builds django42, sympy, …
python run_comparisons.py --dataset testgenevallite --variants testmate no_rag --sample 0 --mutation

# stitch tables
python make_paper_tables.py
```

**Kaggle (self-contained tracks only — TestEval, HumanEval):** clone the repo,
`pip install transformers peft bitsandbytes accelerate datasets sentence-transformers faiss-cpu coverage pytest pytest-cov sortedcontainers`,
upload `models/graphrag_lora_clean/final` as a dataset, set
`model_id=Qwen/Qwen2.5-Coder-7B-Instruct` + `lora_path` to the uploaded path,
Internet ON, GPU T4. **testgenevallite must stay local** (or rebuild the venvs on
Kaggle via `setup_version_venvs.py` with `TESTMATE_VENV_ROOT` set) because its
tests need version-matched dependencies.

---

## 11. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `Layer 1 docs query failed: 'list' object has no attribute 'results'` | Non-fatal: L1 docs layer degraded; other RAG layers still work. Describe RAG as graph+vector+memory, or fix the L1 integration. |
| All testgenevallite tests fail to import | Version venvs missing → host-Python version wall. Run `setup_version_venvs.py`; verify with a 3-file smoke run (non-zero pass@1). |
| Mutation score 0 % everywhere | The source-lookup bug (§9.7) — ensure `run_mutation_pass` builds the id→source map. |
| 0 % scores on an otherwise-working run | The 6-tuple / flat-layout / test_output_dir bugs (§9.2–9.4). |
| Re-run reuses old numbers after a code edit | skip-if-complete served the old result. Use `--fresh`. |
| Windows `UnicodeEncodeError` writing tests | Ensure `encoding="utf-8"` on all `open(...,'w')` and `PYTHONIOENCODING=utf-8`. |
| VRAM not freed between Part A and Part B | Run the VRAM-heavy step in a subprocess (see §5). |

---

## 12. Cross-references
- `PARTB_RESULTS.md` — all final numbers + conclusion.
- `PAPER_README.md` — exact paper-claim wording (what to say / not say).
- `make_paper_tables.py::_TESTEVAL_PAPER` — competitor 7B numbers.
- `results/compare_*/` — every metric traces to a JSON here.
