# TestMate PartB — Run Book (turnkey)

Everything below is the **complete** sequence to produce the paper-ready
results. You only run files — no editing required. The base model is already on
disk (`D:\donwloader\Qwen2.5-Coder-7B-Instruct`), and the eval harness loads it
fully offline (zero download).

**The claim is NOT coverage SOTA** (a 4-bit 7B won't beat GPT-4, and coverage is
saturated — everyone scores 86–98%). The real contribution is a **three-axis
evaluation** that exposes what single-number coverage hides:

| Axis | Metric | Why it matters |
|------|--------|----------------|
| Coverage | line/branch % | did the test inputs *execute* the code (what the leaderboard measures) |
| Correctness | pass@1 % | are the *assertions correct* (TestEval ignores this) |
| **Bug-detection** | **mutation %** | would the test *catch a real bug* (nobody on the leaderboard reports this) |

Key finding already in the data: suite mode gets **91.6% coverage but ~18%
correct assertions** — proof that coverage is misleading. Mutation score is the
honest quality metric.

Tracks (all contamination-free — trained on MBPP only, decontaminated):

| Track | Dataset | Role |
|-------|---------|------|
| **TestEval** (210 LeetCode programs) | recognized benchmark | coverage + correctness + **mutation** comparison |
| **testgenevallite** (101 real framework files) | realistic code | RAG-lift (Δcoverage = testmate − no_rag) |
| **HumanEval** (164 problems) | standard | universal sanity point |

---

## Step 0 — one-time (already done this session)

- `pip install faiss-cpu` ✅ (Layer-1 docs RAG)
- `pip install sortedcontainers` ✅ (TestEval solutions import it)
- `python setup_version_venvs.py --only django42` ✅ (testgenevallite version routing)
  Optional: build the rest for full testgenevallite coverage:
  `python setup_version_venvs.py`

## Step 1 — Clean retrain on **Kaggle** (one job, ~1–2 h, T4)

The only step that needs Kaggle (QLoRA on a 7B needs >8 GB). Internet ON.

1. Upload/open `kaggle_train_clean.py` as a Kaggle notebook.
2. It trains QLoRA (4-bit NF4) on **MBPP only**, decontaminated vs HumanEval,
   on the **Instruct** base (fixes the old base mismatch).
3. Download the resulting adapter and place it at:
   `PartB/models/graphrag_lora_clean/final/`

All local entrypoints auto-prefer `graphrag_lora_clean/final` and fall back to
the old adapter, so everything downstream picks it up automatically.

> Want numbers *before* the retrain? The local commands below already work with
> the existing adapter (fallback) — the only caveat is HumanEval/TestEval would
> be contaminated for MBPP-overlap reasons until the clean adapter is in place.

## Step 2 — Results, all **local** (the "just run files" part)

```bash
cd PartB/testgen/training

# A) QUALITY mode + MUTATION — the bug-detection claim (the real differentiator).
#    Runs testmate (BES + self-correction) and no_lora (raw base), then mutation
#    on the files with passing tests. Answers: do TestMate's tests catch bugs?
python run_comparisons.py --dataset testeval --variants testmate no_lora --mutation --sample 0

# B) SUITE/COVERAGE mode — the fair coverage comparison vs the leaderboard.
#    (already run; --mutation here shows suite tests' bug-detection too)
python run_comparisons.py --dataset testeval --variants testmate no_lora --suite --mutation --sample 0

# C) the other two tracks
python run_comparisons.py --dataset humaneval       --sample 0
python run_comparisons.py --dataset testgenevallite --sample 0

# D) stitch everything into paper tables
python make_paper_tables.py
```

Key artifacts in `paper_tables/`:
- `quality_vs_coverage.md` — **the core table**: coverage vs correctness vs
  bug-detection (mutation), with a data-driven headline verdict.
- `testeval_vs_paper.md` — your coverage next to GPT-4 / 7B models.

The honest story: you're *coverage-competitive* with the 7B pack, and you're the
only one who measures whether the tests are **correct** and **catch bugs** —
where high coverage with wrong assertions is exposed.

Outputs land in `PartB/testgen/training/paper_tables/`:

- `testeval.{md,csv}` — headline coverage table (testmate vs no_rag vs no_lora …)
- `rag_lift.{md,csv}` — testgenevallite with the **Δcoverage RAG-lift**
- `humaneval.{md,csv}` — sanity table
- `SUMMARY.md` — the stitched story + contamination statement

### Resume / skip-if-complete

`run_comparisons.py` **skips any variant that already finished** (its `{name}.json`
has a `summary` block and enough files). So if a sweep is interrupted — or you
re-run the same command — completed variants are reused and it picks up at the
next unfinished one. No wasted GPU time.

- Re-running the same command continues where it stopped.
- A finished variant at `--sample 10` counts as done for `--sample 10` but not
  for `--sample 0` (all) — bump the sample and it re-runs only what's needed.
- Force a fresh run with `--force`.

### Faster first look

Each variant reloads the model, so a full sweep is a long GPU job. For a quick
check, sample a few and run one or two variants:

```bash
python run_comparisons.py --dataset testeval --sample 10 --variants testmate no_rag
python make_paper_tables.py
```

---

## What each variant means

| variant | what it isolates |
|---------|------------------|
| `testmate` | full stack: all RAG layers + self-correction loop + LoRA |
| `no_rag` | LoRA only, no RAG, no loop → total RAG value = testmate − no_rag |
| `graph_only` | only the call-graph layer |
| `no_graph` | docs + vector + memory, no graph |
| `no_lora` | full RAG + loop on the **base** model → LoRA value |

## Notes / known scope

- **Mutation testing** (Phase 2, in the `*_with_mut.json`) is best-effort. It
  runs on flat/self-contained cases (HumanEval, TestEval) and skips safely
  (null score, no crash) on identity/cluster files (testgenevallite).
- **TestEval fetch**: the dataset (`leetcode-py.jsonl`, ~tens of KB) is pulled
  from the authors' GitHub on first use and cached at
  `PartB/eval_lite/testeval_leetcode-py.jsonl`.
- Everything else (model, HumanEval, testgenevallite) is already local.
