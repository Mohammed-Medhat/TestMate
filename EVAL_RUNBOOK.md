# TestMate — Evaluation Runbook

The exact recipe to reproduce every number that goes in the paper.  
Each phase tells you what to upload, what to edit, and what to run.

---

## How the evaluation is organised

You will produce **three tables** for the paper:

| Table | What it shows | Where it runs |
|-------|--------------|---------------|
| **Table 1 — Part B (Test Generation)** | Baseline vs each RAG component vs full TestMate | Kaggle |
| **Table 2 — Part C (Bug Repair)** | Base Qwen APR vs TestMate APR | Kaggle |
| **Table 3 — Full Pipeline** | Spec → tests → repair, end-to-end | Kaggle (gen) + local (true repair rate) |

---

## What to upload to Kaggle (one-time)

Zip each folder (or single file), then upload each as a separate Kaggle dataset:

| What to upload | Kaggle dataset name | Used by |
|---------------|--------------------|---------|
| `PartB/models/graphrag_lora/final/` | `testgen-model` | Table 1 TestMate + Table 3 Part B |
| `PartC/models/adapter/` | `testmate-partc-adapter` | Table 2 + Table 3 Part C |
| `PartC/humaneval_eval_dataset.json` | `testmate-humaneval` | Table 2 + Table 3 |
| **`unified_app/testmate_rag.db`** | **`testmate-rag-db`** | **Conditions with RAG memory enabled (Full RAG, +Loop, +Plan, TestMate)** |

> No knowledge graph upload is needed. The production system builds the graph
> on the fly per file, and all the evaluation scripts do the same.

> The `testmate-rag-db` upload is a single SQLite file (~220 KB) containing
> ~39 past good tests, ~116 bad patterns, etc. — gathered from real TestMate
> runs. The ablation harness auto-detects it on Kaggle and copies it into
> place before evaluation. If you don't upload it, the RAG memory layer
> silently returns empty (functionally OFF).

---

# Phase 1 — Table 1: Part B (Test Generation)

You will run **7 Kaggle notebooks**. Each one uses the same 3-cell template;
only the last cell changes.

**Accelerator:** GPU T4 × 1

### Datasets to add for every Phase 1 notebook

| Condition | Datasets to attach |
|-----------|---------------------|
| Baseline / Graph Only / Vector Only | *(none — code is cloned at runtime)* |
| Full RAG / Full RAG + Loop / Full RAG + Plan | `testmate-rag-db` |
| TestMate | `testgen-model` + `testmate-rag-db` |

> Conditions that enable the RAG memory layer (Full RAG and above) benefit
> from `testmate-rag-db`. Without it, the memory layer silently returns
> empty — script still runs, but you lose one RAG layer's contribution.

### The 3-cell template

**Cell 1 — install dependencies:**
```python
!pip install -q transformers bitsandbytes accelerate peft \
             sentence-transformers faiss-cpu networkx \
             pytest pytest-cov coverage mutmut
```

**Cell 2 — clone the repo:**
```python
!git clone --depth 1 https://github.com/Mohammed-Medhat/TestMate.git \
    /kaggle/working/TestMate
%cd /kaggle/working/TestMate/PartB/testgen/training
```

**Cell 3 — run the chosen condition:**

| Condition | Cell 3 command |
|-----------|---------------|
| Baseline (no RAG, no fine-tuning) | `!python kaggle_ablation_no_rag.py` |
| Graph Only | `!python kaggle_ablation_graph_only.py` |
| Vector Only | `!python kaggle_ablation_no_graph.py` |
| Full RAG | `!python kaggle_ablation_full.py` |
| Full RAG + Loop | `!python kaggle_ablation_full_with_loop.py` |
| Full RAG + Plan | `!python kaggle_ablation_full_with_plan.py` |
| **TestMate** (production) | `!python kaggle_ablation_testmate.py` |

> **TestMate** is the full system — Full RAG + Loop + your LoRA adapter.
> The script auto-detects the LoRA path. It tries both your accounts in order:
> ```python
> _LORA_PATH_CANDIDATES = [
>     "/kaggle/input/datasets/mohammedmedhat08/testgen-model/final",
>     "/kaggle/input/datasets/mohammed8medhat/testgen-model/final",
> ]
> ```
> Whichever one is actually mounted on the current notebook is the one that gets used.
> So you can run the SAME script on either Kaggle account — no edits needed.
> The startup line `[testmate] LoRA path resolved -> ...` shows which path it picked.
> If both paths report MISSING, run `!find /kaggle/input -name "adapter_config.json"`
> to locate the real path and add it to `_LORA_PATH_CANDIDATES`.

### What each notebook produces

Each script writes to `/kaggle/working/results/`:
```
results/
├── ablation_<name>.json            ← Phase 1 output (no mutation testing yet)
└── ablation_<name>_with_mut.json   ← Phase 2 output (adds mutation kill rate)
```

### Where to download everything

Download the `/kaggle/working/results/` folder from each notebook and merge them
locally into:
```
paper_eval/
└── partb_results/
    └── results/
        ├── ablation_no_rag.json                  + _with_mut.json
        ├── ablation_graph_only.json              + _with_mut.json
        ├── ablation_no_graph.json                + _with_mut.json
        ├── ablation_full.json                    + _with_mut.json
        ├── ablation_full_with_loop.json          + _with_mut.json
        ├── ablation_full_with_plan.json          + _with_mut.json
        └── ablation_testmate.json                + _with_mut.json
```

---

# Phase 2 — Table 2: Part C (APR)

One Kaggle notebook handles both rows of Table 2 (base Qwen and TestMate-APR).

**Accelerator:** GPU T4 × 1

### Datasets to attach

| Kaggle dataset | Mount path |
|---------------|-----------|
| `testmate-humaneval` | `/kaggle/input/testmate-humaneval/` |
| `testmate-partc-adapter` | `/kaggle/input/testmate-partc-adapter/` |

### Cells

**Cell 1 — install:**
```python
!pip install -q transformers bitsandbytes accelerate peft pytest
```

**Cell 2 — clone:**
```python
!git clone --depth 1 https://github.com/Mohammed-Medhat/TestMate.git \
    /kaggle/working/TestMate
%cd /kaggle/working/TestMate/PartC
```

**Cell 3 — point the config at the uploaded adapter:**

First, locate the PartC adapter on the current account:
```python
!find /kaggle/input -name "adapter_config.json"
```

You'll see something like one of:
```
/kaggle/input/datasets/mohammedmedhat08/testmate-partc-adapter/.../adapter_config.json
/kaggle/input/datasets/mohammed8medhat/testmate-partc-adapter/.../adapter_config.json
```

Take the **parent directory** of that file (drop `/adapter_config.json` at the end)
and substitute it into the config:
```python
import subprocess
PARTC_ADAPTER = "/kaggle/input/datasets/<YOUR-ACCOUNT>/testmate-partc-adapter/adapter"  # <- edit
subprocess.run([
    "sed", "-i",
    f"s|ADAPTER_PATH.*|ADAPTER_PATH = '{PARTC_ADAPTER}'|",
    "config.py",
])
```

**Cell 4 — APR baseline (base Qwen, gold tests):**
```python
!python eval/apr_evaluator.py \
    --mode base_paper \
    --dataset /kaggle/input/testmate-humaneval/humaneval_eval_dataset.json \
    --output_dir /kaggle/working/apr_results
```

**Cell 5 — APR with TestMate (LoRA + iterative repair):**
```python
!python eval/apr_evaluator.py \
    --mode finetuned_full \
    --dataset /kaggle/input/testmate-humaneval/humaneval_eval_dataset.json \
    --output_dir /kaggle/working/apr_results
```

### Output produced

```
/kaggle/working/apr_results/
├── eval_<timestamp1>.json   ← from Cell 4  (APR Baseline)
└── eval_<timestamp2>.json   ← from Cell 5  (APR with TestMate)
```

**Download** the whole `apr_results/` folder. Save locally as:
```
paper_eval/
└── apr_results/
    ├── eval_<timestamp1>.json
    └── eval_<timestamp2>.json
```

---

# Phase 3 — Table 3: Full Pipeline (Spec → Tests → Repair)

You will run **4 Kaggle notebooks** — one per pipeline condition.

**Accelerator:** GPU T4 × 1

### Datasets to attach to every Phase 3 notebook

| Kaggle dataset | Needed for |
|---------------|-----------|
| `testmate-humaneval` | all four |
| `testmate-partc-adapter` | the three TestMate-APR conditions (i.e. all but the first) |

### Cells (same for all four)

**Cell 1 — install:**
```python
!pip install -q transformers bitsandbytes accelerate peft pytest
```

**Cell 2 — clone:**
```python
!git clone --depth 1 https://github.com/Mohammed-Medhat/TestMate.git \
    /kaggle/working/TestMate
%cd /kaggle/working/TestMate/PartB/testgen/training
```

**Cell 3 — run the chosen pipeline condition:**

### B+C Naive (generated tests, no spec given to Part B, base Qwen APR)
```python
!python pipeline_eval_kaggle.py \
    --condition E2 \
    --dataset /kaggle/input/testmate-humaneval/humaneval_eval_dataset.json
```

> Before running E3/E4/E5, find the PartC adapter path on the current account.
> Run this once in any cell to detect both possible paths:
> ```python
> import os
> _CANDS = [
>     "/kaggle/input/datasets/mohammedmedhat08/testmate-partc-adapter/adapter",
>     "/kaggle/input/datasets/mohammed8medhat/testmate-partc-adapter/adapter",
> ]
> PARTC_ADAPTER = next((p for p in _CANDS if os.path.isdir(p)), None)
> print("PartC adapter:", PARTC_ADAPTER)
> ```
> Then use `$PARTC_ADAPTER` in the shell commands below. If both come back
> `None`, run `!find /kaggle/input -name "adapter_config.json"` to locate it
> and add the path to `_CANDS`.

### B+C with LoRA (generated tests, no spec, TestMate APR)
```python
!python pipeline_eval_kaggle.py \
    --condition E3 \
    --dataset /kaggle/input/testmate-humaneval/humaneval_eval_dataset.json \
    --partc_adapter $PARTC_ADAPTER
```

### Full Pipeline — KEY RESULT (spec → tests → TestMate APR)
```python
!python pipeline_eval_kaggle.py \
    --condition E4 \
    --dataset /kaggle/input/testmate-humaneval/humaneval_eval_dataset.json \
    --partc_adapter $PARTC_ADAPTER
```

### Full Pipeline + Gap-fill (best mode)
```python
!python pipeline_eval_kaggle.py \
    --condition E5 \
    --dataset /kaggle/input/testmate-humaneval/humaneval_eval_dataset.json \
    --partc_adapter $PARTC_ADAPTER
```

### Output produced

Each condition writes to `/kaggle/working/pipeline_results/<condition>/`:
```
E4/
├── results.json        ← per-sample metrics
├── summary.json        ← aggregates (detection, apparent repair, gen time)
└── artifacts/
    ├── HumanEvalFix_000_test.py        ← Part B's generated test
    ├── HumanEvalFix_000_repaired.py    ← Part C's repaired code
    └── ...
```

**Download** each `pipeline_results/<condition>/` folder, save locally as:
```
paper_eval/
└── pipeline_results/
    ├── E2/   (B+C Naive)
    ├── E3/   (B+C with LoRA)
    ├── E4/   (Full Pipeline)
    └── E5/   (Full Pipeline + Gap-fill)
```

---

# Phase 4 — Local: True Repair Rate

Runs on your own machine. Computes the **true repair rate** by running
HumanEval's gold tests against each repaired code artifact.

**Requires:** Python + pytest. No GPU.

For **each** of E2, E3, E4, E5:
```powershell
python PartB/testgen/training/pipeline_eval_local.py `
    --dataset   PartC/humaneval_eval_dataset.json `
    --artifacts paper_eval/pipeline_results/E4/artifacts `
    --results   paper_eval/pipeline_results/E4/results.json `
    --output    paper_eval/pipeline_results/E4/final.json
```
*(repeat with E2, E3, E5 replacing E4 in the three paths)*

Each run prints:
```
=== SUMMARY ===
  apparent_repair_rate                72.00
  true_repair_rate                    61.60     <- the paper number
  plausible_not_correct               17
  plausible_not_correct_pct           10.40
```

---

# Phase 5 — Build the Paper Tables

One command, locally:
```powershell
python PartB/testgen/training/aggregate_results.py `
    --partb_dir    paper_eval/partb_results `
    --pipeline_dir paper_eval/pipeline_results `
    --apr_dir      paper_eval/apr_results `
    --output_dir   paper_eval/paper_tables
```

Prints **Table 1**, **Table 2**, **Table 3** to the terminal and saves them as CSV files:
```
paper_eval/paper_tables/
├── table1_partb.csv      ← Test generation (Baseline ... TestMate)
├── table2_apr.csv        ← APR (Baseline / TestMate)
└── table3_pipeline.csv   ← Full pipeline (B+C Naive / B+C LoRA / Full / Full+Gap-fill)
```

The terminal output also prints **headline numbers** ready to paste into the
paper:
```
HEADLINE NUMBERS
  Part B test quality (baseline -> TestMate):
    Any-pass@1 %    ...  ->  ...
    Line coverage   ...  ->  ...
  Part C APR repair rate (baseline -> TestMate):
    True repair %   67.68  ->  91.46
  Full Pipeline:
    Test detection  ...
    Apparent repair ...
    True repair     ...
```

---

# Troubleshooting

| Problem | Fix |
|---------|-----|
| `No module named 'bitsandbytes'` | Add `!pip install -q bitsandbytes` to Cell 1 and re-run. |
| `adapter_config.json not found` | Your Kaggle dataset folder layout differs. Run `!find /kaggle/input -name "adapter_config.json"` to locate the real path, then edit the `--partc_adapter` flag or the `LORA_PATH` in `kaggle_ablation_testmate.py`. |
| `CUDA out of memory` | Add `import os; os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"` to Cell 1. |
| Ablation script can't find `_ablation_common.py` | Make sure you `%cd /kaggle/working/TestMate/PartB/testgen/training` in Cell 2 first. |
| `git clone` is too slow | Already using `--depth 1`. If it still times out, zip the repo locally and upload it as a Kaggle dataset instead. |

---

# Expected runtime

| Phase | Notebooks | Estimated time each | Total |
|-------|-----------|---------------------|-------|
| Phase 1 (Part B, 7 conditions) | 7 × GPU T4 | ~45 min each | ~5–6 h |
| Phase 2 (APR) | 1 × GPU T4 | ~2 h | ~2 h |
| Phase 3 (Pipeline, 4 conditions) | 4 × GPU T4 | ~1.5 h each | ~6 h |
| Phase 4 (local hidden tests) | local CPU | ~15 min total | 15 min |
| Phase 5 (tables) | local | ~1 min | 1 min |
| **Total** | | | **~14 h** |

Total GPU time ≈ 9 hours, well within Kaggle's free 30 GPU h / week.

---

# File reference

| Script | Phase | Purpose |
|--------|-------|---------|
| `kaggle_ablation_no_rag.py` | 1 — Baseline | Qwen-7B, no RAG, no LoRA |
| `kaggle_ablation_graph_only.py` | 1 — Graph Only | Knowledge graph RAG only |
| `kaggle_ablation_no_graph.py` | 1 — Vector Only | Semantic vector RAG only |
| `kaggle_ablation_full.py` | 1 — Full RAG | All 3 RAG layers |
| `kaggle_ablation_full_with_loop.py` | 1 — Full RAG + Loop | All RAG + self-correction |
| `kaggle_ablation_full_with_plan.py` | 1 — Full RAG + Plan | All RAG + plan-first |
| **`kaggle_ablation_testmate.py`** | **1 — TestMate** | **The production system: RAG + LoRA + Loop** |
| `apr_evaluator.py` (in `PartC/eval/`) | 2 | Bug repair eval (base + TestMate) |
| `pipeline_eval_kaggle.py` | 3 | Full pipeline end-to-end |
| `pipeline_eval_local.py` | 4 | True repair rate (hidden tests) |
| `aggregate_results.py` | 5 | Builds Tables 1, 2, 3 |

---

# What changed after the 2026-05-28 quality/measurement/RAG update

This section records what changed and what to re-run to pick up the new metrics.

## New columns in the ablation JSON (all variants, Phase 1)

| Column | Level | What it contains |
|--------|-------|-----------------|
| `prompt_tokens` | per-file | Total prompt tokens consumed by all `generate_test()` calls for this file (was always 0 before for the `testmate` variant). |
| `completion_tokens` | per-file | Total completion tokens generated for this file. |
| `bes_scores` | per-file | List of BES scores for each test that was accepted through the BES gate (empty for non-testmate variants which skip the BES gate entirely). |
| `mean_bes_score` | per-file | Mean of `bes_scores`, or 0.0 if none accepted. |
| `total_prompt_tokens` | summary | Sum across all files. |
| `total_completion_tokens` | summary | Sum across all files. |
| `mean_bes_score` | summary | Mean of per-file mean_bes_score for files that had at least one accepted test. |

The `testmate` and `testmate_no_bes` variants now report real token counts.
All other variants reported real counts already (the zero bug was testmate-specific because it used `autonomous_loop` internally rather than `generate_single_shot`).

## Self-healing RAG (Tier 4)

The RAG SQLite database (`testmate_rag.db`) now has three additional columns on both `test_examples` and `bad_examples`:
- `use_count` — times a row was retrieved and shown to the model
- `success_count` — times retrieval of that row correlated with an accepted test
- `last_used_at` — last retrieval timestamp

Every `init_db()` call (i.e., every testmate run) automatically migrates existing databases forward with `ALTER TABLE ... ADD COLUMN` (idempotent).

**Trust-factor scoring** (`(success_count+1)/(use_count+2)`) acts as a Laplace-smoothed prior:
- New rows start at 0.5 (neutral) and need ~5 retrievals to develop signal
- High-success rows get a small quality-score boost at retrieval time
- Bad-example rows with <30% success rate after 5+ uses are suppressed (not deleted — just skipped in retrieval)

**Startup decay**: each `init_db()` call runs `UPDATE test_examples SET quality_score = quality_score * 0.95 WHERE last_used_at < 30 days ago`. Rows that no one retrieves for 30 days decay below the 40-point floor in ~10 runs. Hard-delete applies to `bad_examples` rows with 0 successes after 3+ uses and 60 days inactivity — those warnings demonstrably don't help and waste prompt tokens.

## What to re-run

### Minimum re-run (strongly recommended before paper submission)

Re-run only the **TestMate** variant (Phase 1) to get accurate token counts and BES scores:

```python
# Cell 3 (Kaggle, same setup as before)
!python kaggle_ablation_testmate.py
```

Download `results/ablation_testmate_with_mut.json` and replace the old file locally.

Then re-run Phase 5 to rebuild the paper tables:
```powershell
python PartB/testgen/training/aggregate_results.py `
    --partb_dir    paper_eval/partb_results `
    --pipeline_dir paper_eval/pipeline_results `
    --apr_dir      paper_eval/apr_results `
    --output_dir   paper_eval/paper_tables
```

### Optional (for richer ablation comparison)

Re-run all 7 Phase 1 variants if you want `mean_bes_score` to appear in the comparison table for every condition (non-testmate variants will have `mean_bes_score=0.0` since they skip the BES gate, which is correct and informative — it shows how much the BES gate contributes).

### Self-healing RAG warm-up

The self-healing RAG starts cold — `use_count=0` for all existing rows. It only develops trust signal after real runs. For the **first** post-update ablation run: expect identical retrieval scores to before (trust factor = neutral 0.5). By the **second** run (after feedback from run 1 has been written to the DB), retrieval will start preferring examples that actually helped.

To verify the self-healing RAG is working after a run:
```sql
-- Should be > 0 after any testmate run
SELECT COUNT(*) FROM test_examples WHERE use_count > 0;
SELECT COUNT(*) FROM bad_examples WHERE use_count > 0;

-- Check trust distribution
SELECT
  AVG(CAST(success_count AS REAL) / (use_count + 1)) AS mean_trust,
  MIN(quality_score), MAX(quality_score)
FROM test_examples WHERE use_count > 0;
```

## Quality improvements that don't need a re-run to take effect

These are already live in the code — they activate the next time `autonomous_loop` runs:
- **Tautology filter** (`assert x == x` rejected before pytest)
- **Identical-retry skip** (model repeated same test → forced diversity)
- **Stronger semantic dedup** (variable-renamed duplicates caught by normalize())
- **Property-heavy file skip** (>80% `@property` → skip entire file)
- **Stale priming exclusion** (test examples >30 days old excluded from in-context priming)
- **Specific exception handling** (bare `except:` → typed + `logger.debug` for diagnosability)
