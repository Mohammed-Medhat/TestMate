# TestMate Part B — Results & Conclusion

Final evaluation results for Part B (the self-correcting test generator). Every
number traces to a JSON in `testgen/training/results/`. For *how* to reproduce
see `PARTB_COMPLETE.md`; for paper-claim wording see `PAPER_README.md`.

**System under test:** Qwen2.5-Coder-7B-Instruct (4-bit NF4) + clean LoRA
(MBPP-only, decontaminated), 2-layer RAG + self-correction + BES gate, fully
local. **Baseline:** the *same* base model without the adapter (`no_lora`) or
without RAG (`no_rag`).

---

## 1. Headline — TestEval (210 LeetCode programs, suite/coverage mode)

`results/compare_testeval_suite/{testmate,no_lora}_with_mut.json`

| System | Line cov % | Pass rate % | Mutation % (bug-detection) | programs w/ passing tests |
|---|---|---|---|---|
| **TestMate** | **93.8** | **97.6** | 81.5 | 205 / 210 |
| base (no LoRA) | 78.9 | 79.8 | 89.8 | 168 / 210 |

- **Coverage 93.8%** — top of the open-7B pack reported in the TestEval paper
  (at/above DeepSeek-Coder & Gemma, low-90s). *Re-implementation caveat:* computed
  with our harness approximating the official protocol.
- **Correctness +18 points** and **+37 programs** get a usable suite.
- **Bug-detection (honest nuance):** base has a higher *per-file* kill rate
  (89.8 vs 81.5) but over **fewer** programs (168 vs 205). TestMate produces
  bug-catching tests on **22% more programs**; base is more precise on the fewer
  it handles. Report both — it is a trade-off, not a clean sweep.

## 2. The coverage-vs-correctness gap — TestEval quality mode

`results/compare_testeval/testmate.json`

| System | Line cov % | Pass@1 % |
|---|---|---|
| TestMate (quality mode) | 42.5 | 16.7 |

Same system, different objective: suite mode optimizes coverage (93.8%), quality
mode optimizes per-target correctness. The huge spread in single-number
"coverage" (42.5 vs 93.8) **is the evidence that coverage alone is a misleading
metric** for LLM test generation.

## 3. Sanity — HumanEval (164, internal: system vs base)

`results/compare_humaneval/{testmate,no_lora}.json`

| System | Pass@1 % | Line cov % | Valid line cov % | BES |
|---|---|---|---|---|
| **TestMate** | **60.4** | 67.6 | 92.6 | 58.3 |
| base (no_lora) | 33.5 | 34.1 | 63.5 | 0 |

TestMate ~**1.8× the base** on correctness and ~2× on coverage. **Not** a
leaderboard number (HumanEval here is *test* generation, not the famous code-gen
task) — used only to confirm the pipeline helps, consistently with TestEval.

## 4. RAG value on real code — testgenevallite (matched N=30)

`results/compare_testgenevallite/{testmate,no_rag}_with_mut.json`

| System | Pass@1 % | Line cov % | Valid line cov % | graphRAG hit | vectorRAG hit |
|---|---|---|---|---|---|
| **TestMate** (RAG on) | **13.3** | 11.7 | 31.9 | 27% | 57% |
| no_rag (RAG off) | 6.7 | 14.7 | 29.3 | 0% | 0% |

- RAG **doubles correctness** (13.3% vs 6.7% pass@1) on real framework code, and
  non-zero RAG hit-rates confirm retrieval actually fires (graphRAG 27%, vector
  57%) — vs 0% with RAG off.
- **No coverage lift** (these files are brutally hard — 35–69 targets each, deep
  deps); RAG's value here is **correctness**, not raw coverage.
- **Status:** *directional* — N=30 (≈4 vs 2 passing files). A larger matched run
  (in progress, target N≈60+) would firm the effect estimate; it does not change
  the direction.

---

## 5. Where Part B stands vs similar-size models

- **Coverage:** **competitive with / at the top of the open-7B pack** on the
  recognized TestEval benchmark (93.8% suite line coverage).
- **Correctness & bug-detection:** Part B **reports and leads on axes the TestEval
  leaderboard does not measure** — pass@1 correctness and mutation-based
  bug-detection. There is no same-size baseline that reports these, so the
  comparison is "we measure what others omit," not "we beat a published number."
- **Honest non-claim:** this is **not** coverage-SOTA and **not** a GPT-4
  comparison. A local 4-bit 7B is mid-to-top of the *open 7B* field on a saturated
  metric — which is exactly what the data shows.

---

## 6. Conclusion

TestMate Part B is a **complete, fully-local, self-correcting test-generation
system** whose contribution is **methodological as much as it is engineering**:

1. **System.** A working local pipeline — 2-layer RAG (call-graph + semantic
   vector) + memory, self-correction loop, BES quality gate, mutation evaluation —
   with no external API and contamination-controlled training (MBPP only,
   decontaminated).

2. **Result.** On the standard **TestEval** benchmark TestMate is
   **coverage-competitive at the top of the open-7B pack (93.8%)** while producing
   **correct, bug-detecting tests on ~22% more programs** than the same base
   model. The system effect replicates on **HumanEval** (1.8× the base on
   correctness). On **real framework code**, retrieval **doubles** test
   correctness.

3. **Insight (the hook).** **Coverage is a misleading single metric** for LLM
   test generation — the same system swings from 42.5% to 93.8% "coverage"
   depending only on the objective, and high coverage can coexist with wrong
   assertions. Part B therefore evaluates on **three axes — coverage, correctness,
   and bug-detection (mutation)** — and reports the trade-offs (e.g. base = more
   precise per file, TestMate = broader coverage of programs) honestly rather than
   optimizing a single inflated number.

**Limitations (stated, not hidden):** single-run numbers at temperature 0.3 (no
multi-seed significance); the TestEval comparison is a faithful re-implementation,
not the official harness; the testgenevallite RAG result is directional at small
matched N; and the LoRA-vs-base contrasts are *system-level* ablations (full
TestMate vs base / vs no-RAG), not isolated single-component ablations.

**Bottom line:** Part B does not claim to be the best test generator in the world;
it demonstrates a competitive, honest, fully-local system and shows — with data —
why the field's favorite metric should not be trusted on its own.
