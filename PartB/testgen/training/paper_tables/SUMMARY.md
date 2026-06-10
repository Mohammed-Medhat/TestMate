# TestMate PartB — Results Summary

**Model:** Qwen2.5-Coder-7B-Instruct (4-bit NF4) + LoRA.
**Training:** MBPP only, decontaminated vs HumanEval — *no* SWE-bench, *no* Magicoder, *no* HumanEval/TestEval. All three test sets below are therefore contamination-free.

# Coverage vs Correctness vs Bug-Detection

The three axes that matter — and why a single coverage number is misleading. Coverage = did inputs run the code; pass@1 = are the assertions correct; mutation = would the test *catch a bug*.

| variant / mode | line cov% | pass@1% | mutation% | what it shows |
|---|---|---|---|---|
| TestMate (suite) | 93.8 | 97.6 | 81.5 | coverage-optimized |
| base (suite, no_lora) | 78.9 | 80.0 | 89.8 | raw model, suite |
| TestMate (quality) | 42.5 | 16.7 | - | correctness-optimized |

**Headline (data-supported):** the contribution is the *insight + system*, not beating the base per file (mutation 81.5% vs base 89.8% per file). TestMate produces bug-catching suites on **more programs** (205 vs 168); coverage is misleading (93.8% cov / 97.6% pass rate in suite mode, 42.5% cov / 16.7% pass rate in quality mode).

# TestEval — TestMate vs paper

| model | line cov% | branch cov% | exec% | note |
|---|---|---|---|---|
| GPT-4o | 98.7 | 97.2 | 100.0 | paper / API |
| GPT-3.5-turbo | 97.4 | 96.3 | 100.0 | paper / API |
| GPT-4-turbo | 94.8 | 96.1 | 100.0 | paper / API |
| DeepSeek-Coder-6.7B | 93.5 | 91.6 | 82.4 | paper |
| Gemma-7B | 93.2 | 91.5 | 64.6 | paper |
| Llama3-8B | 91.0 | 89.0 | 82.2 | paper |
| CodeQwen1.5-7B | 90.7 | 86.9 | 84.3 | paper |
| CodeLlama-7B | 86.1 | 81.6 | 73.9 | paper |
| TestMate (quality mode) | 42.5 | 22.4 | - | ours · correctness-optimized, 1 test/prog |
| TestMate (suite mode) | 93.8 | 88.2 | - | ours · TestEval protocol, comprehensive suite |

> TestMate also reports **pass@1 (correct assertions)** and **mutation score** — bug-exposure metrics the paper's coverage task does not measure (it counts tests with wrong assertions as valid). Quality-mode coverage is averaged over all 210 programs; suite mode follows the paper's coverage-maximizing protocol.

## TestEval — recognized benchmark (headline)

| variant | pass@1 | passrate | line_cov% | branch_cov% | graphRAG | vecRAG | BES | wall_s |
|---|---|---|---|---|---|---|---|---|
| testmate | 16.7 | 16.7 | 42.52 | 22.43 | 44.8 | 77.6 | 56.47 | 230.69 |

## testgenevallite — realistic code (RAG-context lift)

| variant | pass@1 | passrate | line_cov% | branch_cov% | graphRAG | vecRAG | BES | wall_s |
|---|---|---|---|---|---|---|---|---|
| testmate | 13.3 | 13.3 | 11.68 | 0.83 | 26.7 | 56.7 | 5.33 | 399.92 |
| no_rag | 6.7 | 6.7 | 14.67 | 0.72 | 0.0 | 0.0 | 0.00 | 322.06 |


**RAG lift (testmate − no_rag) (matched-N=30 (files run by both testmate and no_rag; testmate original N=80)):** Δline_cov = **-2.99%**, Δbranch_cov = **+0.11%**, Δpass@1 = **+6.7 pts**


## HumanEval — universal sanity point

| variant | pass@1 | passrate | line_cov% | branch_cov% | graphRAG | vecRAG | BES | wall_s |
|---|---|---|---|---|---|---|---|---|
| testmate | 60.4 | 60.4 | 67.64 | 39.56 | 14.6 | 85.4 | 58.34 | 62.25 |
| no_lora | 33.5 | 37.4 | 34.10 | 20.57 | 16.5 | 86.6 | 0.00 | 113.45 |


---
Coverage columns are percentages; pass@1/passrate/RAG-hit columns are rates shown as %. `wall_s` is mean seconds/file. Mutation scores (where run) live in the per-variant `*_with_mut.json`; mutation is best-effort and may be null on identity/cluster files.
