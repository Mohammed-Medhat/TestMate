# TestMate PartB — Results Summary

**Model:** Qwen2.5-Coder-7B-Instruct (4-bit NF4) + LoRA.
**Training:** MBPP only, decontaminated vs HumanEval — *no* SWE-bench, *no* Magicoder, *no* HumanEval/TestEval. All three test sets below are therefore contamination-free.

# Coverage vs Correctness vs Bug-Detection

The three axes that matter — and why a single coverage number is misleading. Coverage = did inputs run the code; pass@1 = are the assertions correct; mutation = would the test *catch a bug*.

| variant / mode | line cov% | pass@1% | mutation% | what it shows |
|---|---|---|---|---|
| TestMate (suite) | 91.6 | 17.6 | 74.1 | coverage-optimized |
| TestMate (quality) | 42.5 | 16.7 | - | correctness-optimized |

**Headline:** run with `--mutation` to fill the bug-detection column and apply the decision rule. Until then, the proven hook is the coverage–correctness gap (91% coverage, ~18% correct assertions).

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
| TestMate (suite mode) | 91.6 | 82.7 | - | ours · TestEval protocol, comprehensive suite |

> TestMate also reports **pass@1 (correct assertions)** and **mutation score** — bug-exposure metrics the paper's coverage task does not measure (it counts tests with wrong assertions as valid). Quality-mode coverage is averaged over all 210 programs; suite mode follows the paper's coverage-maximizing protocol.

## TestEval — recognized benchmark (headline)

| variant | pass@1 | passrate | line_cov% | branch_cov% | graphRAG | vecRAG | BES | wall_s |
|---|---|---|---|---|---|---|---|---|
| testmate | 16.7 | 16.7 | 42.52 | 22.43 | 44.8 | 77.6 | 56.47 | 230.69 |


_(need both `testmate` and `no_rag` for the RAG-lift delta)_

## testgenevallite — realistic code (RAG-context lift)

_(no data — run the sweep for this dataset)_


## HumanEval — universal sanity point

_(no data — run the sweep for this dataset)_


---
Coverage columns are percentages; pass@1/passrate/RAG-hit columns are rates shown as %. `wall_s` is mean seconds/file. Mutation scores (where run) live in the per-variant `*_with_mut.json`; mutation is best-effort and may be null on identity/cluster files.
