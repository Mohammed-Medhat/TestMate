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
