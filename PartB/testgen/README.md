# Test Generation Module

Trains and evaluates models for generating test cases that reproduce bugs.

## Files

| File | Purpose | Run On |
|------|---------|--------|
| `kaggle_train_testgen.py` | Train model (700 SWE + 300 Magicoder) | Kaggle T4 |
| `kaggle_eval_testgen.py` | Quick heuristic evaluation | Kaggle T4 |
| `full_evaluation.py` | Comprehensive research-grade evaluation | Local |

## Metrics (Based on Research Papers)

### Quick Evaluation (Kaggle)
- Syntax validity
- Has test function
- Has assertions
- Relevance score

### Full Evaluation (Local)
From **TestEval (2024)** and **Mutation 2024**:

| Category | Metrics |
|----------|---------|
| **Syntax** | Valid Python, imports, test function, assertions |
| **Compilation** | Compile success, error messages |
| **Execution** | Run success, execution time |
| **Coverage** | Line, branch, statement coverage |
| **Mutation** | Mutation score, mutants killed/total |
| **Quality** | Relevance, complexity, assertion count |

## Usage

### Train (Kaggle ~2 hours):
```python
train_testgen(swe_samples=700, magic_samples=300)
```

### Quick Eval (Kaggle):
```python
# Use kaggle_eval_testgen.py
```

### Full Eval (Local with Docker):
```bash
pip install coverage mutmut pytest
python full_evaluation.py
```

## Output

```json
{
  "syntax_valid_pct": 75.0,
  "execution_success_pct": 60.0,
  "avg_line_coverage": 45.2,
  "avg_mutation_score": 0.35,
  "valid_and_useful_pct": 70.0
}
```
