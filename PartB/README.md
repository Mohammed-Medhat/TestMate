# SWE-bench Lite RAG Agent

A multi-layer RAG system for solving real-world GitHub issues from SWE-bench Lite benchmark.

## Architecture

```
┌────────────────────────────────────────────────┐
│     Layer 3: LLM Reasoning (Qwen 7B + LoRA)    │
└────────────────────────────────────────────────┘
                      ↑
        ┌─────────────┴─────────────┐
        ↓                           ↓
┌─────────────────┐     ┌─────────────────────┐
│ Layer 1: Docs   │     │ Layer 2: Code Nav   │
│ (RAGFix)        │     │ (KGCompass)         │
└─────────────────┘     └─────────────────────┘
```

## Papers

- **KGCompass** (arXiv 2025): Multi-hop graph traversal
- **RAGFix** (NSF/IEEE 2025): External knowledge retrieval
- **RAG Traceability** (MCSE 2025): Requirement-code linking
- **Knowledge Distillation** (arXiv 2025): Efficient training

## Structure

```
PartB/
├── agent.py                 # Main agent (orchestrator)
├── layer1_docs/             # Documentation retriever
│   └── docs_retriever.py
├── layer2_code/             # Codebase navigator
│   ├── graph_rag.py         # Full KGCompass implementation
│   └── code_navigator.py    # BM25 + symbol search
├── training/                # Model training
│   ├── kaggle_train.py      # Kaggle training notebook
│   └── swe_bench_ingester.py
├── evaluation/              # Evaluation scripts
│   ├── kaggle_eval.py
│   └── convert_predictions.py
└── data/                    # Datasets
```

## Usage

### Training (Kaggle T4 GPU)
```python
# Copy training/kaggle_train.py to Kaggle notebook
# Run with: train_with_graphrag(swe_samples=500, magic_samples=500)
```

### Evaluation
```python
from agent import SWEBenchAgent, SWEBenchIssue

agent = SWEBenchAgent(model_path="models/graphrag/final")
issue = SWEBenchIssue(
    instance_id="django__django-11001",
    repo="django",
    problem_statement="..."
)
patch = agent.solve(issue, repo_path="repos/django")
```

## Target Repositories
astropy, django, flask, matplotlib, pylint, pytest, requests, scikit-learn, seaborn, sphinx, sympy
