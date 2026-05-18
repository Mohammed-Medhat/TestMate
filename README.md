# TestMate: Automated Test Generation via Multi-Model Knowledge Distillation and RAG

## Overview

**TestMate** is an AI-driven system that automates software test generation, refinement, and bug fixing using Large Language Models (LLMs). It integrates **multi-teacher knowledge distillation** and **Retrieval-Augmented Generation (RAG)** to create a lightweight, efficient, and scalable student model that outperforms single-model solutions.

## Motivation

Manual test case creation is slow, expensive, and often incomplete. Existing LLM-based test generators are powerful but costly and not optimized for continuous integration. TestMate solves these challenges by:

* Distilling knowledge from multiple teacher models (DeepSeek-V3, Qwen 3.5-Coder, Llama 3.1, Mistral Large 2).
* Building an open, efficient student model (Qwen 3.5 Coder 7B).
* Integrating RAG for memory and reusability of validated test cases.
* Combining bug fixing, coverage visualization, and dependency analysis in one system.

## Objectives

* Automatically generate test cases from source code (Python/GitHub).
* Perform **multi-teacher distillation** and build a **Meta-Student** through output merging.
* Implement **RAG-based memory** for storing validated tests globally and per user.
* Integrate **mutation testing**, **coverage heatmaps**, and **Control Flow Graphs (CFGs)**.
* Add **Automated Program Repair (APR)** for bug fixing.
* Visualize dependencies using **dependency graphs**.
* Deliver a functional prototype with a user-friendly interface.

## Key Features

* **Automated Test Generation** using LLMs and RAG.
* **Knowledge Distillation Pipeline** with voting and filtering.
* **Bug Fixing** powered by APR.
* **Coverage Analysis** via heatmaps.
* **Visualization** of test paths and dependencies (CFGs & dependency graphs).
* **Open & Efficient Architecture** optimized for real-world integration.

## Tools & Technologies

* **Languages:** Python
* **Libraries:** Hugging Face Transformers, LangChain/LangGraph, spaCy, Pytest, Coverage.py, Mutmut
* **AI Models:** DeepSeek-V3, Qwen 3.5-Coder, Llama 3.1, Mistral Large 2
* **Vector Store:** FAISS / Weaviate
* **Bug Fixing:** PyFix (APR)
* **Visualization:** Matplotlib, Seaborn, Radon, Lizard, NetworkX
* **Environment:** Jupyter, VSCode, Docker

## References

Based on key works in:

* Knowledge Distillation (Chen et al., 2024)
* Automated Program Repair (Zhang et al., 2023; Meng et al., 2024)
* Retrieval-Augmented Generation (Gao et al., 2023; Zhao et al., 2024)
* Graph-based RAG and Dependency-Aware Reranking (Li et al., 2025)

---

## Project Structure

```
TestMate/
├── PartA/                          # Requirement Extraction & Alignment
│   ├── srs_pipeline/               # SRS PDF/DOCX → labeled requirements (spaCy + fuzzy match)
│   │   ├── run_pipeline.py         # CLI entry point
│   │   ├── pipeline_api.py         # Programmatic API (used by unified_app)
│   │   └── src/preprocessing/      # PDF, DOCX readers + sentence segmentation + PURE alignment
│   ├── readme_extractor/           # README → features + test scenarios (FastAPI + Qwen LLM)
│   │   ├── pipeline_api.py         # Programmatic API (used by unified_app)
│   │   └── src/                    # API routes, services, models, utils
│   └── datasets/                   # Input data
│       ├── srs_documents/          # 80+ SRS documents (PDF/DOCX)
│       └── pure_annotated/         # PURE dataset CSVs for requirement alignment
│
├── PartB/                          # Test Generation & Bug Fixing
│   ├── agent.py                    # 3-layer RAG orchestrator (main entry)
│   ├── testgen/                    # Core test generation engine
│   │   ├── main.py                 # Autonomous self-correcting loop (AST → LLM → test → feedback)
│   │   ├── api_server.py           # FastAPI server (standalone mode)
│   │   ├── pipeline_api.py         # Programmatic API (used by unified_app)
│   │   ├── docker_runner.py        # Sandboxed test execution
│   │   ├── rag_store.py            # SQLite RAG memory (examples + patterns)
│   │   ├── build_knowledge_graph.py# AST → knowledge graph
│   │   ├── intense_mode.py         # DPO multi-iteration mode
│   │   ├── mutation_testing.py     # Mutmut integration
│   │   ├── training/               # Kaggle/Colab training & evaluation scripts
│   │   ├── langchain/              # LangChain-based agent wrappers
│   │   ├── tools/                  # Demos, debug utilities, reports
│   │   ├── templates/              # HTML for api_server dashboard
│   │   └── static/                 # CSS/JS assets
│   ├── layers/                     # RAG layers
│   │   ├── docs/                   # Layer 1: external docs + Stack Overflow (FAISS)
│   │   └── code/                   # Layer 2: code graph navigation (BM25 + embeddings)
│   ├── models/                     # Trained LoRA adapters
│   │   └── graphrag_lora/final/    # Production GraphRAG LoRA
│   ├── value_reasoning_model/      # Value reasoning LoRA adapter
│   ├── eval_lite/                  # Lightweight evaluation suite
│   ├── training_data/              # SWE-bench dataset ingestion
│   ├── tests/                      # Unit tests
│   └── outputs/                    # Runtime artifacts (generated tests, results, cloned repos)
│
└── unified_app/                    # Desktop Application (Electron + React + TypeScript)
    ├── electron/                   # Electron main process (spawns Python backend)
    ├── src/                        # React UI (Landing + 3-mode workspace)
    │   ├── components/             # Landing, LeftSidebar, MainContent, RightSidebar, Modal
    │   └── App.tsx                 # Root component + all state management
    └── server.py                   # FastAPI backend (port 8080, all API routes)
```

## Running the App

### Unified Desktop App (recommended)
```bash
cd unified_app
npm install
npm run dev        # opens Electron window with hot-reload
```

### Part A standalone
```bash
cd PartA/srs_pipeline
python run_pipeline.py --input doc.pdf --output results/out.json
```

### Part B standalone
```bash
cd PartB/testgen
python api_server.py    # FastAPI on :8000
```
