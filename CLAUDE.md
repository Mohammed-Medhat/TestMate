# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**TestMate** is an AI-driven system that automates software test generation and bug fixing using Large Language Models (LLMs). It integrates multi-teacher knowledge distillation and Retrieval-Augmented Generation (RAG) to create a lightweight, efficient student model.

The project is split into two major parts:

### PartA: Requirement Extraction & Alignment
- **Purpose**: Extract test case requirements from Software Requirements Specification (SRS) documents
- **Input**: PDF/DOCX SRS documents
- **Output**: JSON files with aligned requirements from the PURE dataset
- **Core Pipeline**: Extract → Segment → Align → Label

### PartB: Test Generation & Bug Fixing with RAG
- **Purpose**: Generate test cases using a 3-layer RAG agent and fix bugs via LLM reasoning
- **Architecture**: 3-layer RAG system combining documentation retrieval (Layer 1), code navigation (Layer 2), and LLM reasoning (Layer 3)
- **Key Models**: 
  - Base: Qwen2.5-Coder-7B (7B parameters, efficient)
  - Fine-tuning: LoRA adapters for knowledge distillation and value reasoning
  - Teacher models: DeepSeek-V3, Qwen 3.5-Coder, Llama 3.1, Mistral Large 2
- **Datasets**: SWE-bench Lite (real GitHub issues with patches)

## Core Architecture

### PartA: SRS2TestCases Pipeline

```
PartA/srs2testcases/
├── run_pipeline.py              # Main entry point (orchestrator)
├── requirements.txt             # Dependencies: spacy, python-docx, PyMuPDF, pandas
├── src/
│   ├── preprocessing/           # Document extraction and processing
│   │   ├── reader_pdf.py       # PDF text extraction (fitz/PyMuPDF)
│   │   ├── reader_docx.py      # DOCX text extraction
│   │   ├── segment_sentences.py # spaCy-based sentence segmentation
│   │   └── align_pure_datasets.py # Fuzzy matching against PURE dataset
│   ├── generation/              # Test generation (future)
│   │   ├── llm_generate.py
│   │   └── templates.py
│   ├── models/                  # ML models (future)
│   │   └── srs_classifier.py
│   └── validation/              # Test validation
│       ├── checks.py
│       └── dedupe.py
├── scripts/
│   ├── align_all_datasets.py   # Batch align PURE datasets
│   ├── test_alignment.py       # Unit tests
│   ├── verify_alignment.py     # Verification tool
│   └── test_pipeline.py        # Pipeline tests
└── data/
    ├── raw/                    # Extracted JSON from SRS docs
    ├── processed/              # Segmented sentences
    └── aligned/                # Fuzzy-matched aligned datasets
```

**Key Concepts**:
- **PURE Dataset**: Annotated requirement classification dataset with 4 splits (train/valid/test/RFIs)
- **Fuzzy Matching**: Uses difflib for text alignment with configurable threshold (default 0.85)
- **Output Format**: JSON with labels (1=requirement, 0=non-requirement, -1=unknown), match scores, and metadata

### PartB: 3-Layer RAG Agent for Test Generation & Bug Fixing

```
PartB/
├── agent.py                     # Main SWE-bench agent orchestrator
├── layer1_docs/                 # RAGFix: External knowledge retrieval
│   ├── docs_retriever.py       # Documentation + Stack Overflow search via FAISS
│   └── __init__.py
├── layer2_code/                 # KGCompass: Code-graph navigation
│   ├── code_navigator.py       # BM25 sparse + semantic vector search
│   ├── graph_rag.py            # Complete graph-RAG (Issue → File → Class → Function)
│   └── __init__.py
├── testgen/                     # Test generation module
│   ├── main.py                 # Self-correcting test generation with AST analysis
│   ├── api_server.py           # REST API server (FastAPI)
│   ├── gui.py                  # Desktop GUI (Tkinter-based)
│   ├── docker_runner.py        # Docker-based execution sandbox
│   ├── rag_store.py            # RAG memory persistence (SQLite)
│   ├── build_knowledge_graph.py # AST parsing → graph construction
│   ├── mutation_testing.py     # Mutmut integration for quality verification
│   ├── kaggle_*.py             # Training on Kaggle T4 GPU
│   ├── full_evaluation.py      # Comprehensive evaluation pipeline
│   └── generated_tests/        # Output: generated test files
├── training/                    # Model training scripts
│   └── swe_bench_ingester.py   # Dataset preparation
├── models/                      # Trained LoRA adapters
│   ├── graphrag_lora/final/    # Graph-RAG trained model
│   └── value_reasoning/        # Value reasoning model
├── testgen_eval_lite/          # Lightweight evaluation
├── utils.py                    # Common utilities
├── ast_parser_complete.py      # Complete AST analysis
├── graph_schema.py             # Node/edge definitions
└── download_datasets.py        # SWE-bench Lite download
```

**Layer 1 (RAGFix - External Knowledge)**:
- Searches official API docs, Stack Overflow, migration guides for target repos
- Uses FAISS for semantic similarity over doc embeddings
- Target repos: astropy, django, flask, matplotlib, pylint, pytest, requests, scikit-learn, seaborn, sphinx, sympy

**Layer 2 (KGCompass - Code Navigation)**:
- Builds knowledge graph: Issue → File → Class → Function
- Sparse retrieval (BM25) for keyword matching
- Semantic retrieval (embeddings) for concept similarity
- Graph traversal for multi-hop context (up to 20 candidate functions)
- Extracts CFG paths, call graphs, complexity metrics via AST

**Layer 3 (LLM Reasoning)**:
- Qwen2.5-Coder-7B with LoRA fine-tuning
- 4-bit quantization via BitsAndBytes
- Generates patch code with context from layers 1 & 2
- Self-correcting loop: runs test, feeds back errors, regenerates

**Test Generation Flow** (main.py):
1. Load target Python file and parse AST for context (CFG paths, raises, callees)
2. Extract docstrings and method dependencies
3. Generate pytest with self-correction (max 15 iterations)
4. Run tests and capture errors
5. Feed errors back to LLM for refinement
6. Store good tests in RAG memory, bad tests marked for learning

## Technology Stack

**PartA**:
- spaCy 3.5+ (NLP, sentence segmentation)
- PyMuPDF (PDF text extraction)
- python-docx (DOCX parsing)
- pandas (data processing)
- pytest (testing)

**PartB**:
- Hugging Face Transformers + PEFT (LoRA fine-tuning)
- PyTorch (4-bit quantization via BitsAndBytes)
- Sentence-Transformers (embeddings)
- FAISS (semantic search)
- NetworkX (graph construction)
- FastAPI (REST API)
- SQLite (RAG memory)
- Mutmut (mutation testing)
- pytest + Coverage.py (test execution & metrics)
- Docker (sandboxed execution)

## Common Commands

### PartA: Requirements Extraction

```bash
# Install dependencies
cd PartA/srs2testcases
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Align PURE datasets (creates data/aligned/*.json)
python scripts/align_all_datasets.py

# Process single SRS document
python run_pipeline.py \
  --input "path/to/document.pdf" \
  --output results/output.json \
  --aligned-dataset data/aligned/pure_train_aligned.json

# Batch process all SRS documents
python run_pipeline.py \
  --batch \
  --input-dir "path/to/req/folder" \
  --output-dir results \
  --aligned-dataset data/aligned/pure_train_aligned.json

# Verify alignment quality
python scripts/verify_alignment.py
python scripts/test_alignment.py  # Unit tests
```

### PartB: Test Generation & Bug Fixing

```bash
# Install dependencies
cd PartB
pip install -r testgen/requirements_docker.txt  # or your env's requirements

# Download SWE-bench Lite dataset
python download_datasets.py

# Build knowledge graph for a repository
python testgen/build_knowledge_graph.py --repo <repo_path>

# Generate tests for a target Python file
python testgen/main.py --target path/to/code.py

# Start REST API server (port 8000)
python testgen/api_server.py

# Launch GUI
python testgen/gui.py

# Run full evaluation pipeline (requires pytest + Coverage + Mutmut)
python testgen/full_evaluation.py

# Quick evaluation (Kaggle-compatible)
python testgen/kaggle_eval_testgen.py

# Run specific test file
pytest PartB/test_graphrag.py -v
pytest PartB/test_context_assembler.py -v
```

### Training (Kaggle T4 GPU)

```bash
# Train on 1000 samples (700 SWE-bench + 300 Magicoder)
python testgen/kaggle_train_testgen.py

# Outputs final model to: PartB/models/graphrag_lora/final/
```

## Key Files to Understand

### PartA - Requirement Alignment
1. **run_pipeline.py**: Orchestrates extraction → segmentation → alignment (main entry)
2. **src/preprocessing/align_pure_datasets.py**: Fuzzy matching logic, similarity scoring
3. **src/preprocessing/reader_pdf.py** & **reader_docx.py**: Document format handling
4. **src/preprocessing/segment_sentences.py**: spaCy-based sentence splitting

### PartB - Test Generation
1. **agent.py**: 3-layer RAG orchestration (loads layers 1, 2, 3)
2. **layer2_code/graph_rag.py**: Complete graph-RAG with IssueNode, FileNode, FunctionNode, etc.
3. **layer1_docs/docs_retriever.py**: FAISS-based external knowledge search
4. **testgen/main.py**: Self-correcting test generation loop (AST → LLM → test → error feedback)
5. **testgen/build_knowledge_graph.py**: AST parsing to construct knowledge graph
6. **testgen/rag_store.py**: SQLite-backed RAG memory (stores examples, bad cases, patterns)

## Design Patterns

### PartA
- **Pipeline Pattern**: Sequential stages (extract → segment → align → label)
- **Fuzzy Matching**: Similarity-based record linking (SequenceMatcher)
- **Lazy Loading**: Load aligned dataset only if needed

### PartB
- **3-Layer RAG**: Modular knowledge sources (docs, code, LLM)
- **Self-Correction Loop**: Generate → Execute → Feedback → Regenerate
- **Graph-Based Context**: Multi-hop traversal (Issue → File → Class → Function)
- **LoRA Fine-Tuning**: Parameter-efficient adaptation without full model retraining
- **Memory Layer**: SQLite RAG store for pattern reuse and error learning

## Important Architectural Decisions

1. **Lightweight Student Model**: Qwen2.5-Coder-7B (7B params) via knowledge distillation from 4 teacher models
2. **4-bit Quantization**: Reduces memory footprint for deployment while maintaining quality
3. **Graph-Based Navigation**: Avoids semantic drift by grounding in code structure (AST)
4. **Modular Layers**: Each layer (docs, code, LLM) can be swapped independently
5. **Self-Healing Memory**: PartB stores both successful tests and failure patterns for continuous learning

## Notes on PartA vs PartB Integration

- **PartA output** (aligned requirements) can seed **PartB's test generation** with requirement labels
- PartA provides structured requirements; PartB generates test implementations
- Current integration is conceptual; full pipeline requires custom connectors
- FIXMATE_PLAN.md describes future repo-level understanding using PartA requirements

