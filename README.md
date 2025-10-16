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

