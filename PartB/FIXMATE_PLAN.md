# FixMate: Repo-Level AI Test Generation

## Vision
**FixMate** is an AI-powered system that **replaces manual test engineers** by:
- Understanding entire repositories at code-graph level
- Generating comprehensive test suites automatically
- Using advanced Graph RAG techniques for context-aware generation

---

## Core Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         INPUT                                    │
│     Repository URL  /  Source Code                              │
│     (Project Requirements → Future Scope)                        │
└───────────────────────────┬─────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│              REPO-LEVEL UNDERSTANDING (Graph RAG)               │
├─────────────────────────────────────────────────────────────────┤
│  Layer 1: Documentation Retriever                               │
│    - Testing patterns & best practices                          │
│    - API docs, Stack Overflow                                   │
├─────────────────────────────────────────────────────────────────┤
│  Layer 2: Code Navigator (Knowledge Graph)                      │
│    - AST parsing of ALL files                                   │
│    - Function call graphs                                       │
│    - Class inheritance trees                                    │
│    - Module dependency analysis                                 │
│    - Complexity hotspot detection                               │
│    - **CFG Path Analysis** (detect execution paths)             │
├─────────────────────────────────────────────────────────────────┤
│  Layer 3: Orchestrator                                          │
│    - Prioritizes what to test                                   │
│    - Routes to appropriate generators                           │
│    - Combines multi-source context                              │
└───────────────────────────┬─────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│              INTELLIGENT TEST GENERATION                         │
├─────────────────────────────────────────────────────────────────┤
│  1. Unit Tests      - Per function with mocks                   │
│  2. Integration     - Cross-module interactions                 │
│  3. Edge Cases      - Boundary values, null checks              │
│  4. Error Handling  - Exception scenarios                       │
│  5. **Path Coverage** - Tests for each CFG path                 │
└───────────────────────────┬─────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│                         OUTPUT                                   │
│     Complete Test Suite  /  Coverage Report  /  Heatmaps        │
└─────────────────────────────────────────────────────────────────┘
```

---

## Advanced Techniques

| Technique | Description | Layer |
|-----------|-------------|-------|
| **Knowledge Graph** | Build code relationship graph | Layer 2 |
| **Call Graph Analysis** | Trace function dependencies | Layer 2 |
| **CFG Path Analysis** | Detect all execution paths | Layer 2 |
| **Path-Based Testing** | Generate tests for each path | Layer 2 |
| **Embedding Similarity** | Find similar test patterns | Layer 1 |
| **Complexity Analysis** | Prioritize complex functions | Layer 2 |
| **Context Window Optimization** | Smart context selection | Layer 3 |
| **Self-Correction Loop** | Retry on syntax error (max N tries) | Generation |
| **Mock Generation** | Auto-generate mocks for external deps | Generation |
| **Test Prioritization** | Rank tests by coverage/risk | Layer 3 |
| **Mutation Testing** | Verify tests catch injected bugs | Validation |
| **Incremental Testing** | Git-diff aware, test only changes | Optimization |
| **Natural Language Specs** | "Test login fails..." → test code | Input |
| **Test Deduplication** | Merge similar/redundant tests | Post-process |

---

## Project Scope

| Feature | Current Focus | Future Scope |
|---------|--------------|--------------|
| Repo-level parsing | ✅ | - |
| Knowledge graph building | ✅ | - |
| Test generation | ✅ | - |
| Test execution | ✅ | - |
| Coverage analysis | ✅ | - |
| CFG Path Analysis | ✅ | - |
| Self-Correction Loop | ✅ | - |
| Mock Generation | ✅ | - |
| Test Prioritization | ✅ | - |
| **Mutation Testing** | ✅ | - |
| **Project Context Analyzer** | - | ✅ Future |
| Incremental Testing | - | ✅ |
| Natural Language Specs | - | ✅ |
| Test Deduplication | - | ✅ |
| Coverage heatmaps | - | ✅ |
| **Code Repair (APR)** | - | ✅ Future |

---

## Current Implementation Status

### ✅ Built
| Component | Location |
|-----------|----------|
| Layer 1: Docs Retriever | `layer1_docs/` |
| Layer 2: Code Navigator | `layer2_code/` |
| Layer 3: Orchestrator | `orchestrator/` |
| Test Gen Training | `testgen/` |
| Evaluation Scripts | `testgen/` |

### ❌ Missing (Priority)
| Component | Next Step |
|-----------|-----------|
| ~~**RAG → Test Gen Integration**~~ | ✅ FIXED (2026-02-09) |
| Repo-level orchestration | Process entire repos |

---

## Evaluation Results

| Metric | Baseline (No RAG) | Fine-tuned (No RAG) | Target (With RAG) |
|--------|-------------------|---------------------|-------------------|
| Syntax Valid | 63.67% | 71.33% | 80%+ |
| Relevance | 20% | 8.89% | 40%+ |
| High Quality | 46% | 11.33% | 50%+ |

**Key Insight**: Fine-tuning WITHOUT Graph RAG context = worse results. RAG integration is critical!

---

## Next Steps

1. **Integrate Graph RAG with Test Generation**
2. **Implement repo-level orchestration**
3. **Re-evaluate with RAG context**
4. **Demo for seminar**

---

*Code Repair (Stage 3) is future scope - focus is on comprehensive test generation.*

---

## Changelog

### 2026-02-09: Major Graph RAG Integration + 10/10 Upgrades

#### Summary
Fixed critical data flow issues and implemented research-grade upgrades to achieve 10/10 project quality.

#### Files Created
| File | Purpose |
|------|---------|
| `testgen/visualize_paths.py` | CFG visualization for seminar slides |

#### Files Modified

**1. `layer2_code/graph_rag.py`**
- Added `paths: List[str] = []` field to `FunctionNode` class
- CFG execution paths now stored directly in graph nodes

**2. `testgen/build_knowledge_graph.py`**
- Added `CFGPathExtractor` class (lines 37-70)
  - Traces IF/ELSE/RAISE/RETURN/FOR/WHILE/TRY blocks
  - Prioritizes "unhappy paths" (RAISES)
  - Limits output to top 5 most complex paths
- Updated `_extract_function()` to extract and store CFG paths
- Updated `build_from_repo()` to pass paths to `FunctionNode`

**3. `testgen/kaggle_train_testgen.py`**
- Updated `GraphContextExtractor.get_2hop_context()` to retrieve paths from graph
- Updated `format_graph_context()` to include "Execution Paths (MUST COVER)" section
- Training prompts now include CFG paths to guide test generation

**4. `testgen/kaggle_eval_testgen.py`**
- Added `GraphContextExtractor` class for graph loading
- Added `extract_function_name()` helper
- Added `calculate_relevance_semantic()` - uses sentence-transformers embeddings
- Added `detect_smart_mocks()` - 2-hop dependency detection
- Added `run_ab_comparison()` - A/B test for demo slides
- Updated `generate_test()` to use graph paths with traceability logging

#### Key Fixes
| Issue | Fix |
|-------|-----|
| CFG paths extracted but not stored | Added `paths` field to `FunctionNode` |
| Training ignored graph context | Updated `GraphContextExtractor` |
| Eval used basic regex | Integrated graph loader + semantic similarity |
| Mocks missed nested deps | Added 2-hop smart mock detection |

#### New Features (10/10 Upgrades)
1. **Semantic Similarity**: Cosine similarity with embeddings (vs word overlap)
2. **Smart Mocks**: 2-hop graph lookup for nested dependencies
3. **A/B Comparison**: Side-by-side demo: Graph RAG vs Standard RAG
4. **CFG Visualization**: Generate images for seminar slides

#### Next Steps
- [ ] Wait for graph build to complete
- [ ] Upload graph to Kaggle
- [ ] Run training with Graph RAG context
- [ ] Run evaluation and compare metrics
- [ ] Generate demo images with `visualize_paths.py`
