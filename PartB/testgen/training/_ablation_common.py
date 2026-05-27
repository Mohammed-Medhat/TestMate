"""
Shared ablation harness for the Kaggle ablation scripts.

6 entry-point scripts use this module:
  1. kaggle_ablation_no_rag.py           (raw LLM, plan OFF, loop OFF)
  2. kaggle_ablation_graph_only.py       (graph only)
  3. kaggle_ablation_no_graph.py         (vector only)
  4. kaggle_ablation_full.py             (all RAG)
  5. kaggle_ablation_full_with_loop.py   (all RAG + self-correction loop)
  6. kaggle_ablation_full_with_plan.py   (all RAG + explicit plan-first generation)

All run on Kaggle T4 GPU.
Model:   Qwen2.5-Coder-7B-Instruct (4-bit NF4, NO LoRA, loaded from HuggingFace).
Dataset: PartB/eval_lite/testgen_eval_files/ (all 102 files).

Two-pass execution:
  Phase 1 — generation + all metrics EXCEPT mutation → results/ablation_<v>.json
  Phase 2 — mutmut on each generated test            → results/ablation_<v>_with_mut.json
"""
from __future__ import annotations

import os
import sys
import ast
import re
import json
import time
import gc
import random
import subprocess
import tempfile
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("ablation")

# ── Paths ──────────────────────────────────────────────────────────────────
_THIS = Path(__file__).resolve()
_TESTGEN_DIR = _THIS.parent.parent             # PartB/testgen/
_PARTB_DIR   = _TESTGEN_DIR.parent             # PartB/
_EVAL_DIR    = _PARTB_DIR / "eval_lite" / "testgen_eval_files"

sys.path.insert(0, str(_TESTGEN_DIR))
sys.path.insert(0, str(_PARTB_DIR))


# ──────────────────────────────────────────────────────────────────────────
# 1. CONFIG
# ──────────────────────────────────────────────────────────────────────────

@dataclass
class AblationConfig:
    """Toggleable components for one ablation run."""
    variant: str
    enable_layer1_docs: bool
    enable_layer2_graph: bool
    enable_layer2_vector: bool
    enable_rag_memory: bool
    enable_self_correction_loop: bool
    enable_plan_mode: bool = False
    enable_lora: bool = False
    lora_path: Optional[str] = None
    max_retries: int = 3
    sample_size: Optional[int] = None             # None = use ALL files in eval_lite
    max_new_tokens: int = 1024
    temperature: float = 0.3
    skip_bes_gate: bool = False                   # testmate_no_bes ablation only


# ──────────────────────────────────────────────────────────────────────────
# 2. MODEL LOADER
# ──────────────────────────────────────────────────────────────────────────

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-Coder-7B-Instruct"


def load_qwen_base_4bit(model_id: str = DEFAULT_MODEL_ID):
    """Load Qwen2.5-Coder-7B-Instruct in 4-bit NF4 from HuggingFace. NO LoRA applied."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    os.environ.setdefault(
        "PYTORCH_CUDA_ALLOC_CONF",
        "expandable_segments:True,garbage_collection_threshold:0.6",
    )

    print(f"🧠 Loading {model_id} from HuggingFace (4-bit NF4, no LoRA)...")

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb,
        device_map={"": 0},
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
    )
    model.eval()
    if torch.cuda.is_available():
        used = torch.cuda.memory_allocated() / 1e9
        print(f"   ✅ Loaded — {used:.1f} GB VRAM in use")
    return model, tokenizer


def attach_lora(model, lora_path: str):
    """Attach a LoRA adapter on top of the base 4-bit model."""
    from peft import PeftModel
    posix = Path(lora_path).resolve().as_posix()
    print(f"🔌 Attaching LoRA adapter from {posix}")
    return PeftModel.from_pretrained(model, posix)


# ──────────────────────────────────────────────────────────────────────────
# 3. DATASET LOADER
# ──────────────────────────────────────────────────────────────────────────

def load_test_dataset(eval_dir: Path = _EVAL_DIR,
                      sample_size: Optional[int] = None,
                      seed: int = 42) -> list[dict]:
    """
    Load source files from eval_lite/testgen_eval_files/.
    `sample_size=None` → use ALL files. Otherwise random-sample N.
    """
    if not eval_dir.exists():
        raise FileNotFoundError(f"Eval dataset not found: {eval_dir}")

    all_files = sorted(eval_dir.glob("*.py"))
    print(f"📁 Found {len(all_files)} files in {eval_dir.name}/")

    if sample_size is None:
        selected = all_files
    else:
        rng = random.Random(seed)
        selected = rng.sample(all_files, min(sample_size, len(all_files)))

    dataset = []
    for f in selected:
        try:
            src = f.read_text(encoding="utf-8", errors="ignore")
            ast.parse(src)
            # Sanitize stem so it's always a valid Python identifier.
            # Files like "0_serializer.py" would cause LLM to generate
            # `from 0_serializer import …` → SyntaxError at test collection.
            raw_stem = f.stem
            safe_stem = re.sub(r'^(\d)', r'm\1', raw_stem)
            dataset.append({
                "id":          raw_stem,
                "module_id":   safe_stem,
                "filename":    f.name,
                "abs_path":    str(f),
                "source_code": src,
                "lines":       len(src.splitlines()),
            })
        except (SyntaxError, OSError):
            continue

    print(f"📊 Loaded {len(dataset)} parseable files"
          f" ({'ALL' if sample_size is None else f'sampled {sample_size}'})")
    return dataset


# ──────────────────────────────────────────────────────────────────────────
# 4. RAG CONTEXT BUILDER (conditional per ablation)
# ──────────────────────────────────────────────────────────────────────────

@dataclass
class RAGContext:
    docs_snippets:    list[str] = field(default_factory=list)
    graph_paths:      list[str] = field(default_factory=list)
    semantic_examples: list[str] = field(default_factory=list)
    memory_examples:  list[str] = field(default_factory=list)
    graph_hop_depth:  int = 0

    def is_empty(self) -> bool:
        return not (self.docs_snippets or self.graph_paths
                    or self.semantic_examples or self.memory_examples)

    def total_chars(self) -> int:
        return sum(len(s) for s in (self.docs_snippets + self.graph_paths
                                     + self.semantic_examples + self.memory_examples))

    def all_text(self) -> str:
        return " ".join(self.docs_snippets + self.graph_paths
                        + self.semantic_examples + self.memory_examples)


def extract_ast_context(source: str) -> dict:
    """Always-on AST parsing (not a RAG layer)."""
    ctx = {"classes": [], "functions": [], "imports": [], "public_symbols": set()}
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return ctx

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            methods = [m.name for m in node.body
                       if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))]
            ctx["classes"].append({"name": node.name, "methods": methods})
            if not node.name.startswith("_"):
                ctx["public_symbols"].add(node.name)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = [a.arg for a in node.args.args]
            ctx["functions"].append({"name": node.name, "args": args})
            if not node.name.startswith("_"):
                ctx["public_symbols"].add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            try:
                ctx["imports"].append(ast.unparse(node))
            except Exception:
                pass
    return ctx


def build_rag_context(source_code: str,
                      file_path: str,
                      config: AblationConfig) -> RAGContext:
    rag = RAGContext()
    if config.enable_layer1_docs:
        rag.docs_snippets = _query_layer1_docs(source_code)
    if config.enable_layer2_graph:
        rag.graph_paths, rag.graph_hop_depth = _query_layer2_graph(file_path, source_code)
    if config.enable_layer2_vector:
        rag.semantic_examples = _query_layer2_vector(source_code)
    if config.enable_rag_memory:
        rag.memory_examples = _query_rag_memory(source_code)
    return rag


# ─── Cached resources (avoid re-loading per file) ────────────────────────────
_DOCS_RETRIEVER = None
_VECTOR_ENCODER = None


def _query_layer1_docs(source_code: str, top_k: int = 3) -> list[str]:
    """FAISS over docs corpus (Django / sklearn / SymPy demo index)."""
    global _DOCS_RETRIEVER
    try:
        if _DOCS_RETRIEVER is None:
            from layers.docs.docs_retriever import create_demo_index
            _DOCS_RETRIEVER = create_demo_index()
        result_obj = _DOCS_RETRIEVER.retrieve(source_code[:500], top_k=top_k)
        # `retrieve` returns a RetrievalResults object with `.results`
        return [r.chunk.content[:300] for r in result_obj.results]
    except Exception as exc:
        logger.warning("Layer 1 docs query failed: %s", exc)
        return []


def _query_layer2_graph(file_path: str, source_code: str, top_k: int = 5) -> tuple[list[str], int]:
    """
    Build an in-memory call graph from the source code being tested,
    then take 2-hop neighborhoods of the top public functions.
    This matches what `PartB/testgen/main.py:build_call_graph` does in production.
    """
    try:
        # Make `from main import …` work (PartB/testgen on sys.path)
        import sys
        from pathlib import Path
        testgen_dir = Path(__file__).resolve().parent.parent
        if str(testgen_dir) not in sys.path:
            sys.path.insert(0, str(testgen_dir))
        from main import build_call_graph, get_2hop_subgraph
    except Exception as exc:
        logger.warning("Layer 2 graph import failed: %s", exc)
        return [], 0

    try:
        G = build_call_graph(source_code)
        if not G.nodes:
            return [], 0

        # Pick public top-level functions / methods as seeds (skip dunders)
        public = [n for n in G.nodes if not n.split(".")[-1].startswith("_")]
        if not public:
            return [], 0
        seeds = sorted(public, key=lambda n: G.nodes[n].get("lineno", 1 << 30))[:top_k]

        out: list[str] = []
        max_hop = 1
        for seed in seeds:
            sub = get_2hop_subgraph(G, seed)
            callees   = sub["callees"][:5]
            callers   = sub["callers"][:3]
            hop2      = sub["callees_of_callees"][:3]
            parts = [f"# {seed}"]
            if callers: parts.append(f"  callers: {', '.join(callers)}")
            if callees: parts.append(f"  callees: {', '.join(callees)}")
            if hop2:
                max_hop = 2
                parts.append(f"  hop-2 callees: {', '.join(hop2)}")
            if len(parts) > 1:
                out.append("\n".join(parts))
        return out, max_hop
    except Exception as exc:
        logger.warning("Layer 2 graph query failed: %s", exc)
        return [], 0


def _query_layer2_vector(source_code: str, top_k: int = 3) -> list[str]:
    """
    Semantic vector RAG over an external corpus of past test examples
    (the `test_examples` table inside testmate_rag.db).

    Strategy:
      1. Embed the input source code with all-MiniLM-L6-v2
      2. Compute cosine sim against every stored test's embedding
      3. Return the top-K matched test snippets (with their method/class context)

    Falls back to intra-file similarity if the DB isn't attached
    (so the layer is never silently empty).
    """
    global _VECTOR_ENCODER
    _bootstrap_rag_db()   # copy uploaded DB into place if available

    # --- 1. External corpus path (preferred — real vector RAG) ----------------
    try:
        import sys
        from pathlib import Path
        testgen_dir = Path(__file__).resolve().parent.parent
        if str(testgen_dir) not in sys.path:
            sys.path.insert(0, str(testgen_dir))
        from rag_store import DB_PATH as _DB_PATH

        import sqlite3, pickle
        db_paths = [
            _DB_PATH,
            str(testgen_dir / "testmate_rag.db"),
            "testmate_rag.db",
        ]
        db_path = next((p for p in db_paths if Path(p).is_file()
                        and Path(p).stat().st_size > 50_000), None)
        if db_path:
            conn = sqlite3.connect(db_path)
            rows = conn.execute(
                "SELECT method_name, class_name, target_signature, test_code, embedding "
                "FROM test_examples WHERE embedding IS NOT NULL"
            ).fetchall()
            conn.close()

            if rows:
                if _VECTOR_ENCODER is None:
                    from sentence_transformers import SentenceTransformer
                    _VECTOR_ENCODER = SentenceTransformer("all-MiniLM-L6-v2")

                import numpy as np
                query_emb = _VECTOR_ENCODER.encode(source_code[:1500],
                                                    convert_to_numpy=True,
                                                    show_progress_bar=False)
                qn = query_emb / (np.linalg.norm(query_emb) + 1e-9)

                scored = []
                for method, klass, sig, test_code, emb_blob in rows:
                    emb = None
                    # Try raw float32 buffer first (how rag_store actually writes it)
                    try:
                        emb = np.frombuffer(emb_blob, dtype=np.float32)
                        if emb.size == 0:
                            emb = None
                    except Exception:
                        emb = None
                    # Fall back to pickle (older rows might be pickled)
                    if emb is None:
                        try:
                            emb = np.asarray(pickle.loads(emb_blob), dtype=np.float32)
                        except Exception:
                            continue
                    try:
                        en = emb / (np.linalg.norm(emb) + 1e-9)
                        sim = float(qn @ en)
                    except Exception:
                        continue
                    scored.append((sim, method, klass, test_code))

                if scored:
                    scored.sort(reverse=True, key=lambda x: x[0])
                    out = []
                    for sim, method, klass, test_code in scored[:top_k]:
                        head = f"# Past test (sim={sim:.2f}) "
                        head += f"for {klass}.{method}" if klass else f"for {method}"
                        out.append(f"{head}\n{(test_code or '')[:400]}")
                    return out
    except Exception as exc:
        logger.warning("Layer 2 vector (DB) query failed: %s", exc)

    # --- 2. Fallback: intra-file similarity (no DB attached) -----------------
    try:
        tree = ast.parse(source_code)
    except SyntaxError:
        return []

    funcs: list[tuple[str, str]] = []
    src_lines = source_code.splitlines()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            start = node.lineno - 1
            end   = node.end_lineno or (start + 1)
            funcs.append((node.name, "\n".join(src_lines[start:end])[:400]))
        elif isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    start = item.lineno - 1
                    end   = item.end_lineno or (start + 1)
                    funcs.append((f"{node.name}.{item.name}",
                                  "\n".join(src_lines[start:end])[:400]))
    if len(funcs) < 2:
        return []
    try:
        if _VECTOR_ENCODER is None:
            from sentence_transformers import SentenceTransformer
            _VECTOR_ENCODER = SentenceTransformer("all-MiniLM-L6-v2")
        import numpy as np
        names    = [f[0] for f in funcs]
        snippets = [f[1] for f in funcs]
        embeddings = _VECTOR_ENCODER.encode(snippets, convert_to_numpy=True,
                                            show_progress_bar=False)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9
        normed = embeddings / norms
        sim = normed @ normed.T
        np.fill_diagonal(sim, -1.0)
        n = len(names)
        triu_idx = sorted([(i, j) for i in range(n) for j in range(i + 1, n)],
                          key=lambda ij: sim[ij[0], ij[1]], reverse=True)
        return [f"# Related-by-semantics: {names[i]} <-> {names[j]} (sim={sim[i, j]:.2f})"
                for i, j in triu_idx[:top_k]]
    except Exception as exc:
        logger.warning("Layer 2 vector (intra-file) query failed: %s", exc)
        return []


# ─── RAG memory DB bootstrap (runs once per process) ────────────────────────
_RAG_DB_READY = False


def _bootstrap_rag_db() -> None:
    """
    Copy the populated testmate_rag.db into the working tree if a Kaggle
    dataset is attached. Tries multiple known account paths. Silently
    no-ops if none are found (memory layer just returns empty results).
    """
    global _RAG_DB_READY
    if _RAG_DB_READY:
        return
    _RAG_DB_READY = True   # set early so we only try once

    import os, shutil
    from pathlib import Path

    candidate_dbs = [
        # ↓ add the path for any other Kaggle account here
        "/kaggle/input/datasets/mohammedmedhat08/testmate-rag-db/testmate_rag.db",
        "/kaggle/input/datasets/mohammed8medhat/testmate-rag-db/testmate_rag.db",
        "/kaggle/input/testmate-rag-db/testmate_rag.db",  # plain dataset name fallback
    ]
    src = next((p for p in candidate_dbs if os.path.isfile(p)), None)
    if not src:
        logger.info("RAG memory DB: no uploaded DB found, using fresh empty DB")
        return

    # Copy into both locations rag_store.py might use: the testgen folder
    # (relative to module) and the CWD (since DB_PATH = "testmate_rag.db").
    testgen_dir = Path(__file__).resolve().parent.parent
    targets = [
        testgen_dir / "testmate_rag.db",
        Path.cwd()  / "testmate_rag.db",
    ]
    for dst in targets:
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists() or dst.stat().st_size < 50_000:
                shutil.copy2(src, dst)
                logger.info("RAG memory DB: copied %s -> %s (%.0f KB)",
                            src, dst, dst.stat().st_size / 1024)
        except Exception as exc:
            logger.warning("RAG memory DB copy failed (%s): %s", dst, exc)


def _query_rag_memory(source_code: str, top_k: int = 3) -> list[str]:
    """
    Retrieve good past test examples from the persistent SQLite store.

    On Kaggle, if a `testmate-rag-db` dataset is attached, the populated
    DB is copied into place at first call. Otherwise this returns an
    empty list (memory layer effectively off).
    """
    _bootstrap_rag_db()
    try:
        import sys
        from pathlib import Path
        testgen_dir = Path(__file__).resolve().parent.parent
        if str(testgen_dir) not in sys.path:
            sys.path.insert(0, str(testgen_dir))
        from rag_store import retrieve_similar, init_db
        init_db()
    except Exception as exc:
        logger.warning("RAG memory import failed: %s", exc)
        return []

    # Use first public top-level def as a probe identifier
    try:
        tree = ast.parse(source_code)
        method_name = ""
        class_name = ""
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                    and not node.name.startswith("_"):
                method_name = node.name
                break
            if isinstance(node, ast.ClassDef):
                class_name = node.name
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                            and not item.name.startswith("_"):
                        method_name = item.name
                        break
                if method_name:
                    break
        if not method_name:
            return []
        rows = retrieve_similar(method_name, class_name, top_k=top_k)
        return [f"# Past test for {r.get('method_name', '?')}:\n{r.get('test_code', '')[:300]}"
                for r in rows if isinstance(r, dict)]
    except Exception as exc:
        logger.warning("RAG memory query failed: %s", exc)
        return []


# ──────────────────────────────────────────────────────────────────────────
# 5. PROMPT BUILDER
# ──────────────────────────────────────────────────────────────────────────

PROMPT_TEMPLATE = """You are a senior Python test engineer. Generate a complete pytest test file for the SOURCE code below.

Requirements:
- Use pytest only. No unittest.
- Test the public API: every public function/class method should have at least one test.
- Include positive cases, edge cases, and error cases.
- Use fixtures where appropriate.
- The test must be runnable as-is. Output ONLY the Python code (no markdown fences).

{ast_summary}{rag_section}{plan_section}
SOURCE FILE: {filename}
```python
{source_code}
```

Generate the complete test file now:"""

PLAN_PROMPT_TEMPLATE = """You are a senior Python test engineer. Read this source file and generate a brief test plan (5-7 bullet points) describing what to test.

Format: numbered bullets, one line each. Focus on the public API. Don't write code yet.

SOURCE FILE: {filename}
```python
{source_code}
```

Test plan:"""


def build_prompt(file_info: dict, ast_ctx: dict, rag: RAGContext,
                 plan: Optional[str] = None) -> str:
    ast_summary = ""
    if ast_ctx.get("classes") or ast_ctx.get("functions"):
        parts = []
        if ast_ctx["classes"]:
            cls = ", ".join(f"{c['name']} ({len(c['methods'])} methods)" for c in ast_ctx["classes"][:5])
            parts.append(f"Classes: {cls}")
        if ast_ctx["functions"]:
            fns = ", ".join(f["name"] for f in ast_ctx["functions"][:8])
            parts.append(f"Top-level functions: {fns}")
        ast_summary = "STRUCTURE:\n" + "\n".join(parts) + "\n\n"

    rag_section = ""
    if not rag.is_empty():
        bits = []
        if rag.docs_snippets:
            bits.append("RELEVANT DOCS:\n" + "\n---\n".join(rag.docs_snippets[:3]))
        if rag.graph_paths:
            bits.append("KNOWLEDGE GRAPH PATHS:\n" + "\n".join(rag.graph_paths[:5]))
        if rag.semantic_examples:
            bits.append("SIMILAR CODE PATTERNS:\n" + "\n---\n".join(rag.semantic_examples[:3]))
        if rag.memory_examples:
            bits.append("PAST EXAMPLES:\n" + "\n---\n".join(rag.memory_examples[:2]))
        rag_section = "RAG CONTEXT:\n" + "\n\n".join(bits) + "\n\n"

    plan_section = f"TEST PLAN (follow this strictly):\n{plan.strip()}\n\n" if plan else ""

    module_name = file_info.get("module_id", file_info["id"]) + ".py"
    return PROMPT_TEMPLATE.format(
        ast_summary=ast_summary,
        rag_section=rag_section,
        plan_section=plan_section,
        filename=module_name,
        source_code=file_info["source_code"][:6000],
    )


# ──────────────────────────────────────────────────────────────────────────
# 6. GENERATION
# ──────────────────────────────────────────────────────────────────────────

def generate_plan(model, tokenizer, file_info: dict,
                  max_new_tokens: int = 400, temperature: float = 0.3) -> tuple[str, int, int]:
    """One-shot plan generation. Returns (plan_text, prompt_tokens, completion_tokens)."""
    import torch
    module_name = file_info.get("module_id", file_info["id"]) + ".py"
    prompt = PLAN_PROMPT_TEMPLATE.format(
        filename=module_name,
        source_code=file_info["source_code"][:6000],
    )
    messages = [
        {"role": "system", "content": "You write concise, structured Python test plans."},
        {"role": "user",   "content": prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=8192).to(model.device)
    pt = inputs["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens, temperature=temperature,
            do_sample=temperature > 0, top_p=0.9, pad_token_id=tokenizer.eos_token_id,
        )
    completion = out[0][pt:]
    ct = completion.shape[0]
    return tokenizer.decode(completion, skip_special_tokens=True).strip(), pt, ct


def generate_single_shot(model, tokenizer, prompt: str,
                         max_new_tokens: int = 1024,
                         temperature: float = 0.3) -> tuple[str, int, int]:
    import torch
    messages = [
        {"role": "system", "content": "You write production-quality pytest test code."},
        {"role": "user",   "content": prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=8192).to(model.device)
    pt = inputs["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens, temperature=temperature,
            do_sample=temperature > 0, top_p=0.9, pad_token_id=tokenizer.eos_token_id,
        )
    completion = out[0][pt:]
    ct = completion.shape[0]
    text_out = tokenizer.decode(completion, skip_special_tokens=True).strip()
    return _strip_markdown_fences(text_out), pt, ct


_RETRY_TEMPS = [0.3, 0.5, 0.7]  # temperatures for attempt 0, 1, 2+

_ERROR_HINTS = {
    "import": (
        "The previous attempt failed with an ImportError. "
        "Make sure all imports are correct. Do NOT use relative imports (`from .x import y`). "
        "Import only from the standard library or from the module shown in SOURCE FILE."
    ),
    "syntax": (
        "The previous attempt failed with a SyntaxError. "
        "Output only valid Python. No markdown fences. No prose outside of code."
    ),
    "assertion": (
        "The previous attempt failed with an AssertionError. "
        "Check the expected values — they may be wrong. Add a try/except or relax assertions."
    ),
    "runtime": (
        "The previous attempt raised a runtime exception. "
        "Add try/except blocks or fix the incorrect assumptions about the API."
    ),
}


def generate_with_loop(model, tokenizer, file_info: dict, prompt: str,
                       max_retries: int = 3) -> tuple[str, int, int, int, str]:
    """
    Loop: generate → validate → run pytest → if fails, regenerate with classified feedback.
    Returns (final_test_code, total_pt, total_ct, iterations, first_try_test_code).
    The first_try_test_code is preserved so we can evaluate Pass@1 separately.
    """
    total_pt, total_ct = 0, 0
    temperature = _RETRY_TEMPS[0]
    test_code, pt, ct = generate_single_shot(model, tokenizer, prompt, temperature=temperature)
    total_pt += pt; total_ct += ct
    first_try_test_code = test_code   # snapshot for Pass@1

    for i in range(max_retries):
        # Reject output that contains no pytest functions at all
        if not re.search(r"\bdef test_\w+", test_code):
            feedback = "NO_TESTS: the output contained no `def test_*` functions. Output only runnable pytest code."
            error_type = "syntax"
        else:
            feedback = _run_test_get_error(test_code, file_info)
            if feedback is None:
                return test_code, total_pt, total_ct, i + 1, first_try_test_code
            error_type = _classify_error(feedback)

        hint = _ERROR_HINTS.get(error_type, "")
        temperature = _RETRY_TEMPS[min(i + 1, len(_RETRY_TEMPS) - 1)]
        correction_prompt = (
            prompt
            + f"\n\n## PREVIOUS ATTEMPT FAILED ({error_type.upper()}) ##\n"
            + (f"{hint}\n\n" if hint else "")
            + f"Generated test:\n```python\n{test_code}\n```\n\n"
            + f"Error:\n{feedback[:1500]}\n\nFix the test (temperature will be higher this round):"
        )
        test_code, pt, ct = generate_single_shot(
            model, tokenizer, correction_prompt, temperature=temperature
        )
        total_pt += pt; total_ct += ct

    return test_code, total_pt, total_ct, max_retries + 1, first_try_test_code


def _strip_markdown_fences(text: str) -> str:
    m = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    return m.group(1).strip() if m else text.strip()


def _classify_error(feedback: str) -> str:
    """Classify pytest error output into: import | syntax | assertion | runtime | unknown."""
    fb = feedback.lower()
    if "importerror" in fb or "modulenotfounderror" in fb or "attempted relative import" in fb:
        return "import"
    if "syntaxerror" in fb or "indentationerror" in fb:
        return "syntax"
    if "assertionerror" in fb or "assert " in fb:
        return "assertion"
    return "runtime"


def _run_test_get_error(test_code: str, file_info: dict) -> Optional[str]:
    mid = file_info.get("module_id", file_info["id"])
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        src = tmp / f"{mid}.py"
        src.write_text(file_info["source_code"], encoding="utf-8")
        tst = tmp / f"test_{mid}.py"
        tst.write_text(test_code, encoding="utf-8")
        # Inject conftest.py so relative-import-style source files can be found
        (tmp / "conftest.py").write_text(
            "import sys, pathlib\nsys.path.insert(0, str(pathlib.Path(__file__).parent))\n"
        )
        try:
            r = subprocess.run(
                ["python", "-m", "pytest", str(tst), "-x", "--tb=short", "-q"],
                cwd=tmp, capture_output=True, text=True, timeout=60,
            )
            if r.returncode == 0:
                return None
            return (r.stdout + "\n" + r.stderr)[-2000:]
        except subprocess.TimeoutExpired:
            return "TIMEOUT (60s exceeded)"
        except Exception as exc:
            return f"EXEC ERROR: {exc}"


# ──────────────────────────────────────────────────────────────────────────
# 7. METRIC HELPERS
# ──────────────────────────────────────────────────────────────────────────

_IDENT_RX = re.compile(r"\b[a-zA-Z_][a-zA-Z0-9_]{2,}\b")


def measure_api_coverage(test_code: str, public_symbols: set) -> float:
    """% of public symbols from source that appear (as Name or Attribute) in test."""
    if not public_symbols:
        return 0.0
    try:
        t = ast.parse(test_code)
    except SyntaxError:
        return 0.0
    refs = set()
    for n in ast.walk(t):
        if isinstance(n, ast.Name):
            refs.add(n.id)
        elif isinstance(n, ast.Attribute):
            refs.add(n.attr)
    return round(len(public_symbols & refs) / len(public_symbols), 3)


def measure_graphrag_hit(rag: RAGContext, test_code: str) -> bool:
    if not rag.graph_paths:
        return False
    g_ids = set(_IDENT_RX.findall(" ".join(rag.graph_paths)))
    t_ids = set(_IDENT_RX.findall(test_code))
    return bool(g_ids & t_ids)


def measure_vectorrag_hit(rag: RAGContext, test_code: str) -> bool:
    sources = rag.docs_snippets + rag.semantic_examples + rag.memory_examples
    if not sources:
        return False
    v_ids = set(_IDENT_RX.findall(" ".join(sources)))
    t_ids = set(_IDENT_RX.findall(test_code))
    return bool(v_ids & t_ids)


def measure_rag_utilization(rag: RAGContext, test_code: str) -> float:
    if rag.is_empty():
        return 0.0
    r_ids = set(_IDENT_RX.findall(rag.all_text()))
    if not r_ids:
        return 0.0
    t_ids = set(_IDENT_RX.findall(test_code))
    return round(len(r_ids & t_ids) / len(r_ids), 3)


def measure_retrieval_coverage(rag: RAGContext, public_symbols: set) -> float:
    """% of source's public API that appears in retrieved RAG context."""
    if not public_symbols:
        return 0.0
    if rag.is_empty():
        return 0.0
    r_ids = set(_IDENT_RX.findall(rag.all_text()))
    return round(len(public_symbols & r_ids) / len(public_symbols), 3)


def measure_test_diversity(test_code: str) -> int:
    """Count of distinct test categories by heuristic naming."""
    categories = set()
    keywords = {
        "positive": ["valid", "success", "happy", "normal", "default", "basic"],
        "negative": ["invalid", "error", "fail", "exception", "raise"],
        "edge":     ["empty", "none", "null", "zero", "max", "min", "boundary", "edge"],
        "boundary": ["large", "small", "overflow", "underflow", "limit"],
    }
    lower = test_code.lower()
    for cat, kws in keywords.items():
        if any(kw in lower for kw in kws):
            categories.add(cat)
    return len(categories)


def measure_assertions_per_test(test_code: str) -> float:
    try:
        t = ast.parse(test_code)
    except SyntaxError:
        return 0.0
    tests = [n for n in ast.walk(t)
             if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
             and n.name.startswith("test_")]
    if not tests:
        return 0.0
    total_asserts = 0
    for fn in tests:
        for node in ast.walk(fn):
            if isinstance(node, ast.Assert):
                total_asserts += 1
            elif isinstance(node, ast.Call):
                # pytest helper calls like pytest.raises, assertEqual, etc.
                func_str = ""
                try:
                    func_str = ast.unparse(node.func)
                except Exception:
                    pass
                if "pytest.raises" in func_str or "assertEqual" in func_str:
                    total_asserts += 1
    return round(total_asserts / len(tests), 2)


def measure_setup_quality(test_code: str) -> bool:
    """True if test uses fixtures, parametrize, or pytest.raises."""
    markers = ["@pytest.fixture", "@pytest.mark.parametrize", "pytest.raises"]
    return any(m in test_code for m in markers)


def split_hallucinations(test_code: str,
                         public_symbols: set,
                         rag: RAGContext) -> tuple[int, int]:
    """
    Returns (model_hallucinations, rag_hallucinations).
      model: refs in test ∉ source ∧ ∉ RAG  → pure model invention
      rag:   refs in test ∈ RAG  ∧ ∉ source → RAG misled the model
    """
    try:
        t = ast.parse(test_code)
    except SyntaxError:
        return 0, 0

    rag_ids = set(_IDENT_RX.findall(rag.all_text())) if not rag.is_empty() else set()
    model_h = 0
    rag_h = 0
    for node in ast.walk(t):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            attr = node.attr
            if (attr not in public_symbols
                and not attr.startswith("_")
                and attr.isidentifier()
                and len(attr) > 3):
                if attr in rag_ids:
                    rag_h += 1
                else:
                    model_h += 1
    return model_h, rag_h


def measure_plan_adherence(plan_text: str, test_code: str) -> float:
    """% of plan bullet points referenced in test code (by keyword overlap)."""
    if not plan_text.strip():
        return 0.0
    bullets = [b.strip() for b in re.split(r"\n[\s]*\d+\.", "\n" + plan_text) if b.strip()]
    if not bullets:
        bullets = [b for b in plan_text.split("\n") if b.strip()]
    if not bullets:
        return 0.0
    test_lower = test_code.lower()
    hits = 0
    for b in bullets:
        # take 2 most informative words from the bullet
        words = [w.lower() for w in re.findall(r"[a-zA-Z][a-zA-Z_]{3,}", b)]
        words = [w for w in words if w not in {"test", "verify", "check", "ensure",
                                                "should", "when", "with", "from", "that"}]
        if not words:
            continue
        # bullet considered "covered" if any 2 of its keywords appear in the test
        matches = sum(1 for w in words[:5] if w in test_lower)
        if matches >= 2 or (len(words) <= 2 and matches >= 1):
            hits += 1
    return round(hits / len(bullets), 3)


# ──────────────────────────────────────────────────────────────────────────
# 8. EVALUATION
# ──────────────────────────────────────────────────────────────────────────

def evaluate_test(test_code: str,
                  file_info: dict,
                  rag: Optional[RAGContext] = None,
                  ast_ctx: Optional[dict] = None,
                  plan_text: Optional[str] = None) -> dict:
    """Evaluate a generated test. Adds all per-file metrics (except mutation)."""
    rag = rag or RAGContext()
    ast_ctx = ast_ctx or extract_ast_context(file_info["source_code"])
    public_symbols = ast_ctx.get("public_symbols", set())

    metrics = {
        "syntax_valid":          False,
        "imports_resolve":       False,
        "tests_runnable":        False,
        "tests_collected":       0,
        "tests_passed":          0,
        "pass_rate":             0.0,
        "line_coverage":         0.0,
        # generation quality
        "api_coverage":          0.0,
        "test_diversity":        0,
        "assertions_per_test":   0.0,
        "setup_quality":         False,
        # RAG eval
        "graphrag_hit":          False,
        "vectorrag_hit":         False,
        "rag_context_size":      rag.total_chars(),
        "rag_utilization_ratio": 0.0,
        "graph_hop_depth":       rag.graph_hop_depth,
        "retrieval_coverage":    0.0,
        # hallucinations
        "model_hallucination":   0,
        "rag_hallucination":     0,
        # plan (only meaningful if plan_text provided)
        "plan_adherence":        None if plan_text is None else 0.0,
    }

    # 1. Syntax
    try:
        ast.parse(test_code)
        metrics["syntax_valid"] = True
    except SyntaxError:
        return metrics

    # 2. Generation-quality / RAG metrics (don't need execution)
    metrics["api_coverage"]          = measure_api_coverage(test_code, public_symbols)
    metrics["test_diversity"]        = measure_test_diversity(test_code)
    metrics["assertions_per_test"]   = measure_assertions_per_test(test_code)
    metrics["setup_quality"]         = measure_setup_quality(test_code)
    metrics["graphrag_hit"]          = measure_graphrag_hit(rag, test_code)
    metrics["vectorrag_hit"]         = measure_vectorrag_hit(rag, test_code)
    metrics["rag_utilization_ratio"] = measure_rag_utilization(rag, test_code)
    metrics["retrieval_coverage"]    = measure_retrieval_coverage(rag, public_symbols)
    m_hall, r_hall                   = split_hallucinations(test_code, public_symbols, rag)
    metrics["model_hallucination"]   = m_hall
    metrics["rag_hallucination"]     = r_hall
    if plan_text is not None:
        metrics["plan_adherence"]    = measure_plan_adherence(plan_text, test_code)

    # 3. Execution + coverage
    mid = file_info.get("module_id", file_info["id"])
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        src = tmp / f"{mid}.py"
        src.write_text(file_info["source_code"], encoding="utf-8")
        tst = tmp / f"test_{mid}.py"
        tst.write_text(test_code, encoding="utf-8")
        # Conftest ensures tempdir is on sys.path so relative-import-style source can be found
        (tmp / "conftest.py").write_text(
            "import sys, pathlib\nsys.path.insert(0, str(pathlib.Path(__file__).parent))\n"
        )

        # 3a. Collection
        try:
            r = subprocess.run(
                ["python", "-m", "pytest", str(tst), "--collect-only", "-q"],
                cwd=tmp, capture_output=True, text=True, timeout=30,
            )
            metrics["imports_resolve"] = (r.returncode == 0)
            metrics["tests_collected"] = sum(1 for l in r.stdout.splitlines() if "::test_" in l)
        except Exception:
            return metrics

        if metrics["tests_collected"] == 0:
            return metrics

        metrics["tests_runnable"] = True

        # 3b. Run with coverage — use absolute path to SUT so --cov finds the module
        try:
            r = subprocess.run(
                ["python", "-m", "pytest", str(tst),
                 f"--cov={str(src)}", "--cov-report=json",
                 "--tb=no", "-q"],
                cwd=tmp, capture_output=True, text=True, timeout=120,
            )
            m_pass = re.search(r"(\d+) passed", r.stdout)
            m_fail = re.search(r"(\d+) failed", r.stdout)
            passed = int(m_pass.group(1)) if m_pass else 0
            failed = int(m_fail.group(1)) if m_fail else 0
            total = passed + failed
            metrics["tests_passed"] = passed
            metrics["pass_rate"]    = round(passed / total, 3) if total > 0 else 0.0

            cov_file = tmp / "coverage.json"
            if cov_file.exists():
                cov_data = json.loads(cov_file.read_text())
                for fname, fdata in cov_data.get("files", {}).items():
                    if mid in fname or file_info["filename"] in fname:
                        metrics["line_coverage"] = round(fdata["summary"]["percent_covered"], 2)
                        break
        except subprocess.TimeoutExpired:
            pass
        except Exception:
            pass

    return metrics


# ──────────────────────────────────────────────────────────────────────────
# 9. MAIN: PHASE 1 (generation)
# ──────────────────────────────────────────────────────────────────────────

def _run_production_autonomous_loop(
    model, tokenizer, file_info: dict, config: AblationConfig
) -> tuple[str, int, int, int]:
    """
    Delegate generation to the real autonomous_loop() from production main.py.
    This is used exclusively for the 'testmate' ablation variant so that the
    ablation actually measures the full stack (BES gate, quality_gates,
    docstring amplifier, pre-pass triage) rather than the harness's stripped-
    down generate_with_loop.

    Returns (test_code, prompt_tokens, completion_tokens, iterations).
    Token counts are 0 because autonomous_loop does not expose them.
    """
    import sys
    # Ensure PartB/testgen is on sys.path so we can import main.py
    _testgen_dir = str(Path(__file__).parent.parent)
    if _testgen_dir not in sys.path:
        sys.path.insert(0, _testgen_dir)

    from main import autonomous_loop  # type: ignore[import]

    mid = file_info.get("module_id", file_info["id"])
    stats: dict = {}
    with tempfile.TemporaryDirectory() as _tmp:
        tmp = Path(_tmp)
        src = tmp / f"{mid}.py"
        src.write_text(file_info["source_code"], encoding="utf-8")

        autonomous_loop(
            model,
            tokenizer,
            str(src),
            max_retries=config.max_retries,
            plan_mode=config.enable_plan_mode,
            stats_out=stats,
            skip_bes_gate=getattr(config, "skip_bes_gate", False),
        )

        # autonomous_loop writes test_{stem}_testmate.py in the same dir as src
        test_path = tmp / f"test_{mid}_testmate.py"
        if not test_path.exists():
            test_path = tmp / f"test_{mid}.py"   # legacy fallback
        test_code = test_path.read_text(encoding="utf-8") if test_path.exists() else ""

    iterations = stats.get("iterations", config.max_retries)
    return test_code, 0, 0, iterations


def _save_checkpoint(per_file: list, config: AblationConfig, output_path: str) -> None:
    """Write a partial JSON so a crash doesn't lose progress."""
    out = Path(output_path)
    partial = out.with_suffix(".partial.json")
    partial.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "variant":  config.variant,
        "model":    DEFAULT_MODEL_ID,
        "config":   asdict(config),
        "partial":  True,
        "per_file": per_file,
    }
    partial.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_ablation(config: AblationConfig, output_path: str) -> dict:
    """Phase 1: generate + evaluate all files (no mutation)."""
    print("=" * 70)
    print(f"  ABLATION: {config.variant}")
    print(f"  Layer1 docs:    {'ON' if config.enable_layer1_docs else 'OFF'}")
    print(f"  Layer2 graph:   {'ON' if config.enable_layer2_graph else 'OFF'}")
    print(f"  Layer2 vector:  {'ON' if config.enable_layer2_vector else 'OFF'}")
    print(f"  RAG memory:     {'ON' if config.enable_rag_memory else 'OFF'}")
    print(f"  Plan mode:      {'ON' if config.enable_plan_mode else 'OFF'}")
    print(f"  Loop:           {'ON' if config.enable_self_correction_loop else 'OFF'}")
    print(f"  LoRA:           {'ON ('+config.lora_path+')' if config.enable_lora else 'OFF'}")
    print(f"  Sample size:    {config.sample_size if config.sample_size else 'ALL'}")
    print("=" * 70)

    model, tokenizer = load_qwen_base_4bit()
    if config.enable_lora:
        if not config.lora_path:
            raise ValueError("enable_lora=True but lora_path is empty")
        model = attach_lora(model, config.lora_path)
    dataset = load_test_dataset(sample_size=config.sample_size)

    per_file = []
    t_total = time.perf_counter()

    for i, file_info in enumerate(dataset, 1):
        print(f"\n[{i}/{len(dataset)}] {file_info['filename']} ({file_info['lines']} lines)")
        t0 = time.perf_counter()

        try:
            ast_ctx = extract_ast_context(file_info["source_code"])
            rag     = build_rag_context(file_info["source_code"], file_info["abs_path"], config)

            # Optional plan generation (Call #1)
            plan_text = None
            plan_pt = plan_ct = 0
            if config.enable_plan_mode:
                plan_text, plan_pt, plan_ct = generate_plan(model, tokenizer, file_info)

            # Build prompt for test generation
            prompt = build_prompt(file_info, ast_ctx, rag, plan=plan_text)

            # Generate
            first_try_test = None
            if config.variant == "testmate":
                # Use the real production pipeline (BES gate, quality gates,
                # docstring amplifier, pre-pass triage) so this cell honestly
                # measures the full TestMate stack.
                test_code, pt, ct, iterations = _run_production_autonomous_loop(
                    model, tokenizer, file_info, config
                )
                first_try_test = test_code  # autonomous_loop already self-corrected
            elif config.enable_self_correction_loop:
                test_code, pt, ct, iterations, first_try_test = generate_with_loop(
                    model, tokenizer, file_info, prompt, config.max_retries,
                )
            else:
                test_code, pt, ct = generate_single_shot(
                    model, tokenizer, prompt,
                    max_new_tokens=config.max_new_tokens, temperature=config.temperature,
                )
                iterations = 1
                first_try_test = test_code

            # Evaluate final
            metrics = evaluate_test(test_code, file_info, rag=rag, ast_ctx=ast_ctx, plan_text=plan_text)

            # Evaluate first-try for Pass@1
            first_metrics = (
                evaluate_test(first_try_test, file_info, rag=rag, ast_ctx=ast_ctx, plan_text=plan_text)
                if config.enable_self_correction_loop
                else metrics
            )
            any_pass_1 = first_metrics["tests_passed"] > 0
            all_pass_1 = (first_metrics["tests_collected"] > 0 and
                          first_metrics["tests_passed"] == first_metrics["tests_collected"])

            wall = time.perf_counter() - t0

            entry = {
                "file":              file_info["filename"],
                "id":                file_info["id"],
                "lines":             file_info["lines"],
                "iterations":        iterations,
                "wall_time_sec":     round(wall, 2),
                "prompt_tokens":     pt + plan_pt,
                "completion_tokens": ct + plan_ct,
                "any_pass_1":        any_pass_1,
                "all_pass_1":        all_pass_1,
                "plan_text":         plan_text if plan_text is not None else "",
                "test_code":         test_code,          # full code (needed for Phase 2 mutation)
                "test_code_preview": test_code[:200],   # legacy: kept for backwards-compat readers
                **metrics,
            }
            per_file.append(entry)
            print(f"   ✓ syntax={metrics['syntax_valid']} pass_rate={metrics['pass_rate']:.0%}"
                  f" cov={metrics['line_coverage']:.1f}% api_cov={metrics['api_coverage']:.0%}"
                  f" iter={iterations} ({wall:.1f}s)")

        except Exception as exc:
            logger.exception("File failed")
            per_file.append({
                "file": file_info["filename"], "error": str(exc)[:300],
                "wall_time_sec": round(time.perf_counter() - t0, 2),
            })
            print(f"   ❌ {exc}")

        # Flush GPU between files
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

        # Checkpoint every 10 files
        if i % 10 == 0:
            _save_checkpoint(per_file, config, output_path)
            elapsed_min = (time.perf_counter() - t_total) / 60
            avg_min = elapsed_min / i
            remaining = avg_min * (len(dataset) - i)
            print(f"   💾 Checkpoint @ {i}/{len(dataset)}"
                  f" — elapsed {elapsed_min:.1f}m · ETA {remaining:.1f}m")

    # Summary
    valid = [e for e in per_file if "error" not in e]
    n_valid = max(len(valid), 1)
    summary = {
        "total_files":              len(per_file),
        "files_succeeded":          sum(1 for e in valid if e.get("syntax_valid")),
        "mean_pass_rate":           _mean(e.get("pass_rate", 0) for e in valid),
        "mean_line_coverage":       _mean(e.get("line_coverage", 0) for e in valid),
        "mean_api_coverage":        _mean(e.get("api_coverage", 0) for e in valid),
        "any_pass_1_rate":          round(sum(1 for e in valid if e.get("any_pass_1")) / n_valid, 3),
        "all_pass_1_rate":          round(sum(1 for e in valid if e.get("all_pass_1")) / n_valid, 3),
        "graphrag_hit_rate":        round(sum(1 for e in valid if e.get("graphrag_hit")) / n_valid, 3),
        "vectorrag_hit_rate":       round(sum(1 for e in valid if e.get("vectorrag_hit")) / n_valid, 3),
        "mean_rag_utilization":     _mean(e.get("rag_utilization_ratio", 0) for e in valid),
        "mean_retrieval_coverage":  _mean(e.get("retrieval_coverage", 0) for e in valid),
        "total_model_hallucinations": sum(e.get("model_hallucination", 0) for e in valid),
        "total_rag_hallucinations":   sum(e.get("rag_hallucination", 0) for e in valid),
        "mean_test_diversity":      _mean(e.get("test_diversity", 0) for e in valid),
        "mean_assertions_per_test": _mean(e.get("assertions_per_test", 0) for e in valid),
        "mean_iterations":          _mean(e.get("iterations", 0) for e in valid),
        "mean_wall_time":           _mean(e.get("wall_time_sec", 0) for e in valid),
        "total_wall_time":          round(time.perf_counter() - t_total, 1),
    }
    if config.enable_plan_mode:
        plan_scores = [e.get("plan_adherence") for e in valid if e.get("plan_adherence") is not None]
        summary["mean_plan_adherence"] = round(sum(plan_scores)/len(plan_scores), 3) if plan_scores else 0.0

    output = {
        "variant":      config.variant,
        "model":        DEFAULT_MODEL_ID,
        "quantization": "nf4-4bit",
        "config":       asdict(config),
        "per_file":     per_file,
        "summary":      summary,
    }
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print("\n" + "=" * 70)
    print(f"  PHASE 1 RESULTS — {config.variant}")
    print("=" * 70)
    for k, v in summary.items():
        print(f"  {k:30s} {v}")
    print(f"\n  Saved → {out}")
    return output


# ──────────────────────────────────────────────────────────────────────────
# 10. MAIN: PHASE 2 (mutation testing)
# ──────────────────────────────────────────────────────────────────────────

def run_mutation_pass(input_json: str, output_json: str) -> dict:
    """
    Phase 2: read Phase 1 results, run mutmut on each generated test,
    add `mutation_score` per file + `mean_mutation_score` to summary.
    """
    in_path = Path(input_json)
    out_path = Path(output_json)
    if not in_path.exists():
        raise FileNotFoundError(f"Phase 1 results not found: {in_path}")

    data = json.loads(in_path.read_text(encoding="utf-8"))
    per_file = data.get("per_file", [])

    print("=" * 70)
    print(f"  PHASE 2: Mutation testing")
    print(f"  Reading: {in_path}")
    print(f"  Files:   {len(per_file)}")
    print("=" * 70)

    t_total = time.perf_counter()
    for i, entry in enumerate(per_file, 1):
        if "error" in entry or not entry.get("syntax_valid"):
            entry["mutation_score"] = 0.0
            continue
        print(f"\n[{i}/{len(per_file)}] {entry['file']}")
        t0 = time.perf_counter()
        score = _run_mutmut_on_pair(entry)
        entry["mutation_score"] = score
        print(f"   mutation_score = {score:.1f}%  ({time.perf_counter()-t0:.1f}s)")

        # Checkpoint every 10 files
        if i % 10 == 0:
            _save_phase2_checkpoint(data, out_path)
            elapsed = (time.perf_counter() - t_total) / 60
            print(f"   💾 Phase 2 checkpoint @ {i}/{len(per_file)} — elapsed {elapsed:.1f}m")

    # Update summary
    valid = [e for e in per_file if "error" not in e and e.get("syntax_valid")]
    scores = [e.get("mutation_score", 0) for e in valid]
    data.setdefault("summary", {})["mean_mutation_score"] = (
        round(sum(scores) / len(scores), 3) if scores else 0.0
    )

    out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"\n✅ Phase 2 complete — saved → {out_path}")
    print(f"   mean_mutation_score = {data['summary']['mean_mutation_score']:.2f}%")
    return data


def _save_phase2_checkpoint(data: dict, out_path: Path) -> None:
    partial = out_path.with_suffix(".partial.json")
    partial.parent.mkdir(parents=True, exist_ok=True)
    partial.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _run_mutmut_on_pair(entry: dict) -> float:
    """
    Run mutation testing on source file + generated test. Returns % killed.

    Uses the production `run_mutation_testing()` from main.py (which works on
    Windows; the public `mutmut` package does not). The production helper applies
    a fixed set of mutation operators, runs pytest against each mutant, and
    returns a feedback string that includes the kill percentage.
    """
    full_test = entry.get("test_code") or entry.get("test_code_preview") or ""
    if not full_test:
        return 0.0

    # Re-read the source from the original eval dir
    src_name = entry.get("file")
    if not src_name:
        return 0.0
    src_disk_path = _EVAL_DIR / src_name
    if not src_disk_path.exists():
        return 0.0
    source_code = src_disk_path.read_text(encoding="utf-8", errors="ignore")

    # Compute module_id (sanitised stem) — must match how Phase 1 named files
    raw_stem = Path(src_name).stem
    mid = re.sub(r'^(\d)', r'm\1', raw_stem)

    # Ensure PartB/testgen is importable so we can call the production helper
    import sys
    _testgen_dir = str(Path(__file__).parent.parent)
    if _testgen_dir not in sys.path:
        sys.path.insert(0, _testgen_dir)
    try:
        from main import run_mutation_testing as _production_mutation_test  # type: ignore[import]
    except Exception:
        return 0.0

    with tempfile.TemporaryDirectory() as _tmp:
        tmp = Path(_tmp)
        src_path = tmp / f"{mid}.py"
        tst_path = tmp / f"test_{mid}.py"
        src_path.write_text(source_code, encoding="utf-8")
        tst_path.write_text(full_test, encoding="utf-8")
        # Same conftest as Phase 1 so the test can import the source
        (tmp / "conftest.py").write_text(
            "import sys, pathlib\nsys.path.insert(0, str(pathlib.Path(__file__).parent))\n"
        )

        try:
            all_killed, feedback = _production_mutation_test(str(src_path), str(tst_path))
        except Exception as exc:
            print(f"   ⚠️  Mutation run failed: {exc}")
            return 0.0

        if all_killed:
            return 100.0
        # Feedback string looks like: "Mutation score: 60% (2 survived: ...)"
        m = re.search(r"(\d+(?:\.\d+)?)\s*%", feedback or "")
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                return 0.0
        return 0.0


# ──────────────────────────────────────────────────────────────────────────
# 11. UTILITIES
# ──────────────────────────────────────────────────────────────────────────

def _mean(values) -> float:
    vs = [v for v in values if v is not None]
    return round(sum(vs) / len(vs), 3) if vs else 0.0
