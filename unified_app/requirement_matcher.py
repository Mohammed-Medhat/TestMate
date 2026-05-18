"""
Hybrid requirement matcher — pairs a code file with the most relevant
requirements from Part A's output (SRS labeled requirements + README scenarios).

Two stages:
  1. KEYWORD filter (fast, runs always):
       - extract function/class names from the file's AST (high-signal terms)
       - extract all words from the file's source
       - score each requirement by token overlap (AST identifiers weighted 3x)
       - keep top 30 candidates

  2. SEMANTIC rerank (slow, only if >top_k candidates remain):
       - lazy-load sentence-transformers (all-MiniLM-L6-v2, ~80MB)
       - embed file source + each candidate requirement
       - cosine-rank → top K

Fallback: if sentence-transformers is not installed or fails, returns
keyword-only top K.

Usage:
    matched = match_requirements_to_file(file_source, requirements, top_k=10)
"""
from __future__ import annotations

import ast
import re
import logging
from typing import Any

logger = logging.getLogger(__name__)

# ── Tokenization & AST helpers ─────────────────────────────────────────────

# Common Python keywords / stopwords that add noise, filtered out
_NOISE = {
    "the", "and", "for", "with", "from", "this", "that", "shall", "must",
    "should", "will", "can", "may", "are", "was", "were", "have", "has",
    "had", "not", "but", "use", "used", "into", "than", "then", "when",
    "what", "where", "which", "any", "all", "none", "true", "false",
    "self", "cls", "def", "class", "import", "return", "yield", "lambda",
    "test", "tests", "function", "method", "value", "values", "data",
}

_TOKEN_RX = re.compile(r"\b[a-zA-Z_][a-zA-Z0-9_]{2,}\b")


def _tokenize(text: str) -> set[str]:
    """Lowercase, length>=3, filter stopwords/digits."""
    out = set()
    for tok in _TOKEN_RX.findall(text.lower()):
        if tok not in _NOISE and not tok.isdigit():
            out.add(tok)
    return out


def _extract_ast_identifiers(source: str) -> set[str]:
    """Return public function/class names + import targets from the file's AST."""
    idents: set[str] = set()
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return idents
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if not node.name.startswith("_"):
                idents.add(node.name.lower())
        elif isinstance(node, ast.ImportFrom):
            for n in node.names:
                if n.name:
                    idents.add(n.name.lower())
    return idents


def _req_text(req: Any) -> str:
    """Pull the text out of either a dict or an object representation."""
    if isinstance(req, dict):
        return str(req.get("text") or req.get("description") or req.get("name") or "")
    return str(getattr(req, "text", None)
               or getattr(req, "description", None)
               or getattr(req, "name", None) or "")


# ── Stage 1: keyword filter ────────────────────────────────────────────────

def _keyword_score(file_idents: set[str], file_words: set[str], req_words: set[str]) -> int:
    """3x weight for AST identifier matches, 1x for general word matches."""
    return 3 * len(file_idents & req_words) + len(file_words & req_words)


def _keyword_top_n(file_source: str, requirements: list[Any], n: int = 30) -> list[Any]:
    if not requirements:
        return []
    file_idents = _extract_ast_identifiers(file_source)
    file_words  = _tokenize(file_source)
    scored: list[tuple[int, Any]] = []
    for req in requirements:
        rt = _req_text(req)
        if not rt:
            continue
        req_words = _tokenize(rt)
        s = _keyword_score(file_idents, file_words, req_words)
        if s > 0:
            scored.append((s, req))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [r for _, r in scored[:n]]


# ── Stage 2: semantic rerank (lazy) ────────────────────────────────────────

_SBERT_MODEL = None  # cached after first load


def _load_sbert():
    global _SBERT_MODEL
    if _SBERT_MODEL is None:
        from sentence_transformers import SentenceTransformer
        _SBERT_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("Loaded sentence-transformers all-MiniLM-L6-v2 for semantic rerank")
    return _SBERT_MODEL


def _semantic_rerank(file_source: str, candidates: list[Any], top_k: int) -> list[Any]:
    try:
        import numpy as np
        model = _load_sbert()
    except Exception as exc:
        logger.warning("Semantic rerank unavailable (%s) — falling back to keyword-only", exc)
        return candidates[:top_k]

    file_vec = model.encode(file_source[:1500], convert_to_numpy=True, show_progress_bar=False)
    texts = [_req_text(c) for c in candidates]
    req_vecs = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)

    fnorm = np.linalg.norm(file_vec) + 1e-9
    rnorms = np.linalg.norm(req_vecs, axis=1) + 1e-9
    sims = (req_vecs @ file_vec) / (rnorms * fnorm)

    ranked = sorted(zip(sims.tolist(), candidates), key=lambda x: x[0], reverse=True)
    return [r for _, r in ranked[:top_k]]


# ── Public API ─────────────────────────────────────────────────────────────

def match_requirements_to_file(file_source: str,
                                requirements: list[Any],
                                top_k: int = 10) -> list[Any]:
    """
    Return the top_k requirements most relevant to the given file.

    Pipeline:
      1. Keyword overlap filter → top 30 candidates
      2. If we have >top_k candidates, semantic rerank → top_k
      3. Otherwise return the keyword top_k directly

    Robust to missing sentence-transformers (falls back to keyword-only).
    Robust to malformed/empty requirements (filters them).
    """
    if not requirements or not file_source.strip():
        return []

    # Stage 1
    candidates = _keyword_top_n(file_source, requirements, n=max(30, top_k * 3))
    if not candidates:
        return []

    if len(candidates) <= top_k:
        return candidates

    # Stage 2
    return _semantic_rerank(file_source, candidates, top_k)
