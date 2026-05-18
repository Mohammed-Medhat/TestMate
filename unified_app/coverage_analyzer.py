"""
SRS Coverage Analyzer — finds which SRS requirements are not covered
by the generated test code, and returns them as "gaps" for the gap-fill pass.

A requirement is considered "covered" if at least 2 of its key identifiers
appear in any of the generated test files.
"""
from __future__ import annotations

import re
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_IDENT_RX = re.compile(r"\b[a-zA-Z_][a-zA-Z0-9_]{2,}\b")
_NOISE = {
    "the", "system", "shall", "must", "accept", "return", "provide", "support",
    "allow", "when", "that", "from", "with", "this", "have", "will", "each",
    "only", "than", "into", "and", "for", "not", "any", "all", "none", "true",
    "false", "self", "import", "def", "class", "assert", "test", "function",
    "method", "value", "data", "type", "object", "string", "number", "bool",
}


def _key_tokens(text: str) -> set[str]:
    """Extract meaningful identifiers from a requirement string."""
    return {
        t.lower() for t in _IDENT_RX.findall(text)
        if t.lower() not in _NOISE and len(t) >= 3
    }


def compute_srs_coverage(
    requirements: list[dict],
    test_files_content: list[str],
) -> dict:
    """
    Given a list of SRS requirements and the content of generated test files,
    return which requirements are covered and which are gaps.

    Args:
        requirements:       List of requirement dicts (with 'text', 'label', etc.)
        test_files_content: List of strings, each being a generated test file's source

    Returns:
        {
          "total":    int,
          "covered":  int,
          "gaps":     [requirement dicts],
          "coverage_pct": float,
        }
    """
    # Only real requirements (label == 1)
    real_reqs = [r for r in requirements if r.get("label") == 1 and r.get("text", "").strip()]
    if not real_reqs:
        return {"total": 0, "covered": 0, "gaps": [], "coverage_pct": 0.0}

    # Build a bag of all tokens from ALL generated test files
    all_test_tokens: set[str] = set()
    for content in test_files_content:
        all_test_tokens |= _key_tokens(content)

    covered = []
    gaps = []

    for req in real_reqs:
        req_tokens = _key_tokens(req.get("text", ""))
        if not req_tokens:
            covered.append(req)
            continue
        hits = len(req_tokens & all_test_tokens)
        # Covered if at least 2 key tokens appear in test code
        if hits >= 2:
            covered.append(req)
        else:
            gaps.append(req)

    total = len(real_reqs)
    n_covered = len(covered)
    pct = round(n_covered / total * 100, 1) if total > 0 else 0.0

    logger.info("[coverage] %d/%d requirements covered (%.1f%%), %d gaps",
                n_covered, total, pct, len(gaps))

    return {
        "total":        total,
        "covered":      n_covered,
        "gaps":         gaps,
        "coverage_pct": pct,
    }


def load_generated_tests(per_file_results: list[dict]) -> list[str]:
    """Read test_code from per-file results; also try reading from disk."""
    contents = []
    for f in per_file_results:
        code = f.get("test_code", "")
        if code:
            contents.append(code)
            continue
        # fall back to reading from disk
        test_file = f.get("test_file") or f.get("file", "")
        if test_file:
            p = Path(test_file)
            if p.exists():
                try:
                    contents.append(p.read_text(encoding="utf-8", errors="ignore"))
                except OSError:
                    pass
    return contents
