"""
Ground-truth coverage metric for SWE-bench evaluation.

Measures how much overlap exists between the generated tests and the
human-written reference tests (test_patch from SWE-bench instances).

Two metrics:
1. Symbol overlap  — what % of symbols/identifiers in ground-truth tests
                     appear in generated tests
2. Assertion overlap — what % of assertion patterns in ground-truth
                       appear in generated tests

These are NOT functional equivalence checks — they're coverage proxies
that correlate with the likelihood that generated tests exercise
the same behavior as the reference tests.
"""
from __future__ import annotations

import ast
import re
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_IDENT_RX = re.compile(r"\b[a-zA-Z_][a-zA-Z0-9_]{2,}\b")
_NOISE = {
    "def", "class", "import", "from", "return", "assert", "raise",
    "with", "for", "while", "if", "else", "elif", "try", "except",
    "finally", "pass", "None", "True", "False", "self", "cls",
    "pytest", "mock", "patch", "unittest", "MagicMock",
}


def _extract_assertion_patterns(code: str) -> set[str]:
    """Extract abstracted assertion patterns (independent of exact values)."""
    patterns = set()
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return patterns

    for node in ast.walk(tree):
        if isinstance(node, ast.Assert):
            try:
                src = ast.unparse(node.test)
                # Abstract out specific literals → pattern
                pattern = re.sub(r"['\"][^'\"]*['\"]", "STR", src)
                pattern = re.sub(r"\b\d+\b", "NUM", pattern)
                pattern = re.sub(r"\b[a-z_][a-z0-9_]*\b", "VAR", pattern)
                patterns.add(pattern[:80])
            except Exception:
                pass
        elif isinstance(node, ast.Call):
            try:
                func = ast.unparse(node.func)
                if "raises" in func or "warns" in func:
                    patterns.add(f"expects_exception:{func}")
            except Exception:
                pass

    return patterns


def _extract_key_identifiers(code: str) -> set[str]:
    """Extract key identifier tokens (method names, class names) from test code."""
    tokens = {t.lower() for t in _IDENT_RX.findall(code) if t not in _NOISE}
    # Filter very common Python words
    return {t for t in tokens if len(t) >= 3}


def compute_gt_overlap(
    generated_tests: list[str],
    ground_truth_test: str,
) -> dict:
    """
    Compute overlap between generated tests and the ground-truth test.

    Args:
        generated_tests:   List of generated test file contents
        ground_truth_test: The reference test from SWE-bench test_patch

    Returns:
        {
          "symbol_overlap_pct":    float,   # % of GT symbols in generated
          "assertion_overlap_pct": float,   # % of GT assertion patterns in generated
          "gt_symbols":            int,     # total symbols in GT
          "gt_assertions":         int,     # total assertion patterns in GT
          "matched_symbols":       int,
          "matched_assertions":    int,
        }
    """
    if not ground_truth_test or not generated_tests:
        return {
            "symbol_overlap_pct": 0.0,
            "assertion_overlap_pct": 0.0,
            "gt_symbols": 0,
            "gt_assertions": 0,
            "matched_symbols": 0,
            "matched_assertions": 0,
        }

    gt_symbols    = _extract_key_identifiers(ground_truth_test)
    gt_assertions = _extract_assertion_patterns(ground_truth_test)

    # Combine all generated tests
    all_generated = "\n\n".join(generated_tests)
    gen_symbols    = _extract_key_identifiers(all_generated)
    gen_assertions = _extract_assertion_patterns(all_generated)

    matched_sym = len(gt_symbols & gen_symbols)
    matched_ass = len(gt_assertions & gen_assertions)

    return {
        "symbol_overlap_pct":    round(matched_sym / max(len(gt_symbols), 1) * 100, 1),
        "assertion_overlap_pct": round(matched_ass / max(len(gt_assertions), 1) * 100, 1),
        "gt_symbols":            len(gt_symbols),
        "gt_assertions":         len(gt_assertions),
        "matched_symbols":       matched_sym,
        "matched_assertions":    matched_ass,
    }


def compute_gt_overlap_for_results(
    per_file_results: list[dict],
    ground_truth_map: dict[str, str],
) -> dict:
    """
    Compute GT overlap across multiple files.

    Args:
        per_file_results: List of Part B result dicts (with 'file' and 'test_code')
        ground_truth_map: Dict mapping filename → ground_truth_test_code
                          (from SWE-bench test_patch field)

    Returns aggregate metrics.
    """
    all_overlaps = []

    for result in per_file_results:
        fname = Path(result.get("file", "")).name
        gt    = ground_truth_map.get(fname, "")
        if not gt:
            continue
        generated = [result.get("test_code", "")]
        overlap = compute_gt_overlap(generated, gt)
        overlap["file"] = fname
        all_overlaps.append(overlap)
        logger.info(
            "[gt_overlap] %s: symbol=%.1f%% assertion=%.1f%%",
            fname,
            overlap["symbol_overlap_pct"],
            overlap["assertion_overlap_pct"],
        )

    if not all_overlaps:
        return {"files_evaluated": 0, "mean_symbol_overlap": 0.0, "mean_assertion_overlap": 0.0}

    return {
        "files_evaluated":       len(all_overlaps),
        "mean_symbol_overlap":   round(sum(o["symbol_overlap_pct"] for o in all_overlaps) / len(all_overlaps), 1),
        "mean_assertion_overlap": round(sum(o["assertion_overlap_pct"] for o in all_overlaps) / len(all_overlaps), 1),
        "per_file":              all_overlaps,
    }
