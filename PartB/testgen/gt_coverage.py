"""
Ground-truth coverage metric for test quality.

Given:
  - A ground-truth (human-written) test for a function
  - Our generated test for the same function

Measure how much of the human test's behavioural coverage the generated
test reproduces, via two signals:

  1. Symbol overlap   — which functions/attrs from the SUT are exercised
  2. Assertion overlap — which logical assertions appear in both

Pure CPU, AST-only. No model, no network. Designed to be cheap enough to
run after every generation in the autonomous loop.
"""
from __future__ import annotations

import ast
import logging
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


# ── Symbol extraction ─────────────────────────────────────────────────────

def _extract_called_symbols(test_code: str) -> set[str]:
    """
    Return all names called or attribute-accessed in test_code,
    minus pytest/mock noise.
    """
    _NOISE = {
        "pytest", "raises", "approx", "fixture", "mark", "parametrize",
        "warns", "assert_called", "assert_called_once", "assert_called_with",
        "assert_called_once_with", "patch", "Mock", "MagicMock", "call",
        "isinstance", "len", "str", "int", "float", "list", "dict", "tuple",
        "set", "bool", "print", "type", "open", "range", "enumerate",
        "ANY", "side_effect", "return_value",
    }
    syms: set[str] = set()
    try:
        tree = ast.parse(test_code)
    except SyntaxError:
        return syms
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name = None
            if isinstance(func, ast.Name):
                name = func.id
            elif isinstance(func, ast.Attribute):
                name = func.attr
            if name and name not in _NOISE and not name.startswith("_"):
                syms.add(name)
        elif isinstance(node, ast.Attribute):
            if node.attr and node.attr not in _NOISE and not node.attr.startswith("_"):
                syms.add(node.attr)
    return syms


# ── Assertion extraction ──────────────────────────────────────────────────

def _normalize_assert(node: ast.Assert) -> str:
    """
    Reduce an assertion to a canonical string for comparison.
    Strips literal values but keeps structure + operator + identifiers.
    """
    try:
        src = ast.unparse(node.test)
    except Exception:
        return ""

    # Normalize: replace string literals and numbers with placeholders so
    # 'assert x == 5' and 'assert x == 3' compare equal.
    import re
    src = re.sub(r"'[^']*'", "'_str_'", src)
    src = re.sub(r'"[^"]*"', "'_str_'", src)
    src = re.sub(r"\b\d+\.?\d*\b", "_num_", src)
    return src.strip()


def _extract_assertions(test_code: str) -> set[str]:
    """Return the set of normalized assertion expressions."""
    asserts: set[str] = set()
    try:
        tree = ast.parse(test_code)
    except SyntaxError:
        return asserts
    for node in ast.walk(tree):
        if isinstance(node, ast.Assert):
            norm = _normalize_assert(node)
            if norm:
                asserts.add(norm)
    return asserts


# ── Public scoring API ────────────────────────────────────────────────────

@dataclass
class GTCoverageResult:
    symbol_recall:    float  # how much of GT's SUT call set we cover
    assertion_recall: float  # how much of GT's logical asserts we match
    combined:         float  # weighted average (0.4 sym + 0.6 assert)
    gt_symbols:       int
    matched_symbols:  int
    gt_assertions:    int
    matched_assertions: int

    def to_dict(self) -> dict:
        return asdict(self)


def compute_gt_coverage(
    generated_test: str,
    ground_truth_test: str,
) -> GTCoverageResult:
    """
    Compute how well the generated test reproduces the ground-truth test's
    coverage of the SUT.
    """
    gt_syms     = _extract_called_symbols(ground_truth_test)
    gen_syms    = _extract_called_symbols(generated_test)
    gt_asserts  = _extract_assertions(ground_truth_test)
    gen_asserts = _extract_assertions(generated_test)

    matched_syms    = len(gt_syms & gen_syms)
    matched_asserts = len(gt_asserts & gen_asserts)

    sym_recall    = matched_syms    / len(gt_syms)    if gt_syms    else 1.0
    assert_recall = matched_asserts / len(gt_asserts) if gt_asserts else 1.0
    combined      = 0.4 * sym_recall + 0.6 * assert_recall

    return GTCoverageResult(
        symbol_recall      = round(sym_recall, 3),
        assertion_recall   = round(assert_recall, 3),
        combined           = round(combined, 3),
        gt_symbols         = len(gt_syms),
        matched_symbols    = matched_syms,
        gt_assertions      = len(gt_asserts),
        matched_assertions = matched_asserts,
    )


def batch_gt_coverage(
    pairs: list[tuple[str, str]],
) -> dict:
    """
    Run gt_coverage on a list of (generated, ground_truth) test pairs.
    Returns an aggregate summary.
    """
    if not pairs:
        return {"total": 0, "avg_combined": 0.0, "per_case": []}

    per_case = [compute_gt_coverage(g, t).to_dict() for g, t in pairs]
    avg_combined = sum(c["combined"] for c in per_case) / len(per_case)

    return {
        "total":         len(pairs),
        "avg_combined":  round(avg_combined, 3),
        "avg_symbol":    round(sum(c["symbol_recall"]    for c in per_case) / len(pairs), 3),
        "avg_assertion": round(sum(c["assertion_recall"] for c in per_case) / len(pairs), 3),
        "per_case":      per_case,
    }


if __name__ == "__main__":
    import sys, json
    if len(sys.argv) != 3:
        print("Usage: python gt_coverage.py <generated_test.py> <ground_truth_test.py>")
        sys.exit(1)
    gen = open(sys.argv[1], encoding="utf-8").read()
    ref = open(sys.argv[2], encoding="utf-8").read()
    r = compute_gt_coverage(gen, ref)
    print(json.dumps(r.to_dict(), indent=2))
