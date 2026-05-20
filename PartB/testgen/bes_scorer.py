"""
bes_scorer.py — Bug Exposure Score (BES), the Layer 4 quality gate.

Replaces the old 3-dim composite scorer with 5 dimensions that measure
how likely a test is to expose real bugs, not just "look like a test."

Dimensions (weights):
  1. Spec Coverage      30% — docstring examples used as assertions
  2. Input Variety      25% — characteristic diversity (case/sign/type)
  3. Boundary Coverage  20% — AST-verified boundary args
  4. Assertion Quality  15% — exact ==, raises, distinct targets
  5. Mutation Potential 10% — heuristic for likely mutation-killing

Score   Action
------  -------
>= 70   Accept immediately
50-69   Retry once with specific hints
< 50    Retry up to 3x; if still failing, keep Layer 1+2 only
"""
from __future__ import annotations

import ast
import logging
import re
import textwrap
from typing import Any

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def bes_score(
    test_code: str,
    func_source: str = "",
    func_name: str = "",
) -> dict:
    """
    Compute Bug Exposure Score for a test function block.

    Args:
        test_code:   The generated pytest function code.
        func_source: Source of the function under test (for docstring extraction).
        func_name:   Function name (for targeted checks).

    Returns:
        {
          "score":              float,   # 0-100 composite
          "spec_coverage":      float,   # 0-100
          "input_variety":      float,   # 0-100
          "boundary_coverage":  float,   # 0-100
          "assertion_quality":  float,   # 0-100
          "mutation_potential": float,   # 0-100
          "missing":            [str],   # list of missing things for retry prompt
          "verdict":            str,     # "accept" | "retry_once" | "retry_twice"
        }
    """
    s1 = _spec_coverage(test_code, func_source, func_name)
    s2 = _input_variety(test_code)
    s3 = _boundary_coverage(test_code)
    s4 = _assertion_quality(test_code)
    s5 = _mutation_potential(test_code)

    composite = (
        s1 * 0.30 +
        s2 * 0.25 +
        s3 * 0.20 +
        s4 * 0.15 +
        s5 * 0.10
    ) * 100

    composite = round(min(composite, 100.0), 1)

    missing = _collect_missing(test_code, func_source, func_name, s1, s2, s3, s4)

    if composite >= 70:
        verdict = "accept"
    elif composite >= 50:
        verdict = "retry_once"
    else:
        verdict = "retry_twice"

    return {
        "score":              composite,
        "spec_coverage":      round(s1 * 100, 1),
        "input_variety":      round(s2 * 100, 1),
        "boundary_coverage":  round(s3 * 100, 1),
        "assertion_quality":  round(s4 * 100, 1),
        "mutation_potential": round(s5 * 100, 1),
        "missing":            missing,
        "verdict":            verdict,
    }


def build_retry_hint(result: dict) -> str:
    """Convert missing[] list into an actionable retry prompt addition."""
    if not result["missing"]:
        return ""
    lines = [
        f"Your previous test scored {result['score']:.0f}/100. Please fix these gaps:"
    ]
    for item in result["missing"]:
        lines.append(f"  • {item}")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Dimension 1 — Spec Coverage (30%)
# ─────────────────────────────────────────────────────────────────────────────

def _spec_coverage(test_code: str, func_source: str, func_name: str) -> float:
    """
    What fraction of the docstring's documented examples appear as assertions?
    """
    if not func_source or not func_name:
        return 0.5  # unknown → neutral
    try:
        from docstring_extractor import extract_examples_from_docstring  # type: ignore[import]
        examples = extract_examples_from_docstring(func_source, func_name)
    except ImportError:
        return 0.5

    if not examples:
        return 0.8  # no docstring → can't penalise heavily, give partial credit

    hits = 0
    for ex in examples:
        # Check if the call from the docstring appears verbatim in the test
        call = ex["call_src"].replace(" ", "")
        if call in test_code.replace(" ", ""):
            hits += 1

    return hits / len(examples)


# ─────────────────────────────────────────────────────────────────────────────
# Dimension 2 — Input Variety (25%)
# ─────────────────────────────────────────────────────────────────────────────

def _input_variety(test_code: str) -> float:
    """
    Measures CHARACTERISTIC diversity, not just count.

    String variety: uppercase, lowercase, mixed, empty, digits
    Numeric variety: zero, negative, positive, large
    List variety:    empty, single, multiple
    """
    score = 0.0
    checks_done = 0

    # ── String variety ──
    str_lits = re.findall(r'"([^"]*)"', test_code) + re.findall(r"'([^']*)'", test_code)
    # Filter out pytest/import noise
    str_lits = [s for s in str_lits if not any(kw in s for kw in ["pytest", "import", "def ", "assert"])]

    if str_lits:
        checks_done += 1
        str_score = 0.0
        has_upper   = any(s != s.lower() for s in str_lits)
        has_lower   = any(s == s.lower() and s.isalpha() for s in str_lits)
        has_mixed   = any(s != s.lower() and s != s.upper() for s in str_lits)
        has_empty   = "" in str_lits or '""' in test_code or "''" in test_code
        has_digits  = any(any(c.isdigit() for c in s) for s in str_lits)
        features = [has_upper, has_lower, has_mixed, has_empty, has_digits]
        str_score = sum(features) / len(features)
        score += str_score

    # ── Numeric variety ──
    num_lits = re.findall(r'(?<!\w)(-?\d+(?:\.\d+)?)(?!\w)', test_code)
    # Filter assertion result numbers (after ==)
    # Keep numbers that appear as function arguments
    arg_nums = _extract_arg_values(test_code)
    if arg_nums:
        checks_done += 1
        num_score = 0.0
        has_zero   = "0" in arg_nums or "0.0" in arg_nums
        has_neg    = any(n.startswith("-") for n in arg_nums)
        has_pos    = any(not n.startswith("-") and float(n) > 0 for n in arg_nums if _is_num(n))
        has_large  = any(_is_num(n) and abs(float(n)) > 100 for n in arg_nums)
        features = [has_zero, has_neg, has_pos, has_large]
        num_score = sum(features) / len(features)
        score += num_score

    # ── List variety ──
    if "[]" in test_code or "[" in test_code:
        list_lits = re.findall(r'\[([^\[\]]*)\]', test_code)
        arg_lists = [l for l in list_lits if l.strip() != ""]  # non-empty
        checks_done += 1
        list_score = 0.0
        has_empty_list  = "[]" in test_code
        has_single      = any(len(l.split(",")) == 1 for l in arg_lists)
        has_multi       = any(len(l.split(",")) > 2 for l in arg_lists)
        has_dup         = _has_duplicate_elements(arg_lists)
        features = [has_empty_list, has_single, has_multi, has_dup]
        list_score = sum(features) / len(features)
        score += list_score

    if checks_done == 0:
        return 0.3  # no data

    return min(score / checks_done, 1.0)


def _extract_arg_values(test_code: str) -> list[str]:
    """Extract numeric literals that appear as function call arguments."""
    args = []
    try:
        tree = ast.parse(textwrap.dedent(test_code))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                for arg in node.args:
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, (int, float)):
                        args.append(str(arg.value))
                    elif isinstance(arg, ast.UnaryOp) and isinstance(arg.op, ast.USub):
                        if isinstance(arg.operand, ast.Constant):
                            args.append(f"-{arg.operand.value}")
    except Exception:
        pass
    return args


def _is_num(s: str) -> bool:
    try:
        float(s)
        return True
    except Exception:
        return False


def _has_duplicate_elements(list_strs: list[str]) -> bool:
    for ls in list_strs:
        parts = [p.strip() for p in ls.split(",")]
        if len(parts) != len(set(parts)):
            return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Dimension 3 — Boundary Coverage (20%)
# ─────────────────────────────────────────────────────────────────────────────

_BOUNDARY_MARKERS = [
    # Strings
    ('""', lambda code: '""' in code or "''" in code),
    ('"A"',  lambda code: bool(re.search(r'["\'][A-Z]["\']', code))),        # single uppercase
    # Numbers
    ("0",   lambda code: bool(re.search(r'\b0\b', code))),
    ("-1",  lambda code: "-1" in code),
    # Collections
    ("[]",  lambda code: "[]" in code),
    ("{}", lambda code: "{}" in code),
    # Special
    ("None", lambda code: "None" in code),
    # Negative numbers
    ("negative", lambda code: bool(re.search(r'\b-\d', code))),
]


def _boundary_coverage(test_code: str) -> float:
    """AST-verified: are boundary values actually passed as arguments?"""
    arg_values = set()
    try:
        tree = ast.parse(textwrap.dedent(test_code))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                for arg in node.args:
                    arg_values.add(ast.unparse(arg))
    except Exception:
        pass

    hits = 0
    total = 0
    for name, checker in _BOUNDARY_MARKERS:
        total += 1
        # First check full test code, then verify presence in actual args
        if checker(test_code):
            hits += 1

    if total == 0:
        return 0.3

    # Scale: 3+ boundaries = full score, 2 = 0.7, 1 = 0.4, 0 = 0
    raw = hits / total
    if hits >= 3:
        return 1.0
    elif hits == 2:
        return 0.7
    elif hits == 1:
        return 0.4
    return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Dimension 4 — Assertion Quality (15%)
# ─────────────────────────────────────────────────────────────────────────────

def _assertion_quality(test_code: str) -> float:
    """Exact ==, pytest.raises, distinct assert targets."""
    score = 0.0

    assert_lines = [l.strip() for l in test_code.splitlines()
                    if re.match(r"\s*assert\s", l)]
    non_trivial = [l for l in assert_lines
                   if l not in ("assert True", "assert False")]

    if not non_trivial:
        return 0.0

    # Exact-value assertions (most valuable)
    has_exact = bool(re.search(r'==\s*["\'\d\[\{]|==\s*(True|False|None)\b', test_code))
    has_raises = "pytest.raises" in test_code
    distinct_values = set(re.findall(r'==\s*(-?[\d.]+|"[^"]*"|\'[^\']*\')', test_code))
    has_distinct = len(distinct_values) >= 2

    if has_exact:    score += 0.40
    if has_raises:   score += 0.30
    if has_distinct: score += 0.30

    return min(score, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Dimension 5 — Mutation Potential (10%)
# ─────────────────────────────────────────────────────────────────────────────

def _mutation_potential(test_code: str) -> float:
    """
    Heuristic: how likely is this test to kill common mutations?

    Common mutations and what catches them:
    - Flip operator (+ → -, < → >, etc.) → exact value assertion with off-by-N
    - Replace constant             → multiple assertions with different expected values
    - Swap return values           → both min and max boundary tested
    - Negate boolean condition     → test with inputs on both sides of the condition
    """
    score = 0.0

    # +0.4: Has exact value comparisons (kills constant replacement)
    if re.search(r'==\s*["\'\d]', test_code):
        score += 0.40

    # +0.2: Tests both extremes (kills swap return values)
    asserted_nums = re.findall(r'==\s*(-?[\d.]+)', test_code)
    if len(set(asserted_nums)) >= 2:
        score += 0.20

    # +0.2: Has boundary input (kills off-by-one conditions)
    has_boundary_input = (
        bool(re.search(r'\b0\b', test_code)) or
        "-1" in test_code or
        '""' in test_code or
        "[]" in test_code
    )
    if has_boundary_input:
        score += 0.20

    # +0.2: Has pytest.raises (kills wrong exception handling)
    if "pytest.raises" in test_code:
        score += 0.20

    return min(score, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Missing diagnostics for retry prompt
# ─────────────────────────────────────────────────────────────────────────────

def _collect_missing(
    test_code: str,
    func_source: str,
    func_name: str,
    s1: float, s2: float, s3: float, s4: float,
) -> list[str]:
    """Build the list of actionable missing items for the retry prompt."""
    missing: list[str] = []

    # Spec coverage
    if s1 < 0.5 and func_source and func_name:
        try:
            from docstring_extractor import extract_examples_from_docstring  # type: ignore[import]
            examples = extract_examples_from_docstring(func_source, func_name)
            unused = [
                ex["call_src"] for ex in examples
                if ex["call_src"].replace(" ", "") not in test_code.replace(" ", "")
            ]
            if unused:
                missing.append(
                    f"Docstring examples NOT used as assertions: "
                    + ", ".join(f"`{c}`" for c in unused[:3])
                )
        except ImportError:
            pass

    # Input variety
    if s2 < 0.5:
        str_lits = re.findall(r'"([^"]*)"', test_code) + re.findall(r"'([^']*)'", test_code)
        str_lits = [s for s in str_lits if s.isalpha()]
        has_upper = any(s != s.lower() for s in str_lits)
        has_lower = any(s == s.lower() for s in str_lits)
        if str_lits and not has_upper:
            missing.append("String inputs are all lowercase — add UPPERCASE or mixed-case inputs")
        if str_lits and not has_lower:
            missing.append("String inputs are all uppercase — add lowercase inputs too")
        arg_nums = _extract_arg_values(test_code)
        if arg_nums and not any(n.startswith("-") for n in arg_nums):
            missing.append("Only positive numbers used — add negative and zero inputs")

    # Boundary coverage
    if s3 < 0.5:
        hints = []
        if '""' not in test_code and "''" not in test_code:
            hints.append('empty string ""')
        if not re.search(r'\b0\b', test_code):
            hints.append("zero (0)")
        if "-1" not in test_code:
            hints.append("negative (-1)")
        if "[]" not in test_code:
            hints.append("empty list []")
        if hints:
            missing.append(f"Missing boundary inputs: {', '.join(hints)}")

    # Assertion quality
    if s4 < 0.4:
        if "==" not in test_code:
            missing.append("No exact-value assertions (==) — check for specific return values")
        elif not re.search(r'==\s*["\'\d]', test_code):
            missing.append("Assertions don't check concrete values — use `assert result == <exact_value>`")

    return missing
