"""
Post-generation quality gates for generated test files.

Gates (conservative — when in doubt, keep the test):
1. AST symbol verification  — reject tests calling clearly non-existent methods
2. Tautology / placeholder  — reject trivially-true assertions and empty bodies
3. Duplicate function names — rename collisions so pytest runs all tests
4. "None" string fix        — replace == "None" with is None

All gates operate on the test source string. None involve model inference.
"""
from __future__ import annotations

import ast
import re
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Whitelisted names that are always safe (never flag as hallucinated) ───
_SAFE_ATTRS = {
    # pytest
    "raises", "mark", "fixture", "approx", "warns", "deprecated_call",
    "param", "parametrize", "skip", "skipif", "xfail", "importorskip",
    # mock
    "call", "call_count", "called", "return_value", "side_effect",
    "assert_called", "assert_called_once", "assert_called_with",
    "assert_called_once_with", "assert_any_call", "assert_has_calls",
    "reset_mock", "configure_mock",
    # general Python
    "args", "kwargs", "message", "errno", "strerror",
    "status_code", "text", "json", "content", "headers", "url",
    "ok", "reason", "encoding", "elapsed", "history", "cookies",
    # common testing patterns
    "startswith", "endswith", "strip", "split", "join", "format",
    "decode", "encode", "read", "write", "close", "open",
    "append", "extend", "pop", "get", "set", "update", "items",
    "keys", "values", "copy", "clear", "count", "index",
    "__class__", "__dict__", "__name__", "__module__", "__doc__",
}

_TAUTOLOGY_PATTERNS = [
    r"\bassert\s+True\s*(?:==\s*True)?\b",
    r"\bassert\s+False\s*==\s*False\b",
    r"\bassert\s+1\s*==\s*1\b",
    r"\bassert\s+0\s*==\s*0\b",
    r"\bassert\s+\"[^\"]*\"\s*==\s*\"[^\"]*\"\b",   # assert "x" == "x"
    r"\bassert\s+'[^']*'\s*==\s*'[^']*'\b",
    r"\bassert\s+(\w+)\s*==\s*\1\b",               # assert x == x
]
_TAUTOLOGY_RX = re.compile("|".join(_TAUTOLOGY_PATTERNS))

# ── 1. AST symbol verification ────────────────────────────────────────────

def _extract_source_symbols(source_code: str) -> set[str]:
    """Get all public names defined in the source file (classes, functions, methods)."""
    symbols: set[str] = set()
    try:
        tree = ast.parse(source_code)
    except SyntaxError:
        return symbols
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if not node.name.startswith("__"):
                symbols.add(node.name.lower())
    return symbols


def _count_suspicious_calls(test_code: str, source_symbols: set[str]) -> int:
    """
    Count attribute accesses that look clearly invented.
    Only flag: SomeImportedClass().invented_snake_case_method()
    Conservative: return count (caller decides threshold).
    """
    suspicious = 0
    try:
        tree = ast.parse(test_code)
    except SyntaxError:
        return 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue
        attr = node.attr
        if attr.startswith("_") or attr in _SAFE_ATTRS:
            continue
        # Only flag if the attribute looks like a method name (snake_case, >4 chars)
        # AND is not found anywhere in source symbols
        if (
            len(attr) > 4
            and "_" in attr                    # snake_case → likely invented method
            and attr.lower() not in source_symbols
            and not attr[0].isupper()          # not a class name
        ):
            suspicious += 1
    return suspicious


def verify_ast(test_code: str, source_code: str, threshold: int = 4) -> tuple[bool, str]:
    """
    Returns (is_valid, reason).
    Conservative: only rejects if suspicious count >= threshold.
    """
    source_symbols = _extract_source_symbols(source_code)
    count = _count_suspicious_calls(test_code, source_symbols)
    if count >= threshold:
        return False, f"AST gate: {count} likely-hallucinated attribute calls (threshold={threshold})"
    return True, ""


# ── 2. Tautology / placeholder detection ─────────────────────────────────

def _get_test_functions(test_code: str) -> list[ast.FunctionDef]:
    try:
        tree = ast.parse(test_code)
        return [n for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef) and n.name.startswith("test_")]
    except SyntaxError:
        return []


def _function_has_meaningful_assertion(fn: ast.FunctionDef) -> bool:
    """True if the function body does something other than trivial or no assertions."""
    has_any_assert = False
    for node in ast.walk(fn):
        if isinstance(node, ast.Assert):
            has_any_assert = True
            # Check for tautology: assert True / assert x == x
            test_src = ""
            try:
                test_src = ast.unparse(node.test)
            except Exception:
                pass
            if test_src in ("True", "False == False", "1 == 1", "not False"):
                continue  # skip tautological assert
            return True  # found a non-tautological assert
        if isinstance(node, ast.Call):
            func_str = ""
            try:
                func_str = ast.unparse(node.func)
            except Exception:
                pass
            if "raises" in func_str or "warn" in func_str:
                return True
    return has_any_assert


def filter_tautologies(test_code: str) -> tuple[str, int]:
    """
    Remove test functions that are pure placeholders.
    Returns (cleaned_code, n_removed).
    """
    try:
        tree = ast.parse(test_code)
    except SyntaxError:
        return test_code, 0

    lines = test_code.splitlines(keepends=True)
    remove_ranges: list[tuple[int, int]] = []  # (start_lineno, end_lineno) 1-indexed

    for fn in _get_test_functions(test_code):
        if not _function_has_meaningful_assertion(fn):
            # Check for string tautologies via regex in the function source
            fn_src = "".join(lines[fn.lineno - 1: fn.end_lineno])
            if _TAUTOLOGY_RX.search(fn_src):
                remove_ranges.append((fn.lineno, fn.end_lineno))
            elif not _function_has_meaningful_assertion(fn):
                # Completely empty body (only pass / docstring / return)
                body_nodes = [n for n in fn.body
                              if not isinstance(n, (ast.Pass, ast.Expr, ast.Return))]
                if not body_nodes:
                    # Check for any assert at all
                    if not any(isinstance(n, ast.Assert) for n in ast.walk(fn)):
                        remove_ranges.append((fn.lineno, fn.end_lineno))

    if not remove_ranges:
        return test_code, 0

    removed = 0
    result_lines = list(lines)
    # Remove in reverse order so line numbers stay valid
    for start, end in sorted(remove_ranges, reverse=True):
        # Add a comment stub so pytest doesn't complain about empty modules
        result_lines[start - 1: end] = []
        removed += 1

    return "".join(result_lines), removed


# ── 3. Duplicate function name dedup ──────────────────────────────────────

def fix_duplicate_names(test_code: str) -> tuple[str, int]:
    """
    Rename duplicate test function definitions so all tests run.
    test_foo, test_foo  →  test_foo, test_foo_v2
    Returns (fixed_code, n_renamed).
    """
    seen: dict[str, int] = {}
    renamed = 0

    def _replacer(m: re.Match) -> str:
        nonlocal renamed
        name = m.group(1)
        if name not in seen:
            seen[name] = 1
            return m.group(0)
        seen[name] += 1
        renamed += 1
        new_name = f"{name}_v{seen[name]}"
        return m.group(0).replace(f"def {name}", f"def {new_name}", 1)

    fixed = re.sub(r"^(def (test_\w+))", lambda m: _replacer(m), test_code,
                   flags=re.MULTILINE)
    # Simpler reliable approach:
    seen2: dict[str, int] = {}
    renamed2 = 0
    lines = test_code.splitlines(keepends=True)
    result = []
    for line in lines:
        m = re.match(r"^(def (test_\w+)\s*\()", line)
        if m:
            name = m.group(2)
            if name in seen2:
                seen2[name] += 1
                new_name = f"{name}_v{seen2[name]}"
                line = line.replace(f"def {name}", f"def {new_name}", 1)
                renamed2 += 1
            else:
                seen2[name] = 1
        result.append(line)
    return "".join(result), renamed2


# ── 4. Fix common semantic mistakes ──────────────────────────────────────

def fix_none_string(test_code: str) -> tuple[str, int]:
    """
    Replace == "None" / == 'None' with is None in assertions.
    Returns (fixed_code, n_fixed).
    """
    original = test_code
    fixed = re.sub(r'==\s*["\']None["\']', 'is None', test_code)
    fixed = re.sub(r'!=\s*["\']None["\']', 'is not None', fixed)
    count = fixed.count('is None') - original.count('is None')
    count += fixed.count('is not None') - original.count('is not None')
    return fixed, max(0, count)


# ── Master gate function ──────────────────────────────────────────────────

def apply_all_gates(
    test_code: str,
    source_code: str,
    ast_threshold: int = 4,
    verbose: bool = True,
) -> tuple[str | None, dict]:
    """
    Apply all quality gates to a generated test file.

    Returns (cleaned_test_code, report).
    Returns (None, report) if the whole test file should be discarded.
    """
    report: dict = {
        "ast_rejected": False,
        "tautologies_removed": 0,
        "duplicates_renamed": 0,
        "none_strings_fixed": 0,
    }

    # Gate 1: AST verification (whole-file reject)
    valid, reason = verify_ast(test_code, source_code, threshold=ast_threshold)
    if not valid:
        report["ast_rejected"] = True
        report["ast_reason"] = reason
        if verbose:
            logger.warning("[quality_gate] REJECTED: %s", reason)
        return None, report

    # Gate 2: Tautology filter (remove individual bad tests)
    test_code, n_removed = filter_tautologies(test_code)
    report["tautologies_removed"] = n_removed
    if n_removed and verbose:
        logger.info("[quality_gate] Removed %d tautological test(s)", n_removed)

    # Gate 3: Duplicate name fix
    test_code, n_renamed = fix_duplicate_names(test_code)
    report["duplicates_renamed"] = n_renamed
    if n_renamed and verbose:
        logger.info("[quality_gate] Renamed %d duplicate function(s)", n_renamed)

    # Gate 4: Semantic None-string fix
    test_code, n_fixed = fix_none_string(test_code)
    report["none_strings_fixed"] = n_fixed
    if n_fixed and verbose:
        logger.info("[quality_gate] Fixed %d '== \"None\"' → 'is None'", n_fixed)

    return test_code, report
