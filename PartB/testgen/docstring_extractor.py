"""
docstring_extractor.py — Layer 1 of the test amplifier.

Parses function docstrings to extract concrete (call, expected) examples
that the LLM might otherwise overlook. These deterministic examples are
GUARANTEED to expose any bug that contradicts the documented spec.

Supported formats:
  1. Arrow:    func(args)  ->  result          (Google/numpy style)
  2. Arrow:    func(args)  =>  result
  3. Doctest:  >>> func(args)
               result
  4. Raises:   func(args) raises ValueError
               func(args)  ->  raises ValueError
"""
from __future__ import annotations

import ast
import logging
import re
import textwrap
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def extract_examples_from_docstring(func_source: str, func_name: str) -> list[dict]:
    """
    Pull all (call_source, expected, raises) tuples out of the docstring.

    Args:
        func_source: Full source of the function (including def + docstring).
        func_name:   Name of the function we're extracting examples for.

    Returns:
        List of:
            {"call_src": "count_vowels(\"AEIOU\")", "expected": 5, "raises": None}
            {"call_src": "divide(1, 0)", "expected": None, "raises": "ZeroDivisionError"}
        Empty list if no docstring or no parseable examples.
    """
    docstring = _extract_docstring(func_source)
    if not docstring:
        return []

    examples: list[dict] = []
    examples += _parse_arrow_examples(docstring, func_name)
    examples += _parse_doctest_examples(docstring, func_name)
    examples += _parse_raises_examples(docstring, func_name)

    # De-dup by call_src
    seen: set[str] = set()
    unique: list[dict] = []
    for ex in examples:
        key = ex["call_src"]
        if key not in seen:
            seen.add(key)
            unique.append(ex)
    return unique


def build_docstring_tests(
    func_source: str,
    func_name: str,
    actual_import: str,
) -> str:
    """
    Convert extracted docstring examples into a block of pytest functions.
    Returns the test code as a string (empty if no examples found).
    """
    examples = extract_examples_from_docstring(func_source, func_name)
    if not examples:
        return ""

    blocks: list[str] = []
    for i, ex in enumerate(examples, 1):
        call_src = ex["call_src"]
        expected = ex.get("expected")
        raises = ex.get("raises")

        if raises:
            blocks.append(
                f"def test_{func_name}_docstring_raises_{i}():\n"
                f"    \"\"\"Docstring example {i}: should raise {raises}.\"\"\"\n"
                f"    import pytest\n"
                f"    with pytest.raises({raises}):\n"
                f"        {call_src}\n"
            )
        elif expected is not None:
            # Use approx for floats to be forgiving of precision
            if isinstance(expected, float):
                blocks.append(
                    f"def test_{func_name}_docstring_example_{i}():\n"
                    f"    \"\"\"Docstring example {i}: {call_src} -> {expected}.\"\"\"\n"
                    f"    import pytest\n"
                    f"    assert {call_src} == pytest.approx({expected!r}, rel=1e-6)\n"
                )
            else:
                blocks.append(
                    f"def test_{func_name}_docstring_example_{i}():\n"
                    f"    \"\"\"Docstring example {i}: {call_src} -> {expected!r}.\"\"\"\n"
                    f"    assert {call_src} == {expected!r}\n"
                )

    return "\n\n".join(blocks)


# ─────────────────────────────────────────────────────────────────────────────
# Internals
# ─────────────────────────────────────────────────────────────────────────────

def _extract_docstring(func_source: str) -> Optional[str]:
    """Extract docstring text from a function source snippet."""
    try:
        tree = ast.parse(textwrap.dedent(func_source))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                return ast.get_docstring(node)
    except Exception as e:
        logger.debug("docstring extraction failed: %s", e)
    return None


def _safe_literal_eval(text: str) -> Any:
    """Parse a literal Python value (int, float, str, list, dict, etc.) safely."""
    try:
        return ast.literal_eval(text.strip())
    except Exception:
        # Maybe it's a bare identifier like 'True' / 'False' / 'None'
        v = text.strip()
        if v == "True":  return True
        if v == "False": return False
        if v == "None":  return None
        return None


def _looks_like_call(text: str, func_name: str) -> bool:
    """Quick check: does this look like a call to func_name?"""
    return re.search(rf"\b{re.escape(func_name)}\s*\(", text) is not None


# ── Format 1: Arrow style ───────────────────────────────────────────────────
# Examples this matches:
#   count_vowels("Hello") -> 3
#   count_vowels("Hello")  ->  3
#   count_vowels("Hello") => 3
#   count_vowels("Hello"): 3                      (colon style)

_ARROW_RE = re.compile(
    r"^\s*([\w\.]+\s*\(.*?\))\s*(?:->|=>|:)\s*(.+?)\s*$",
    re.MULTILINE,
)


def _parse_arrow_examples(docstring: str, func_name: str) -> list[dict]:
    """Parse 'func(args) -> result' lines from docstring."""
    out: list[dict] = []
    for match in _ARROW_RE.finditer(docstring):
        call_src, result_text = match.group(1).strip(), match.group(2).strip()
        if not _looks_like_call(call_src, func_name):
            continue
        # Strip trailing inline comments
        result_text = re.sub(r"\s*#.*$", "", result_text)
        # Strip parenthetical annotations like "(freezing point)"
        result_text = re.sub(r"\s*\(.*?\)\s*$", "", result_text)
        if not result_text:
            continue

        # Validate the call parses as Python
        try:
            ast.parse(call_src, mode="eval")
        except SyntaxError:
            continue

        # Check if it's a "raises X" pattern
        m_raises = re.match(r"^raises?\s+(\w+)", result_text, re.I)
        if m_raises:
            out.append({"call_src": call_src, "expected": None, "raises": m_raises.group(1)})
            continue

        expected = _safe_literal_eval(result_text)
        if expected is None and result_text not in ("None", ""):
            continue
        out.append({"call_src": call_src, "expected": expected, "raises": None})
    return out


# ── Format 2: Doctest style ─────────────────────────────────────────────────
# >>> count_vowels("Hello")
# 3
# >>> divide(1, 0)
# Traceback (most recent call last):
#   ...
# ZeroDivisionError

_DOCTEST_CALL_RE = re.compile(r"^\s*>>>\s+(.+?)\s*$", re.MULTILINE)


def _parse_doctest_examples(docstring: str, func_name: str) -> list[dict]:
    """Parse >>> doctest examples."""
    out: list[dict] = []
    lines = docstring.splitlines()
    i = 0
    while i < len(lines):
        m = _DOCTEST_CALL_RE.match(lines[i])
        if not m:
            i += 1
            continue
        call_src = m.group(1).strip()
        if not _looks_like_call(call_src, func_name):
            i += 1
            continue
        # Validate it parses
        try:
            ast.parse(call_src, mode="eval")
        except SyntaxError:
            i += 1
            continue

        # Look at next non-empty line for result
        j = i + 1
        while j < len(lines) and not lines[j].strip():
            j += 1
        if j >= len(lines):
            break
        next_line = lines[j].strip()

        # Check for "Traceback" pattern (exception)
        if next_line.startswith("Traceback"):
            # Find the exception class name (last non-empty before next >>> or end)
            k = j
            exc_name = None
            while k < len(lines):
                if _DOCTEST_CALL_RE.match(lines[k]):
                    break
                m_exc = re.match(r"^\s*(\w*Error|Exception)\b", lines[k])
                if m_exc:
                    exc_name = m_exc.group(1)
                k += 1
            if exc_name:
                out.append({"call_src": call_src, "expected": None, "raises": exc_name})
            i = k
            continue

        # Plain value result
        if not next_line.startswith(">>>"):
            expected = _safe_literal_eval(next_line)
            if expected is not None or next_line in ("None", "True", "False"):
                out.append({"call_src": call_src, "expected": expected, "raises": None})
        i = j + 1
    return out


# ── Format 3: Raises style ──────────────────────────────────────────────────
# Examples:
#   divide(1, 0) raises ZeroDivisionError
#   func() raises ValueError

_RAISES_RE = re.compile(
    r"^\s*([\w\.]+\s*\(.*?\))\s+raises?\s+(\w+)",
    re.MULTILINE | re.IGNORECASE,
)


def _parse_raises_examples(docstring: str, func_name: str) -> list[dict]:
    """Parse 'func(args) raises ExceptionClass' lines."""
    out: list[dict] = []
    for m in _RAISES_RE.finditer(docstring):
        call_src = m.group(1).strip()
        exc_name = m.group(2).strip()
        if not _looks_like_call(call_src, func_name):
            continue
        try:
            ast.parse(call_src, mode="eval")
        except SyntaxError:
            continue
        out.append({"call_src": call_src, "expected": None, "raises": exc_name})
    return out
