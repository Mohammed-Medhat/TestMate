"""
boundary_synthesizer.py — Layer 2 of the test amplifier.

Generates boundary/edge-case test functions deterministically from
type hints + docstring anchor, with zero LLM calls.

Only runs when docstring examples exist to calibrate expected values
(per design decision: avoids guessing at expected output).

Strategy per param type:
  str   → "", "A", "AaA", mixed-case, digits, spaces
  int   → 0, 1, -1, large positive, large negative
  float → 0.0, -1.0, very small, very large
  list  → [], [x], [x,x] (duplicates), many elements
  dict  → {}, {k:v}

For each boundary input, we derive the expected output by:
  1. Calling the function via subprocess (Oracle 4 style, safe)
  2. If that fails → skip that boundary (don't guess)
"""
from __future__ import annotations

import ast
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import textwrap
from typing import Any, Optional

logger = logging.getLogger(__name__)

_EXEC_TIMEOUT = 8


# ─────────────────────────────────────────────────────────────────────────────
# Boundary input templates per type
# ─────────────────────────────────────────────────────────────────────────────

_STR_BOUNDARIES = [
    ("empty",        '""'),
    ("single_lower", '"a"'),
    ("single_upper", '"A"'),
    ("all_upper",    '"AEIOU"'),
    ("mixed_case",   '"Hello World"'),
    ("digits",       '"12345"'),
    ("whitespace",   '" "'),
]

_INT_BOUNDARIES = [
    ("zero",           "0"),
    ("one",            "1"),
    ("neg_one",        "-1"),
    ("large_positive", "1000"),
    ("large_negative", "-1000"),
]

_FLOAT_BOUNDARIES = [
    ("zero",         "0.0"),
    ("small",        "0.001"),
    ("neg",          "-1.0"),
    ("large",        "1e6"),
]

_LIST_BOUNDARIES = [
    ("empty",      "[]"),
    ("single",     "[1]"),
    ("two",        "[1, 2]"),
    ("duplicates", "[1, 1, 2]"),
    ("many",       "[3, 1, 4, 1, 5, 9, 2, 6]"),
]

_BOUNDARIES_BY_TYPE: dict[str, list[tuple[str, str]]] = {
    "str":   _STR_BOUNDARIES,
    "int":   _INT_BOUNDARIES,
    "float": _FLOAT_BOUNDARIES,
    "list":  _LIST_BOUNDARIES,
}


# ─────────────────────────────────────────────────────────────────────────────
# Parameter extraction
# ─────────────────────────────────────────────────────────────────────────────

def _extract_params(func_source: str) -> list[dict]:
    """
    Parse a function's parameter names and type annotations.
    Returns: [{"name": "text", "type": "str"}, ...]
    """
    try:
        tree = ast.parse(textwrap.dedent(func_source))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                params = []
                for arg in node.args.args:
                    if arg.arg in ("self", "cls"):
                        continue
                    ann = None
                    if arg.annotation:
                        try:
                            ann = ast.unparse(arg.annotation).strip()
                            # Simplify Optional[X] -> X
                            ann = re.sub(r"Optional\[(\w+)\]", r"\1", ann)
                            # Simplify List[X] -> list
                            ann = re.sub(r"List\[.*?\]", "list", ann)
                            # Simplify Dict[...] -> dict
                            ann = re.sub(r"Dict\[.*?\]", "dict", ann)
                        except Exception:
                            ann = None
                    params.append({"name": arg.arg, "type": (ann or "").lower()})
                return params
    except Exception as e:
        logger.debug("param extraction failed: %s", e)
    return []


# ─────────────────────────────────────────────────────────────────────────────
# Safe function execution (oracle)
# ─────────────────────────────────────────────────────────────────────────────

_RUNNER_TMPL = """
import sys, json, types

# Mock common external dependencies
_MOCK_MODULES = ["requests","httpx","django","flask","sqlalchemy","pymongo","redis","boto3","celery"]
for _m in _MOCK_MODULES:
    if _m not in sys.modules:
        _pkg = types.ModuleType(_m)
        _pkg.__path__ = []
        sys.modules[_m] = _pkg

sys.path.insert(0, {source_dir!r})
try:
    from {module_name} import *
except Exception as e:
    print(json.dumps({{"error": str(e)}}))
    sys.exit(0)

try:
    _result = {call}
    print(json.dumps({{"ok": True, "result": repr(_result)}}))
except Exception as _e:
    print(json.dumps({{"ok": False, "raises": type(_e).__name__}}))
"""


def _exec_call(target_file: str, module_name: str, call_src: str) -> dict:
    """Execute call_src importing from target_file, return {ok, result/raises}."""
    source_dir = str(__import__("pathlib").Path(target_file).parent)
    script = _RUNNER_TMPL.format(
        source_dir=source_dir, module_name=module_name, call=call_src
    )
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as tf:
            tf.write(script)
            path = tf.name
        proc = subprocess.run(
            [sys.executable, path],
            capture_output=True, text=True, timeout=_EXEC_TIMEOUT,
        )
        raw = proc.stdout.strip()
        if raw:
            return json.loads(raw)
    except Exception as e:
        logger.debug("boundary exec failed: %s", e)
    finally:
        try:
            os.unlink(path)
        except Exception:
            pass
    return {"ok": False, "raises": "ExecutionError"}


# ─────────────────────────────────────────────────────────────────────────────
# Boundary test generation
# ─────────────────────────────────────────────────────────────────────────────

def build_boundary_tests(
    func_source: str,
    func_name: str,
    target_file: str,
    module_name: str,
    actual_import: str,
) -> str:
    """
    Generate boundary pytest functions for func_name.

    Only generates boundaries for parameters where a docstring anchor exists
    (we need at least one known input→output to trust the function's behavior).
    Falls back silently when execution fails.

    Returns pytest function block as a string (empty if nothing usable found).
    """
    from docstring_extractor import extract_examples_from_docstring  # type: ignore[import]

    # Need at least one docstring example to calibrate
    examples = extract_examples_from_docstring(func_source, func_name)
    if not examples:
        logger.debug("boundary_synthesizer: no docstring anchor for %s — skipping", func_name)
        return ""

    params = _extract_params(func_source)
    if not params:
        return ""

    blocks: list[str] = []

    for param in params:
        ptype = param["type"]
        # Match type to boundary table
        boundary_key = None
        for key in _BOUNDARIES_BY_TYPE:
            if key in ptype:
                boundary_key = key
                break
        if not boundary_key:
            continue

        boundaries = _BOUNDARIES_BY_TYPE[boundary_key]

        for label, boundary_literal in boundaries:
            # Build the call using boundary value for this param,
            # keeping other params at their docstring-example values.
            call_src = _build_call_with_boundary(
                func_name, params, param["name"], boundary_literal, examples
            )
            if not call_src:
                continue

            # Execute to get the expected output from the REAL function
            result = _exec_call(target_file, module_name, call_src)

            if not result.get("ok"):
                exc = result.get("raises", "")
                # If it raises, write a pytest.raises test
                if exc and exc != "ExecutionError":
                    blocks.append(
                        f"def test_{func_name}_boundary_{label}_raises():\n"
                        f"    import pytest\n"
                        f"    with pytest.raises({exc}):\n"
                        f"        {call_src}\n"
                    )
                continue

            expected_repr = result.get("result", "")
            if not expected_repr:
                continue

            # Validate the expected repr is safe to embed in code
            try:
                ast.literal_eval(expected_repr)
            except Exception:
                continue

            # Check if this boundary is already covered by docstring examples
            if _already_in_examples(call_src, examples):
                continue

            expected_val = expected_repr
            # Use approx for floats
            if "." in expected_repr and re.match(r"^-?\d+\.\d+$", expected_repr):
                blocks.append(
                    f"def test_{func_name}_boundary_{label}():\n"
                    f"    import pytest\n"
                    f"    assert {call_src} == pytest.approx({expected_val}, rel=1e-6)\n"
                )
            else:
                blocks.append(
                    f"def test_{func_name}_boundary_{label}():\n"
                    f"    assert {call_src} == {expected_val}\n"
                )

    return "\n\n".join(blocks)


def _build_call_with_boundary(
    func_name: str,
    params: list[dict],
    target_param: str,
    boundary_literal: str,
    examples: list[dict],
) -> Optional[str]:
    """
    Build a call like func(boundary_val, other_param_val) by pulling
    non-target param values from the first docstring example.
    """
    if not examples:
        return None

    # Try to extract arg values from the first docstring example
    first_example = examples[0]["call_src"]
    try:
        call_ast = ast.parse(first_example, mode="eval").body
        if not isinstance(call_ast, ast.Call):
            return None
        example_args = [ast.unparse(a) for a in call_ast.args]
    except Exception:
        return None

    if len(example_args) != len(params):
        return None

    # Substitute boundary_literal for target param
    param_names = [p["name"] for p in params]
    new_args: list[str] = []
    for i, (pname, _) in enumerate(zip(param_names, example_args)):
        if pname == target_param:
            new_args.append(boundary_literal)
        else:
            new_args.append(example_args[i])

    return f"{func_name}({', '.join(new_args)})"


def _already_in_examples(call_src: str, examples: list[dict]) -> bool:
    """True if this exact call already appears in the docstring examples."""
    for ex in examples:
        if ex["call_src"].strip() == call_src.strip():
            return True
    return False
