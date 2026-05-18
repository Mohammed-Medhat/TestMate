"""
Cheap AST-based "is this file worth generating tests for?" filter.

Skips:
  - Empty modules / pure docstrings
  - __init__.py with only imports / __all__
  - Files with no public functions or classes
  - Files that are themselves test files
  - Files dominated by constants / type aliases

100% CPU, no model needed. Used to prune target lists before Part B.
"""
from __future__ import annotations

import ast
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def is_worth_testing(path: str | Path) -> tuple[bool, str]:
    """
    Returns (worth_testing, reason).
    Conservative — when in doubt, returns True (don't skip).
    """
    p = Path(path)
    if not p.exists():
        return False, "file does not exist"

    name = p.name
    if name.startswith("test_") or name.endswith("_test.py") or name == "conftest.py":
        return False, "is itself a test file"

    try:
        source = p.read_text(encoding="utf-8", errors="ignore")
    except OSError as exc:
        return False, f"unreadable: {exc}"

    stripped = source.strip()
    if not stripped:
        return False, "empty file"

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False, "syntax error in source"

    public_funcs = 0
    public_classes = 0
    has_logic = False

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not node.name.startswith("_"):
                public_funcs += 1
                has_logic = True
        elif isinstance(node, ast.ClassDef):
            if not node.name.startswith("_"):
                public_classes += 1
                has_logic = True
                for cls_node in node.body:
                    if isinstance(cls_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        if not cls_node.name.startswith("_") or cls_node.name == "__init__":
                            public_funcs += 1
        elif isinstance(node, (ast.If, ast.For, ast.While, ast.Try)):
            has_logic = True

    if public_funcs == 0 and public_classes == 0:
        return False, "no public functions or classes"

    if name == "__init__.py" and not has_logic and public_funcs == 0 and public_classes == 0:
        return False, "__init__.py with no logic"

    return True, f"{public_funcs} func(s), {public_classes} class(es)"


def filter_testable(target_files: list[dict]) -> tuple[list[dict], list[dict]]:
    """
    Split a target list into (worth_testing, skipped).
    Each item must have 'abs_path'. Skipped items get a 'skip_reason' field.
    """
    keep: list[dict] = []
    skip: list[dict] = []
    for f in target_files:
        ok, reason = is_worth_testing(f.get("abs_path", ""))
        if ok:
            keep.append(f)
        else:
            skip.append({**f, "skip_reason": reason})
    return keep, skip
