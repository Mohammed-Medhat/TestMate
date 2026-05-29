"""
O5 — Public-symbol allowlist for generated test imports.

Validates that a generated test file only imports symbols that actually exist
in the target module's public API.  Catches hallucinated imports BEFORE the
first pytest run, saving the token budget that would otherwise be spent on a
retry triggered by an `ImportError` traceback.

Approach: pure-AST.  Parse the source module to collect every top-level public
name (functions, classes, module-level assignments whose target name does not
start with '_').  Parse the generated test, find every `from <import_path>
import <names>`, and check each requested name against that allow-list.

Returns a list of human-readable warnings; an empty list means all imports are
known-valid.  Used as a hint injected into the retry prompt.

Designed to be framework-agnostic — no Django/SymPy/Flask knowledge — and to
fail open: if either AST parse fails, returns [] so the validator never
blocks generation.
"""
from __future__ import annotations

import ast


def collect_public_symbols(source_code: str) -> set[str]:
    """Return the set of top-level public names in the source module.

    Public means: defined at module level (not inside a class or function),
    and the name does not start with '_'.  Includes:
      - `def name():`
      - `async def name():`
      - `class Name:`
      - `name = ...` (simple assignments)
      - `name: int = ...` (annotated assignments)
      - `from x import name` and `import name` (re-exports)

    If `__all__` is defined explicitly, returns exactly that set instead
    (the standard Python convention for "this is my public surface").
    """
    try:
        tree = ast.parse(source_code)
    except SyntaxError:
        return set()

    # If __all__ exists, use it as the authoritative public surface.
    all_names = _extract_all_list(tree)
    if all_names is not None:
        return all_names

    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if not node.name.startswith("_"):
                names.add(node.name)
        elif isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and not tgt.id.startswith("_"):
                    names.add(tgt.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if not node.target.id.startswith("_"):
                names.add(node.target.id)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.asname or alias.name.split(".")[0]
                if not name.startswith("_"):
                    names.add(name)
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name == "*":
                    continue
                name = alias.asname or alias.name
                if not name.startswith("_"):
                    names.add(name)
    return names


def _extract_all_list(tree: ast.Module) -> set[str] | None:
    """If the module has `__all__ = [...]`, return the literal name set.
    Returns None if absent or not a static list/tuple of string literals."""
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id == "__all__":
                    if isinstance(node.value, (ast.List, ast.Tuple)):
                        out: set[str] = set()
                        for elt in node.value.elts:
                            if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                                out.add(elt.value)
                        return out
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id == "__all__" and isinstance(node.value, (ast.List, ast.Tuple)):
                out = set()
                for elt in node.value.elts:
                    if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                        out.add(elt.value)
                return out
    return None


def validate_test_imports(
    test_code: str, import_path: str, source_code: str
) -> list[str]:
    """Check that every `from <import_path> import X` in the test refers to a
    real public symbol in `source_code`.

    Returns a list of warning strings, one per suspect import.  Empty list = OK.

    Soft-fail: if parsing fails or `import_path` is empty, returns [] so the
    validator never blocks the test generation flow.
    """
    if not import_path or not test_code:
        return []

    try:
        test_tree = ast.parse(test_code)
    except SyntaxError:
        return []  # if the test won't parse, downstream pytest will catch it

    public = collect_public_symbols(source_code)
    if not public:
        return []  # if we can't determine the public API, don't warn

    warnings: list[str] = []
    # `from a.b.c import X` matches when the test imports from the target
    # module OR from a strict ancestor (e.g. `from django.contrib.admin import X`
    # when the target is `django.contrib.admin.checks` — the symbol still
    # belongs to that ancestor's __init__).  We only validate the EXACT match
    # to avoid false positives on ancestor imports we can't see into.
    for node in ast.walk(test_tree):
        if isinstance(node, ast.ImportFrom) and node.module == import_path:
            for alias in node.names:
                if alias.name == "*":
                    continue
                if alias.name.startswith("_"):
                    warnings.append(
                        f"imports private symbol '{alias.name}' from {import_path} "
                        f"— private symbols are not part of the public API"
                    )
                elif alias.name not in public:
                    warnings.append(
                        f"imports unknown symbol '{alias.name}' from {import_path} "
                        f"— not found in the module's public API"
                    )
    return warnings


def format_warnings_for_prompt(warnings: list[str]) -> str:
    """Format a warning list as a prompt-friendly hint block. Empty if no
    warnings."""
    if not warnings:
        return ""
    lines = ["\n⚠️  IMPORT-CHECK WARNINGS — these imports likely won't resolve:"]
    for w in warnings:
        lines.append(f"  - {w}")
    lines.append(
        "  → On the next attempt, only import names you SEE in the source code "
        "above.  Do not invent or guess private/underscore-prefixed names.\n"
    )
    return "\n".join(lines)
