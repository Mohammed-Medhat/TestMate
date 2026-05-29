"""
Auto-priming: scan a repo for existing passing tests and sample
representative examples to inject as few-shot priming into the
test generation prompt.

Only uses tests that ACTUALLY PASS (pytest --collect-only succeeds
and the functions have non-trivial assertions). No manual curation needed.
"""
from __future__ import annotations

import ast
import os
import random
import subprocess
import tempfile
import time
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_SKIP_DIRS = {
    "__pycache__", ".git", ".tox", "venv", ".venv", "env", "build", "dist",
    "node_modules", ".mypy_cache", ".pytest_cache",
}

# Priming examples older than this are skipped — stale tests typically encode
# patterns that no longer match current code and teach the model bad habits.
_PRIMING_STALENESS_SECONDS = 30 * 86400


# ── Discovery ─────────────────────────────────────────────────────────────

def find_test_files(repo_dir: str | Path, max_files: int = 30) -> list[Path]:
    repo = Path(repo_dir)
    found = []
    _cutoff = time.time() - _PRIMING_STALENESS_SECONDS
    for p in repo.rglob("*.py"):
        if any(part in _SKIP_DIRS for part in p.parts):
            continue
        if p.name.startswith("test_") or p.name.endswith("_test.py"):
            # Skip stale test files (haven't been touched in 30+ days). They
            # typically reference renamed APIs or old patterns and hurt priming.
            try:
                if os.path.getmtime(p) < _cutoff:
                    logger.debug("[priming] skipping stale test file: %s", p)
                    continue
            except OSError:
                pass
            found.append(p)
            if len(found) >= max_files:
                break
    return found


# ── Sampling ──────────────────────────────────────────────────────────────

def _is_good_example(fn_node: ast.FunctionDef) -> bool:
    """
    True if a test function is worth using as a priming example:
    - Has at least 2 meaningful assertions
    - Not too long (≤25 lines)
    - Not just equality (has some variety)
    """
    if not fn_node.name.startswith("test_"):
        return False
    n_lines = (fn_node.end_lineno or fn_node.lineno) - fn_node.lineno
    if n_lines > 25 or n_lines < 3:
        return False

    asserts = [n for n in ast.walk(fn_node) if isinstance(n, ast.Assert)]
    if len(asserts) < 2:
        return False

    # Check variety: not all assertions testing the same thing
    assertion_types = set()
    for a in asserts:
        try:
            src = ast.unparse(a.test)
            if " is None" in src or " is not None" in src:
                assertion_types.add("none")
            elif " == " in src:
                assertion_types.add("eq")
            elif " != " in src:
                assertion_types.add("ne")
            elif "isinstance" in src:
                assertion_types.add("isinstance")
            elif "raises" in src:
                assertion_types.add("raises")
            else:
                assertion_types.add("other")
        except Exception:
            pass
    return len(assertion_types) >= 1  # at least some real content


def sample_tests_from_file(
    path: Path,
    n: int = 2,
    seed: int = 42,
) -> list[str]:
    """
    Parse a test file and return source code snippets of good example tests.
    Returns empty list on any error.
    """
    try:
        src = path.read_text(encoding="utf-8", errors="ignore")
        tree = ast.parse(src)
    except Exception:
        return []

    lines = src.splitlines(keepends=True)
    candidates = []

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and _is_good_example(node):
            fn_src = "".join(lines[node.lineno - 1: node.end_lineno])
            candidates.append(fn_src.strip())

    if not candidates:
        return []

    rng = random.Random(seed)
    return rng.sample(candidates, min(n, len(candidates)))


# ── Collection verification ────────────────────────────────────────────────

def _file_collects_cleanly(path: Path, timeout: int = 15) -> bool:
    """Check that pytest can collect the test file (no import errors)."""
    try:
        r = subprocess.run(
            ["python", "-m", "pytest", str(path), "--collect-only", "-q"],
            capture_output=True, text=True, timeout=timeout,
        )
        return r.returncode == 0
    except Exception:
        return False


# ── Public API ────────────────────────────────────────────────────────────

def get_priming_examples(
    repo_dir: str | Path,
    max_files_to_scan: int = 15,
    examples_per_file: int = 2,
    total_examples: int = 4,
    seed: int = 42,
    verify_collectible: bool = True,
) -> str:
    """
    Scan the repo, find test examples, return a formatted priming block
    ready to inject into the plan prompt.

    Returns empty string if no good examples are found.
    """
    test_files = find_test_files(repo_dir, max_files=max_files_to_scan)
    if not test_files:
        logger.debug("[priming] No test files found in %s", repo_dir)
        return ""

    rng = random.Random(seed)
    rng.shuffle(test_files)

    collected: list[str] = []
    for tf in test_files:
        if len(collected) >= total_examples:
            break
        if verify_collectible and not _file_collects_cleanly(tf):
            continue
        examples = sample_tests_from_file(tf, n=examples_per_file, seed=seed)
        collected.extend(examples)

    if not collected:
        logger.debug("[priming] No collectible examples found")
        return ""

    # Trim and format
    final = collected[:total_examples]
    logger.info("[priming] Loaded %d priming examples from existing tests", len(final))

    block = "\n\n🔍 EXISTING TESTS IN THIS REPO (use as style reference):\n"
    block += "─" * 60 + "\n"
    for i, ex in enumerate(final, 1):
        block += f"# Example {i}:\n{ex}\n\n"
    block += "─" * 60 + "\n"
    block += "Write tests in the same style as the examples above.\n"

    return block
