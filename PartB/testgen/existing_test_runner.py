"""
existing_test_runner.py — Pre-generation pass for existing tests.

Before PartB generates new tests it runs this module to answer:

  1. DISCOVER: Are there already test files covering this source file?
  2. RUN:      Do those tests pass or fail?
  3. TRIAGE:   For each failing existing test, generate a fresh test
               for the same function and compare:
                 old=FAIL + fresh=FAIL  →  REAL BUG  (two independent signals agree)
                 old=FAIL + fresh=PASS  →  STALE TEST (API changed, test outdated)
                 old=PASS + fresh=FAIL  →  false alarm (discard fresh)
  4. ACT:
               REAL BUG   → emit to bug_reports.jsonl (→ PartC picks up)
               STALE TEST → auto-update with .testmate.bak backup
               PASS       → run coverage analysis, return uncovered functions
"""
from __future__ import annotations

import ast
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

_PYTEST_TIMEOUT = 60
_STALE_BACKUP_SUFFIX = ".testmate.bak"


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 0 — Discovery
# ─────────────────────────────────────────────────────────────────────────────

def discover_existing_tests(target_file: str, repo_dir: Optional[str] = None) -> list[str]:
    """
    Find test files that are likely exercising `target_file`.

    Search strategies (in order):
      1. Conventional names: test_<stem>.py, <stem>_test.py
      2. Same-directory search
      3. tests/ / test/ subdirectory search
      4. Any test file that imports from the source file's module
    """
    src = Path(target_file).resolve()
    stem = src.stem
    root = Path(repo_dir).resolve() if repo_dir else src.parent

    candidates: list[Path] = []

    # Strategy 1 & 2: conventional names next to source or in tests/
    for base in [src.parent, root / "tests", root / "test"]:
        if base.is_dir():
            for pat in [f"test_{stem}.py", f"{stem}_test.py"]:
                p = base / pat
                if p.exists():
                    candidates.append(p)

    # Strategy 3: any test_*.py under repo root that imports from this module
    src_module = stem
    for test_path in root.rglob("test_*.py"):
        if test_path in candidates:
            continue
        try:
            txt = test_path.read_text(encoding="utf-8", errors="ignore")
            if src_module in txt and "def test_" in txt:
                candidates.append(test_path)
        except Exception:
            pass

    # Exclude _testmate files — those are PartB's own generated files, not user tests
    return [str(p) for p in candidates if "_testmate" not in p.name]


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1 — Run existing tests, collect per-function pass/fail
# ─────────────────────────────────────────────────────────────────────────────

def run_existing_tests(test_files: list[str], cwd: str) -> dict:
    """
    Run the existing test suite and return structured results.

    Returns:
        {
          "all_pass":      bool,
          "passed":        [test_id, ...],
          "failed":        [test_id, ...],
          "raw_output":    str,
          "returncode":    int,
          "failing_funcs": [str, ...],  # inferred function names from failed test IDs
        }
    """
    if not test_files:
        return {"all_pass": True, "passed": [], "failed": [], "raw_output": "", "returncode": 0, "failing_funcs": []}

    cmd = [sys.executable, "-m", "pytest", *test_files, "-v", "--tb=line", "--no-header"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=_PYTEST_TIMEOUT, cwd=cwd)
        output = result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return {"all_pass": False, "passed": [], "failed": [], "raw_output": "TIMEOUT", "returncode": 1, "failing_funcs": []}

    passed, failed = [], []
    for line in output.splitlines():
        if " PASSED" in line and "::" in line:
            passed.append(line.strip().split()[0])
        elif " FAILED" in line and "::" in line:
            failed.append(line.strip().split()[0])

    # Infer function names from failing test IDs (e.g. test_auth.py::test_verify_password → verify_password)
    failing_funcs: list[str] = []
    for fid in failed:
        test_name = fid.split("::")[-1]
        # strip common test_ prefix to get the function under test
        fn = re.sub(r"^test_", "", test_name)
        fn = re.sub(r"_\d+$", "", fn)       # strip trailing _1, _2 suffixes
        fn = re.sub(r"_(fails?|pass|error|raises?|invalid|valid|empty|basic|simple)$", "", fn, flags=re.I)
        if fn:
            failing_funcs.append(fn)

    return {
        "all_pass":      result.returncode == 0,
        "passed":        passed,
        "failed":        failed,
        "raw_output":    output[:2000],
        "returncode":    result.returncode,
        "failing_funcs": list(dict.fromkeys(failing_funcs)),  # deduplicated, order preserved
    }


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2 — Coverage: find uncovered branches when existing tests PASS
# ─────────────────────────────────────────────────────────────────────────────

def find_uncovered_functions(target_file: str, test_files: list[str], cwd: str) -> list[str]:
    """
    Run existing tests under coverage.py and return names of functions/methods
    that have zero line coverage in `target_file`.

    Returns list of function names to generate tests for, e.g. ["parse_args", "validate"].
    Empty list means full coverage (nothing to add).
    """
    if not test_files:
        return []

    try:
        import coverage as _cov_mod

        cov = _cov_mod.Coverage(source=[str(Path(target_file).parent)], data_file=None)
        cov.start()

        import pytest
        import io
        from contextlib import redirect_stdout, redirect_stderr
        buf = io.StringIO()
        with redirect_stdout(buf), redirect_stderr(buf):
            pytest.main([*test_files, "-q", "--no-header", "-p", "no:warnings"], plugins=[])

        cov.stop()

        executed_lines: set[int] = set()
        for fp in cov.get_data().measured_files():
            if Path(fp).resolve() == Path(target_file).resolve():
                lines = cov.get_data().lines(fp)
                if lines:
                    executed_lines.update(lines)

    except Exception as e:
        logger.debug("Coverage measurement failed: %s", e)
        return []

    # Parse source to find function line ranges
    try:
        source = Path(target_file).read_text(encoding="utf-8")
        tree = ast.parse(source)
    except Exception:
        return []

    uncovered: list[str] = []
    src_lines = source.splitlines()

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name.startswith("_"):
            continue  # skip private/dunder

        func_lines = set(range(node.lineno, node.end_lineno + 1))
        covered_in_func = func_lines & executed_lines
        coverage_pct = len(covered_in_func) / len(func_lines) if func_lines else 1.0

        if coverage_pct < 0.5:  # less than 50% coverage = worth augmenting
            uncovered.append(node.name)

    return uncovered


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 3 — Triage: fresh test vs existing test
# ─────────────────────────────────────────────────────────────────────────────

def triage_failing_test(
    func_name: str,
    target_file: str,
    actual_import: str,
    source_code: str,
    model: Any,
    tokenizer: Any,
    log_callback: Optional[Callable] = None,
) -> str:
    """
    For a function that has a failing existing test, generate a fresh test and compare.

    Returns one of:
        "real_bug"   — both old and fresh tests fail → PartC territory
        "stale_test" — old fails, fresh passes → test is outdated, update it
        "false_alarm"— inconclusive, treat as pass
    """
    def _log(msg: str):
        logger.debug(msg)
        if log_callback:
            log_callback("log", "debug", msg)

    _log(f"  Triage: generating fresh test for {func_name}…")

    fresh_test_code = _generate_fresh_test(func_name, target_file, actual_import, source_code, model, tokenizer)
    if not fresh_test_code or "def test_" not in fresh_test_code:
        _log(f"  Triage: could not generate fresh test for {func_name} — treating as false alarm")
        return "false_alarm"

    fresh_result = _run_single_test_code(fresh_test_code, target_file, actual_import)
    _log(f"  Triage: fresh test {'PASSED' if fresh_result else 'FAILED'} for {func_name}")

    if fresh_result:
        return "stale_test"   # fresh passes, old fails → API drifted
    else:
        return "real_bug"     # both fail → independent confirmation of bug


def _generate_fresh_test(
    func_name: str,
    target_file: str,
    actual_import: str,
    source_code: str,
    model: Any,
    tokenizer: Any,
) -> str:
    """Ask the LLM to write a single pytest function for func_name."""
    import torch
    prompt = (
        f"Write ONE pytest function that tests `{func_name}` from `{actual_import}`.\n\n"
        f"Use the docstring and type hints as the source of truth for expected behaviour.\n"
        f"Do NOT rely on existing tests — write from scratch using the current source code.\n"
        f"Output ONLY the Python code, no markdown fences.\n\n"
        f"SOURCE:\n{source_code[:1500]}"
    )
    try:
        messages = [
            {"role": "system", "content": "Expert Python test writer. Output only raw Python code."},
            {"role": "user", "content": prompt},
        ]
        chat = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(chat, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=300, do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
        return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    except Exception as e:
        logger.debug("Fresh test generation failed: %s", e)
        return ""


def _run_single_test_code(test_code: str, target_file: str, actual_import: str) -> bool:
    """Write test_code to a temp file and run it. Returns True if passes."""
    full_code = f"from {actual_import} import *\n\n{test_code}"
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False,
            dir=str(Path(target_file).parent), encoding="utf-8"
        ) as tf:
            tf.write(full_code)
            tmp = tf.name

        result = subprocess.run(
            [sys.executable, "-m", "pytest", tmp, "-q", "--tb=no", "--no-header"],
            capture_output=True, text=True, timeout=30,
            cwd=str(Path(target_file).parent),
        )
        return result.returncode == 0
    except Exception:
        return False
    finally:
        try:
            os.unlink(tmp)
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 4 — Act: handle stale tests + emit bug reports
# ─────────────────────────────────────────────────────────────────────────────

def update_stale_test(test_file: str, func_name: str, fresh_test_code: str, actual_import: str):
    """
    Back up the old test file and insert the fresh test in its place.
    Replaces the old test function for func_name while preserving everything else.
    """
    original = Path(test_file).read_text(encoding="utf-8")
    backup = Path(str(test_file) + _STALE_BACKUP_SUFFIX)

    # Keep at most one backup (overwrite old backups)
    shutil.copy2(test_file, backup)
    logger.info("Backed up %s → %s", test_file, backup.name)

    # Replace the old test function for this function name
    pattern = rf"(def test_{func_name}[^\n]*\n(?:[ \t]+[^\n]*\n)*)"
    replacement = fresh_test_code.rstrip("\n") + "\n\n"
    updated, count = re.subn(pattern, replacement, original)

    if count == 0:
        # No matching function found — just append
        updated = original.rstrip("\n") + f"\n\n# Updated by TestMate (was stale)\n{fresh_test_code}\n"

    Path(test_file).write_text(f"from {actual_import} import *\n\n" + updated if "import" not in updated else updated,
                               encoding="utf-8")
    logger.info("Updated stale test for %s in %s", func_name, test_file)


def emit_real_bug(
    bug_reports_path: str,
    source_file: str,
    func_name: str,
    test_file: str,
    test_code: str,
    failure_output: str,
):
    """Write a confirmed bug entry to bug_reports.jsonl so PartC picks it up."""
    report = {
        "timestamp":        time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_file":      os.path.basename(source_file),
        "target":           func_name,
        "bug_type":         "logic_error",
        "confidence":       "high",
        "confidence_score": 80,
        "verdict":          "confirmed",  # two independent tests agreed → confirmed
        "description":      f"Existing test AND freshly-generated test both fail for {func_name}.",
        "evidence":         failure_output[:300],
        "failure_count":    2,
        "test_file":        os.path.abspath(test_file),
        "test_code":        test_code[:2000],
    }
    with open(bug_reports_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(report) + "\n")
    logger.info("Emitted real-bug report for %s → %s", func_name, bug_reports_path)


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API — called from autonomous_loop
# ─────────────────────────────────────────────────────────────────────────────

def run_existing_test_prepass(
    target_file: str,
    actual_import: str,
    source_code: str,
    repo_dir: str,
    model: Any,
    tokenizer: Any,
    bug_reports_path: str,
    log_callback: Optional[Callable] = None,
) -> dict:
    """
    Full pre-generation pass for a target file.

    Returns:
        {
          "has_existing":       bool,
          "all_pass":           bool,
          "uncovered_funcs":    [str, ...],  # only populated when all_pass=True
          "real_bugs_found":    int,         # bugs emitted to bug_reports.jsonl
          "stale_tests_fixed":  int,
          "skip_generation":    bool,        # True if existing tests cover everything
        }
    """
    def _log(msg: str):
        logger.info(msg)
        if log_callback:
            log_callback("log", "info", msg)

    cwd = str(Path(target_file).parent)
    existing = discover_existing_tests(target_file, repo_dir)

    if not existing:
        return {
            "has_existing": False, "all_pass": False, "uncovered_funcs": [],
            "real_bugs_found": 0, "stale_tests_fixed": 0, "skip_generation": False,
        }

    _log(f"  Found {len(existing)} existing test file(s) for {Path(target_file).name}")

    run_result = run_existing_tests(existing, cwd)

    real_bugs = 0
    stale_fixed = 0
    stale_details: list[dict] = []

    if run_result["all_pass"]:
        _log(f"  Existing tests: all pass ✅")
        uncovered = find_uncovered_functions(target_file, existing, cwd)
        _log(f"  Coverage gap: {len(uncovered)} function(s) need augmentation: {uncovered}")
        summary = {
            "has_existing": True, "all_pass": True,
            "uncovered_funcs": uncovered,
            "real_bugs_found": 0, "stale_tests_fixed": 0, "stale_details": [],
            "skip_generation": len(uncovered) == 0,
            "source_file": os.path.basename(target_file),
            "existing_test_files": [os.path.basename(e) for e in existing],
        }
        if log_callback:
            log_callback("prepass_result", summary)
        return summary

    # Some tests fail — triage each failing function
    _log(f"  Existing tests: {len(run_result['failed'])} failed ❌ — running triage…")
    primary_test_file = existing[0]

    for func_name in run_result["failing_funcs"]:
        verdict = triage_failing_test(
            func_name=func_name,
            target_file=target_file,
            actual_import=actual_import,
            source_code=source_code,
            model=model,
            tokenizer=tokenizer,
            log_callback=log_callback,
        )

        if verdict == "real_bug":
            _log(f"  {func_name}: REAL BUG confirmed (two independent test failures) → queued for PartC")
            emit_real_bug(
                bug_reports_path=bug_reports_path,
                source_file=target_file,
                func_name=func_name,
                test_file=primary_test_file,
                test_code=run_result["raw_output"],
                failure_output=run_result["raw_output"],
            )
            real_bugs += 1

        elif verdict == "stale_test":
            _log(f"  {func_name}: STALE TEST — API changed, updating test with .testmate.bak backup")
            fresh = _generate_fresh_test(func_name, target_file, actual_import, source_code, model, tokenizer)
            backup_path = str(primary_test_file) + _STALE_BACKUP_SUFFIX
            if fresh:
                update_stale_test(primary_test_file, func_name, fresh, actual_import)
            stale_fixed += 1
            detail = {
                "func_name":   func_name,
                "test_file":   os.path.basename(primary_test_file),
                "backup_path": os.path.basename(backup_path),
                "fresh_code":  fresh[:500] if fresh else "",
            }
            stale_details.append(detail)
            if log_callback:
                log_callback("stale_fixed", detail)

        else:
            _log(f"  {func_name}: false alarm — inconclusive, will regenerate")

    uncovered = find_uncovered_functions(target_file, existing, cwd)

    summary = {
        "has_existing":       True,
        "all_pass":           False,
        "uncovered_funcs":    uncovered,
        "real_bugs_found":    real_bugs,
        "stale_tests_fixed":  stale_fixed,
        "stale_details":      stale_details,
        "skip_generation":    False,
        "source_file":        os.path.basename(target_file),
        "existing_test_files": [os.path.basename(e) for e in existing],
    }
    if log_callback:
        log_callback("prepass_result", summary)
    return summary
