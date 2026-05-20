"""
run_and_collect.py — Per-test coverage runner for SBFL.

Strategy: sequential in-process execution with one Coverage() instance
per test. This avoids:
  - subprocess startup overhead (1-2s per test)
  - CTracer thread-safety conflicts (only occur with concurrent threads)

Each test gets its own Coverage() object created fresh, started, stopped,
and discarded before the next test begins — no shared state.
"""
import os
import sys
import json
from pathlib import Path
from collections import defaultdict
import subprocess


# ── Helpers ───────────────────────────────────────────────────────────

def get_all_tests(test_path: str = ".") -> list:
    """Collect all pytest test IDs without running them."""
    result = subprocess.run(
        ["pytest", "--collect-only", "-q", test_path],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    return [
        line.strip()
        for line in result.stdout.splitlines()
        if line.strip() and "::" in line
    ]


def _run_single_test_with_coverage(test_id: str, source_dir: str) -> tuple:
    """
    Run one pytest test in-process under a fresh Coverage() instance.
    Returns (passed: bool, output: str, executed_lines: dict[filepath, set[int]])
    """
    import coverage as coverage_module
    import io
    from contextlib import redirect_stdout, redirect_stderr

    # Ensure source_dir is on sys.path so the module under test can be imported.
    # This is critical when called from a server process whose cwd is elsewhere.
    _abs_src = os.path.abspath(source_dir)
    _injected = _abs_src not in sys.path
    if _injected:
        sys.path.insert(0, _abs_src)

    cov = coverage_module.Coverage(source=[_abs_src], data_file=None)
    cov.start()

    buf = io.StringIO()
    try:
        import pytest
        with redirect_stdout(buf), redirect_stderr(buf):
            ret = pytest.main(
                [test_id, "-x", "-q", "--tb=short",
                 "--no-header", "-p", "no:warnings"],
                plugins=[]
            )
        passed = (ret == 0)
    except Exception as e:
        passed = False
        buf.write(f"\nException during test: {e}")
    finally:
        cov.stop()

    executed = defaultdict(set)
    try:
        for filepath in cov.get_data().measured_files():
            lines = cov.get_data().lines(filepath)
            if lines:
                executed[filepath] = set(lines)
    except Exception:
        pass

    # Remove the injected path so we don't permanently pollute sys.path
    if _injected and _abs_src in sys.path:
        sys.path.remove(_abs_src)

    return passed, buf.getvalue(), dict(executed)


# ── Core SBFL logic ───────────────────────────────────────────────────

def run_tests_with_coverage(test_list: list, source_dir: str = ".") -> tuple:
    """
    Run each test sequentially in-process under coverage.
    Returns (coverage_per_test, fail_output).
    """
    coverage_per_test = {}
    fail_output = ""

    for test in test_list:
        passed, output, executed = _run_single_test_with_coverage(test, source_dir)
        print(f"  {'PASS' if passed else 'FAIL'} {test}")
        if not passed:
            fail_output += f"\n=== {test} ===\n{output}"
        coverage_per_test[test] = {"passed": passed, "coverage": executed}

    return coverage_per_test, fail_output


def detect_target_file(coverage_per_test: dict, hint: str = "buggy_code.py") -> str:
    """
    Find the source file most likely to contain the bug.
    1. Use hint if any covered filepath ends with it.
    2. Otherwise pick the non-test .py file covered by the most failing tests.
    """
    all_files: set = set()
    for info in coverage_per_test.values():
        all_files.update(info["coverage"].keys())

    for fp in all_files:
        if fp.endswith(hint) or os.path.basename(fp) == hint:
            return hint

    print(f"WARN  Hint file '{hint}' was NOT found in coverage data.")
    print(f"   Covered files: {sorted(os.path.basename(f) for f in all_files)}")
    print(f"   SBFL results may target the wrong file — verify your test imports.")

    scores: dict = defaultdict(int)
    for info in coverage_per_test.values():
        if not info["passed"]:
            for fp in info["coverage"]:
                basename = os.path.basename(fp)
                if not basename.startswith("test_") and basename.endswith(".py"):
                    scores[basename] += 1

    if scores:
        best = max(scores, key=scores.get)
        print(f"SBFL Auto-detected target file: '{best}' (using as fallback)")
        return best

    print(f"WARN  Could not auto-detect target — falling back to hint '{hint}' (may be inaccurate)")
    return hint


def build_spectrum(coverage_per_test: dict, target_file: str = "buggy_code.py") -> dict:
    """
    Build {line: [pass_count, fail_count]} spectrum for the target file only.
    """
    target_file = detect_target_file(coverage_per_test, hint=target_file)

    all_covered = {os.path.basename(fp)
                   for info in coverage_per_test.values()
                   for fp in info["coverage"]}
    print(f"FILES Files in coverage data: {sorted(all_covered)}")

    spectrum = defaultdict(lambda: [0, 0])

    for test, info in coverage_per_test.items():
        for filepath, lines in info["coverage"].items():
            if not (filepath.endswith(target_file) or
                    os.path.basename(filepath) == target_file):
                continue
            for line in lines:
                if info["passed"]:
                    spectrum[line][0] += 1
                else:
                    spectrum[line][1] += 1

    if not spectrum:
        print(f"WARN  Spectrum empty — '{target_file}' not matched in coverage paths.")
    else:
        print(f"OK Spectrum built: {len(spectrum)} lines tracked for '{target_file}'")

    return spectrum


# ── Public API ────────────────────────────────────────────────────────

def collect_spectrum(
    test_list: list = None,
    target_file: str = "buggy_code.py",
    source_dir: str = ".",
    test_path: str = None,
) -> tuple:
    """
    Full SBFL data collection pipeline.
    Returns (spectrum_dict, fail_output_string).

    Args:
        test_list:   Explicit list of pytest test IDs (collected from test_path if None).
        target_file: Filename hint for the buggy source file.
        source_dir:  Directory to scope coverage measurement.
        test_path:   Path to the test file (used for collection when test_list is None).
    """
    if test_list is None:
        collect_from = test_path or "."
        test_list = get_all_tests(collect_from)

    if not test_list:
        print("WARN  No tests found.")
        return {}, ""

    coverage_data, fail_output = run_tests_with_coverage(test_list, source_dir=source_dir)
    spectrum = build_spectrum(coverage_data, target_file=target_file)

    return spectrum, fail_output
