"""
run_and_collect.py — Per-test coverage runner for SBFL.
Uses the Python coverage API directly (no subprocess) to avoid
working-directory and path issues when locating the target file.
"""
import os
import sys
import json
from pathlib import Path
from collections import defaultdict

# ── Helpers ───────────────────────────────────────────────────────────

def get_all_tests(test_dir: str = ".") -> list:
    """Collect all pytest test IDs without running them."""
    import subprocess
    result = subprocess.run(
        ["pytest", "--collect-only", "-q", test_dir],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    return [
        line.strip()
        for line in result.stdout.splitlines()
        if line.strip() and "::" in line
    ]


def _run_single_test_with_coverage(test_id: str, source_dir: str) -> tuple:
    """
    Run one pytest test under Python coverage API.
    Returns (passed: bool, output: str, executed_lines: dict[filepath, set[int]])
    """
    import coverage as coverage_module
    import io
    from contextlib import redirect_stdout, redirect_stderr

    cov = coverage_module.Coverage(source=[source_dir], data_file=None)
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

    # Extract per-file executed lines
    executed = defaultdict(set)
    try:
        file_reporters = cov.get_data().measured_files()
        for filepath in file_reporters:
            lines = cov.get_data().lines(filepath)
            if lines:
                executed[filepath] = set(lines)
    except Exception:
        pass

    return passed, buf.getvalue(), dict(executed)


# ── Core SBFL logic ───────────────────────────────────────────────────

def run_tests_with_coverage(test_list: list, source_dir: str = ".") -> tuple:
    """Run each test individually under coverage. Returns (coverage_per_test, fail_output)."""
    coverage_per_test = {}
    fail_output = ""

    for test in test_list:
        print(f"  ▶ Running {test}")
        passed, output, executed = _run_single_test_with_coverage(test, source_dir)

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

    # Try hint first
    for fp in all_files:
        if fp.endswith(hint) or os.path.basename(fp) == hint:
            return hint

    # Auto-detect by failure correlation
    scores: dict = defaultdict(int)
    for info in coverage_per_test.values():
        if not info["passed"]:
            for fp in info["coverage"]:
                basename = os.path.basename(fp)
                if not basename.startswith("test_") and basename.endswith(".py"):
                    scores[basename] += 1

    if scores:
        best = max(scores, key=scores.get)
        print(f"🔍 Auto-detected target file: '{best}' (hint '{hint}' not found)")
        return best

    print(f"⚠️  Could not auto-detect target — falling back to '{hint}'")
    return hint


def build_spectrum(coverage_per_test: dict, target_file: str = "buggy_code.py") -> dict:
    """
    Build {line: [pass_count, fail_count]} spectrum for the target file only.
    """
    target_file = detect_target_file(coverage_per_test, hint=target_file)

    all_covered = {os.path.basename(fp)
                   for info in coverage_per_test.values()
                   for fp in info["coverage"]}
    print(f"📂 Files in coverage data: {sorted(all_covered)}")

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
        print(f"⚠️  Spectrum empty — '{target_file}' not matched in coverage paths.")
    else:
        print(f"✅ Spectrum built: {len(spectrum)} lines tracked for '{target_file}'")

    return spectrum


# ── Public API ────────────────────────────────────────────────────────

def collect_spectrum(test_list: list = None,
                     target_file: str = "buggy_code.py",
                     source_dir: str = ".") -> tuple:
    """
    Full SBFL data collection pipeline.
    Returns (spectrum_dict, fail_output_string).
    """
    if test_list is None:
        test_list = get_all_tests()

    if not test_list:
        print("⚠️  No tests found.")
        return {}, ""

    coverage_data, fail_output = run_tests_with_coverage(test_list, source_dir=source_dir)
    spectrum = build_spectrum(coverage_data, target_file=target_file)

    with open("spectrum.json", "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in spectrum.items()}, f, indent=2)

    return spectrum, fail_output