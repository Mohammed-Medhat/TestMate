"""
run_and_collect.py — Per-test coverage runner for SBFL.
Runs each test individually, records which lines in the target file
were executed, and builds a pass/fail spectrum.
"""
import subprocess
import json
import os
from collections import defaultdict

COVERAGE_FILE = ".coverage"
COVERAGE_JSON = "coverage.json"


# ── Helpers ───────────────────────────────────────────────────────────

def run_command(cmd: list) -> tuple:
    result = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    return result.returncode == 0, result.stdout + result.stderr


def get_all_tests() -> list:
    result = subprocess.run(
        ["pytest", "--collect-only", "-q"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    return [
        line.strip()
        for line in result.stdout.splitlines()
        if line.strip() and "::" in line
    ]


def extract_executed_lines() -> dict:
    """Parse coverage.json and return {filename: set_of_executed_lines}."""
    if not os.path.exists(COVERAGE_JSON):
        return {}
    with open(COVERAGE_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    executed = defaultdict(set)
    for filename, info in data.get("files", {}).items():
        if filename.endswith(".py"):
            for line in info.get("executed_lines", []):
                executed[filename].add(line)
    return executed


# ── Core SBFL logic ───────────────────────────────────────────────────

def run_tests_with_coverage(test_list: list) -> tuple:
    """Run each test under coverage and return per-test coverage data."""
    coverage_per_test = {}
    fail_output = ""

    for test in test_list:
        print(f"  ▶ Running {test}")

        for path in (COVERAGE_FILE, COVERAGE_JSON):
            if os.path.exists(path):
                os.remove(path)

        passed, output = run_command(["coverage", "run", "-m", "pytest", test])

        if not passed:
            fail_output += f"\n=== {test} ===\n{output}"

        run_command(["coverage", "json", "-o", COVERAGE_JSON])
        executed = extract_executed_lines()

        coverage_per_test[test] = {"passed": passed, "coverage": executed}

    return coverage_per_test, fail_output


def build_spectrum(coverage_per_test: dict, target_file: str = "buggy_code.py") -> dict:
    """
    Build a {line: [passed_count, failed_count]} spectrum for the target file only.

    BUG FIX: previously all files were merged by line number alone, which caused
    lines from test files to collide with buggy_code.py lines, zeroing out
    failed counts and making all Ochiai scores = 0.0.
    """
    spectrum = defaultdict(lambda: [0, 0])

    for test, info in coverage_per_test.items():
        for filepath, lines in info["coverage"].items():
            # Only track coverage for the file we're trying to repair
            if not filepath.endswith(target_file):
                continue
            for line in lines:
                if info["passed"]:
                    spectrum[line][0] += 1
                else:
                    spectrum[line][1] += 1

    return spectrum


# ── Public API ────────────────────────────────────────────────────────

def collect_spectrum(test_list: list = None, target_file: str = "buggy_code.py") -> tuple:
    """
    Full SBFL data collection pipeline.
    Returns (spectrum_dict, fail_output_string).
    """
    if test_list is None:
        test_list = get_all_tests()

    coverage_data, fail_output = run_tests_with_coverage(test_list)
    spectrum = build_spectrum(coverage_data, target_file=target_file)

    # Save snapshot for inspection
    with open("spectrum.json", "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in spectrum.items()}, f, indent=2)

    return spectrum, fail_output
