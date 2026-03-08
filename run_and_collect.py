import subprocess
import json
import os
import math
from collections import defaultdict

PROJECT_ROOT = os.getcwd()
COVERAGE_FILE = ".coverage"
COVERAGE_JSON = "coverage.json"


# -------------------------------------------------
# Helpers
# -------------------------------------------------

def run_command(cmd):
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    return result.returncode == 0, result.stdout + result.stderr


def get_all_tests():
    result = subprocess.run(
        ["pytest", "--collect-only", "-q"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def extract_executed_lines():
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


# -------------------------------------------------
# Core SBFL logic
# -------------------------------------------------

def run_tests_with_coverage(test_list):
    coverage_per_test = {}
    fail_output = ""

    for test in test_list:
        print(f"  ▶ Running {test}")

        if os.path.exists(COVERAGE_FILE):
            os.remove(COVERAGE_FILE)
        if os.path.exists(COVERAGE_JSON):
            os.remove(COVERAGE_JSON)

        passed, output = run_command(
            ["coverage", "run", "-m", "pytest", test]
        )

        if not passed:
            fail_output += f"\n=== {test} ===\n{output}"

        # Try to generate coverage.json
        run_command(["coverage", "json", "-o", COVERAGE_JSON])

        executed = extract_executed_lines()

        coverage_per_test[test] = {
            "passed": passed,
            "coverage": executed
        }

    return coverage_per_test, fail_output


def build_spectrum(coverage_per_test):
    spectrum = defaultdict(lambda: [0, 0])  # [passed, failed]

    for test, info in coverage_per_test.items():
        for file, lines in info["coverage"].items():
            for line in lines:
                if info["passed"]:
                    spectrum[line][0] += 1
                else:
                    spectrum[line][1] += 1

    return spectrum


# -------------------------------------------------
# Function expected by control_loop.py
# -------------------------------------------------

def collect_spectrum(test_list=None):
    if test_list is None:
        test_list = get_all_tests()

    coverage_data, fail_output = run_tests_with_coverage(test_list)
    spectrum = build_spectrum(coverage_data)

    # Save spectrum for inspection
    with open("spectrum.json", "w", encoding="utf-8") as f:
        json.dump(
            {str(k): v for k, v in spectrum.items()},
            f,
            indent=2
        )

    return spectrum, fail_output
