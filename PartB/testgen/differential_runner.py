"""
Differential test execution — the SWE-bench-correct way to verify a test.

A test is only useful if it:
  1. PASSES on patched code   (confirms the fix works)
  2. FAILS on unpatched code  (proves the test would have caught the bug)

If a test passes on BOTH versions, it doesn't distinguish the bug from the fix.
If it fails on both, it's broken.

Usage (programmatic):
    from differential_runner import run_differential

    verdict = run_differential(
        test_code     = "def test_foo(): ...",
        source_code   = current_broken_source,
        patched_code  = fixed_source,
        import_as     = "requests.auth",
    )
    print(verdict)
    # {"verdict": "resolved", "patched_pass": True, "unpatched_fail": True}

verdict values:
  "resolved"         — passes on patch, fails without (ideal)
  "test_broken"      — fails on BOTH (broken test, discard)
  "not_distinguishing" — passes on BOTH (doesn't catch the bug)
  "error"            — could not run
"""
from __future__ import annotations

import os
import subprocess
import tempfile
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def _run_test_against_source(
    test_code: str,
    source_code: str,
    source_filename: str,
    timeout: int = 60,
) -> tuple[bool, str]:
    """
    Run a test against a specific version of the source code.
    Returns (passed, output_snippet).
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        src_file = tmp_dir / source_filename
        src_file.write_text(source_code, encoding="utf-8")
        test_file = tmp_dir / f"test_{source_filename}"
        test_file.write_text(test_code, encoding="utf-8")

        try:
            r = subprocess.run(
                ["python", "-m", "pytest", str(test_file),
                 "--tb=short", "-q", "--no-header"],
                cwd=tmp_dir,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            passed = r.returncode == 0
            output = (r.stdout + r.stderr)[-500:]
            return passed, output
        except subprocess.TimeoutExpired:
            return False, "TIMEOUT"
        except Exception as exc:
            return False, str(exc)


def run_differential(
    test_code: str,
    patched_source: str,
    unpatched_source: str,
    source_filename: str,
    timeout: int = 60,
) -> dict:
    """
    Run the test against both patched and unpatched versions.

    Returns a verdict dict:
    {
      "verdict":        "resolved" | "test_broken" | "not_distinguishing" | "error",
      "patched_pass":   bool,
      "unpatched_fail": bool,
      "patched_output": str,
      "unpatched_output": str,
    }
    """
    result = {
        "verdict":          "error",
        "patched_pass":     False,
        "unpatched_fail":   False,
        "patched_output":   "",
        "unpatched_output": "",
    }

    try:
        patched_pass, patched_out = _run_test_against_source(
            test_code, patched_source, source_filename, timeout=timeout
        )
        unpatched_pass, unpatched_out = _run_test_against_source(
            test_code, unpatched_source, source_filename, timeout=timeout
        )

        result["patched_pass"]     = patched_pass
        result["unpatched_fail"]   = not unpatched_pass
        result["patched_output"]   = patched_out
        result["unpatched_output"] = unpatched_out

        if patched_pass and not unpatched_pass:
            result["verdict"] = "resolved"
        elif not patched_pass and not unpatched_pass:
            result["verdict"] = "test_broken"
        elif patched_pass and unpatched_pass:
            result["verdict"] = "not_distinguishing"
        else:
            result["verdict"] = "only_fails_on_patch"  # unusual

    except Exception as exc:
        logger.warning("[differential] Error: %s", exc)
        result["error"] = str(exc)

    return result


def filter_tests_by_differential(
    test_codes: list[str],
    patched_source: str,
    unpatched_source: str,
    source_filename: str,
) -> list[tuple[str, dict]]:
    """
    Run differential on a list of test snippets.
    Returns list of (test_code, verdict) for tests that are "resolved".
    """
    results = []
    for test in test_codes:
        verdict = run_differential(test, patched_source, unpatched_source, source_filename)
        if verdict["verdict"] == "resolved":
            results.append((test, verdict))
        else:
            logger.info("[differential] Test filtered out: verdict=%s", verdict["verdict"])
    return results
