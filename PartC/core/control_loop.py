"""
control_loop.py — End-to-End APR Pipeline (CLI mode).
Runs SBFL → LLM prompt → AST patch in a loop until tests pass or max attempts reached.
"""
import sys
import os
import subprocess
import json
import ast
import re
from pathlib import Path
from typing import Any, Callable, Optional

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.sbfl_localiser import rank_suspicious_lines
from core.run_and_collect import collect_spectrum
from core.model_runner import run_testmate
from config import REPAIR_TEMPERATURES, MAX_REPAIR_ATTEMPTS

# ── Default paths (CLI backwards-compat) ─────────────────────────────
_DEFAULT_BUGGY    = PROJECT_ROOT / "buggy_code.py"
_DEFAULT_ARTIFACTS = PROJECT_ROOT / "artifacts"


# ── Phase 1: Filter pipeline results ─────────────────────────────────

def load_and_filter_candidates(json_path):
    """Filter test results to find valid repair candidates (logic errors only)."""
    path = Path(json_path)
    if not path.exists():
        print(f"WARN  Results file not found: {json_path}")
        return []

    with open(path, "r", encoding="utf-8") as f:
        try:
            results = json.load(f)
        except json.JSONDecodeError:
            print("FAIL Invalid JSON in results file.")
            return []

    valid_candidates = []
    print("\nSBFL Analysing pipeline test results...")

    BAD_TEST_INDICATORS = ["SyntaxError", "IndentationError", "missing", "NameError", "ImportError"]
    LOGIC_ERRORS        = {"AssertionError", "ValueError", "RuntimeError", "AttributeError", "TypeError"}

    for entry in results:
        instance_id = entry.get("instance_id", "unknown")
        metrics     = entry.get("metrics", {})
        error_msg   = metrics.get("execution_error", "")

        if metrics.get("execution_success", False):
            continue

        if any(ind in error_msg for ind in BAD_TEST_INDICATORS):
            print(f"⏭️  Skipping {instance_id}: bad test generation")
            continue

        if any(err in error_msg for err in LOGIC_ERRORS):
            print(f"OK Queued {instance_id}: logic failure detected")
            valid_candidates.append({
                "id":          instance_id,
                "problem":     entry.get("problem", ""),
                "failed_test": entry.get("generated_test", ""),
                "error":       error_msg,
            })

    print(f"📊 {len(valid_candidates)} candidates ready for APR.\n")
    return valid_candidates


# ── Phase 2: Test execution & code extraction ─────────────────────────

def run_tests(test_file: Path, cwd: Path) -> tuple:
    try:
        result = subprocess.run(
            ["pytest", str(test_file), "-v", "--tb=short", "--no-header"],
            capture_output=True, text=True, timeout=120, cwd=str(cwd)
        )
        return result.returncode == 0, result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        print("TIMEOUT Tests timed out after 120s — likely an infinite loop in buggy code.")
        return False, "TimeoutExpired: test suite exceeded 120 seconds."


def extract_snippet(source_file: Path, suspicious_lines: list, window: int = 5) -> str:
    """Extract clean code context around suspicious lines (no line numbers)."""
    code = source_file.read_text(encoding="utf-8").splitlines()
    indices = set()
    for ln in suspicious_lines:
        for offset in range(-window, window + 1):
            idx = ln - 1 + offset
            if 0 <= idx < len(code):
                indices.add(idx)
    return "\n".join(code[i] for i in sorted(indices))


def extract_full_code(source_file: Path) -> str:
    return source_file.read_text(encoding="utf-8")


# ── Phase 3: AST smart patching ───────────────────────────────────────

def apply_patch(source_file: Path, new_code: str) -> bool:
    """
    Safely replace only the modified functions using AST surgery.
    Reads original BEFORE the try-block so rollback is always possible.
    """
    original_source = source_file.read_text(encoding="utf-8")
    try:
        for marker in ("FULL BUGGY CODE", "SUSPICIOUS LINES", "SUSPICIOUS SNIPPET", "TRACE:", "ISSUE:"):
            if marker in new_code:
                new_code = new_code[:new_code.index(marker)]
                print(f"  Truncated at prompt-echo marker: '{marker}'")

        match = re.search(r"```(?:python)?(.*?)```", new_code, re.DOTALL | re.IGNORECASE)
        if match:
            clean_code = match.group(1)
        else:
            lines = new_code.split("\n")
            start = 0
            for i, line in enumerate(lines):
                if line.strip().startswith(("def ", "class ", "@")):
                    start = i
                    break
            clean_code = "\n".join(lines[start:])

        clean_code = clean_code.replace("WARN", "").replace("WARN", "").strip()
        clean_code = re.sub(r"^\s*\d+:\s?", "", clean_code, flags=re.MULTILINE)

        if clean_code.count("def") == 0:
            print("FAIL Patch rejected: no function definitions found in model output.")
            return False

        new_ast  = ast.parse(clean_code)
        new_fns  = {n.name: n for n in new_ast.body if isinstance(n, ast.FunctionDef)}
        if not new_fns:
            print("WARN  Patch rejected: no functions parsed from model output.")
            return False

        try:
            orig_ast = ast.parse(original_source)
        except SyntaxError:
            print("PATCH Original has syntax error — replacing full file with model output")
            source_file.write_text(clean_code, encoding="utf-8")
            print("OK Full file replaced.")
            return True

        modified = False
        for i, node in enumerate(orig_ast.body):
            if isinstance(node, ast.FunctionDef) and node.name in new_fns:
                orig_ast.body[i] = new_fns[node.name]
                modified = True
                print(f"PATCH Replaced function: '{node.name}'")

        if not modified:
            print("WARN  Patch rejected: function names from model didn't match the file.")
            return False

        source_file.write_text(ast.unparse(orig_ast), encoding="utf-8")
        print("OK AST patch applied.")
        return True

    except SyntaxError as e:
        print(f"FAIL Patch rejected: invalid Python syntax — {e}")
        source_file.write_text(original_source, encoding="utf-8")
        return False
    except Exception as e:
        print(f"FAIL AST patching failed: {e}")
        source_file.write_text(original_source, encoding="utf-8")
        return False


# ── Phase 4: Prompt building & artifact saving ────────────────────────

def build_repair_prompt(fail_output: str, suspicious_lines: list, snippet: str, full_code: str) -> str:
    failure_lines = [
        l.strip() for l in fail_output.split("\n")
        if "E   " in l or "FAILED" in l or "Error" in l
    ]
    trace_text = "\n".join(failure_lines[:4]) or "AssertionError: Logic verification failed."
    lines_str  = ", ".join(str(ln) for ln in suspicious_lines)

    return (
        f"ISSUE: Fix the logic bug causing the test failure.\n\n"
        f"TRACE:\n{trace_text}\n\n"
        f"SUSPICIOUS LINES (SBFL): {lines_str}\n\n"
        f"SUSPICIOUS SNIPPET:\n{snippet}\n\n"
        f"BUGGY CODE:\n{full_code}"
    )


def save_model_output(artifacts_dir: Path, code: str, attempt: int):
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    path = artifacts_dir / f"model_output_attempt_{attempt}.py"
    path.write_text(code, encoding="utf-8")


# ── Phase 5: Main repair loop ─────────────────────────────────────────

def repair_loop(
    source_file: Optional[Path] = None,
    test_file: Optional[Path] = None,
    artifacts_dir: Optional[Path] = None,
    max_attempts: int = MAX_REPAIR_ATTEMPTS,
    top_k: int = 5,
    model: Any = None,
    tokenizer: Any = None,
    log_callback: Optional[Callable] = None,
) -> tuple:
    """
    Run the APR loop.

    Args:
        source_file:   Path to the buggy Python file (modified in workspace copy).
        test_file:     Path to the pytest test suite.
        artifacts_dir: Where to save per-attempt model outputs.
        max_attempts:  Max repair iterations (default 3).
        top_k:         Top suspicious lines to include in prompt.
        model/tokenizer: Pre-loaded Qwen model (loaded here if None).
        log_callback:  Optional callback(event_type, *args) for SSE streaming.

    Returns:
        (success: bool, attempts_log: list[dict], suspicious: list[dict])
    """
    source_file   = Path(source_file)   if source_file   else _DEFAULT_BUGGY
    test_file     = Path(test_file)     if test_file     else PROJECT_ROOT / "test_logic.py"
    artifacts_dir = Path(artifacts_dir) if artifacts_dir else _DEFAULT_ARTIFACTS
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    work_dir = source_file.parent

    def _log(msg: str):
        print(msg)
        if log_callback:
            log_callback("log", "info", msg)

    _log("=" * 70)
    _log("START TestMate APR — SBFL + AST Repair Loop")
    _log("=" * 70)

    original_snapshot = source_file.read_text(encoding="utf-8")
    attempts_log: list[dict] = []
    last_suspicious: list[dict] = []

    for attempt in range(1, max_attempts + 1):
        _log(f"\nRELOAD Repair Attempt {attempt}/{max_attempts}")
        attempt_snapshot = source_file.read_text(encoding="utf-8")

        passed, test_output = run_tests(test_file, cwd=work_dir)
        if passed:
            _log("\nSUCCESS SUCCESS! All tests passed.")
            return True, attempts_log, last_suspicious

        _log("FAIL Tests failed. Running SBFL analysis...")

        spectrum, fail_output = collect_spectrum(
            test_list=None,
            target_file=source_file.name,
            source_dir=str(work_dir),
            test_path=str(test_file),
        )
        ranked = rank_suspicious_lines(spectrum)
        full_code = extract_full_code(source_file)

        if not ranked:
            _log("WARN  No SBFL data — checking for syntax error...")
            syntax_msg = ""
            try:
                ast.parse(full_code)
            except SyntaxError as e:
                syntax_msg = f"SyntaxError at line {e.lineno}: {e.msg}\n    {e.text or ''}"
                _log(f"SBFL Detected: {syntax_msg}")

            prompt = (
                f"ISSUE: Fix the Python syntax error in this code.\n\n"
                f"ERROR:\n{syntax_msg or fail_output[:400] or 'SyntaxError detected'}\n\n"
                f"BUGGY CODE:\n{full_code}\n\n"
                f"Return ONLY the complete corrected Python file with all functions."
            )
        else:
            suspicious_lines = [ln for ln, _ in ranked[:top_k]]
            code_lines = full_code.splitlines()
            last_suspicious = [
                {"line": ln, "score": round(sc, 3),
                 "code": code_lines[ln - 1] if 0 < ln <= len(code_lines) else ""}
                for ln, sc in ranked[:top_k]
            ]
            _log(f"SBFL Top suspicious lines: {suspicious_lines}")
            if log_callback:
                log_callback("sbfl_result", last_suspicious)

            snippet = extract_snippet(source_file, suspicious_lines, window=5)
            prompt  = build_repair_prompt(fail_output, suspicious_lines, snippet, full_code)

        if attempt > 1 and fail_output:
            prompt += (
                f"\n\n[SYSTEM WARNING - ATTEMPT {attempt}]:\n"
                f"Your previous fix FAILED and produced this error:\n{fail_output[:300]}\n"
                f"Please analyze the root cause deeply and try a COMPLETELY DIFFERENT logical approach. "
                f"Do not repeat the previous code."
            )

        _log(f"MODEL Calling TestMate model (Attempt {attempt})...")
        try:
            new_code = run_testmate(prompt, attempt=attempt, model=model, tokenizer=tokenizer)
        except Exception as e:
            _log(f"FAIL Model call failed: {e}")
            attempts_log.append({"n": attempt, "status": "fail", "patch": "Model error", "result": str(e)})
            return False, attempts_log, last_suspicious

        save_model_output(artifacts_dir, new_code, attempt)

        _log("PATCH Applying smart patch...")
        if not apply_patch(source_file, new_code):
            _log("FAIL Patch failed. Restoring attempt snapshot...")
            source_file.write_text(attempt_snapshot, encoding="utf-8")
            attempts_log.append({"n": attempt, "status": "fail", "patch": "AST patch failed", "result": "Patch rejected"})
            continue

        # Verify
        passed2, _ = run_tests(test_file, cwd=work_dir)
        if passed2:
            patched_code = source_file.read_text(encoding="utf-8")
            attempts_log.append({"n": attempt, "status": "success", "patch": patched_code, "result": "All tests passed OK"})
            if log_callback:
                log_callback("attempt_complete", {"n": attempt, "status": "success",
                                                   "patched": patched_code, "sbfl": last_suspicious})
            _log(f"\nSUCCESS SUCCESS on attempt {attempt}!")
            return True, attempts_log, last_suspicious
        else:
            _log(f"WARN  Patch applied but tests still failing")
            attempts_log.append({"n": attempt, "status": "fail", "patch": "Applied, tests still failing", "result": "Tests still failing"})
            if log_callback:
                log_callback("attempt_complete", {"n": attempt, "status": "fail",
                                                   "patched": source_file.read_text(encoding="utf-8"),
                                                   "sbfl": last_suspicious})

    _log(f"\nFAIL FAILED: max attempts ({max_attempts}) reached.")
    _log("RELOAD Restoring original file...")
    source_file.write_text(original_snapshot, encoding="utf-8")
    return False, attempts_log, last_suspicious


# ── Entry point ───────────────────────────────────────────────────────

if __name__ == "__main__":
    success, _, __ = repair_loop(max_attempts=MAX_REPAIR_ATTEMPTS, top_k=5)
    exit(0 if success else 1)
