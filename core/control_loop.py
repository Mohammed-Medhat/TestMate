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

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.sbfl_localiser import rank_suspicious_lines
from core.run_and_collect import collect_spectrum
from core.model_runner import run_testmate

# ── Paths ─────────────────────────────────────────────────────────────
BUGGY_FILE    = PROJECT_ROOT / "buggy_code.py"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
RESULTS_FILE  = PROJECT_ROOT / "local_run_results.json"

ARTIFACTS_DIR.mkdir(exist_ok=True)


# ── Phase 1: Filter pipeline results ─────────────────────────────────

def load_and_filter_candidates(json_path):
    """Filter test results to find valid repair candidates (logic errors only)."""
    path = Path(json_path)
    if not path.exists():
        print(f"⚠️  Results file not found: {json_path}")
        return []

    with open(path, "r", encoding="utf-8") as f:
        try:
            results = json.load(f)
        except json.JSONDecodeError:
            print("❌ Invalid JSON in results file.")
            return []

    valid_candidates = []
    print("\n🔍 Analysing pipeline test results...")

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
            print(f"✅ Queued {instance_id}: logic failure detected")
            valid_candidates.append({
                "id":          instance_id,
                "problem":     entry.get("problem", ""),
                "failed_test": entry.get("generated_test", ""),
                "error":       error_msg,
            })

    print(f"📊 {len(valid_candidates)} candidates ready for APR.\n")
    return valid_candidates


# ── Phase 2: Test execution & code extraction ─────────────────────────

def run_tests() -> tuple:
    try:
        result = subprocess.run(
            ["pytest", "-v"],
            capture_output=True, text=True, timeout=120
        )
        return result.returncode == 0, result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        print("⏰ Tests timed out after 120s — likely an infinite loop in buggy code.")
        return False, "TimeoutExpired: test suite exceeded 120 seconds."


def extract_snippet(suspicious_lines: list, window: int = 5) -> str:
    """Extract clean code context around suspicious lines (no line numbers)."""
    code = BUGGY_FILE.read_text(encoding="utf-8").splitlines()
    indices = set()
    for ln in suspicious_lines:
        for offset in range(-window, window + 1):
            idx = ln - 1 + offset
            if 0 <= idx < len(code):
                indices.add(idx)
    return "\n".join(code[i] for i in sorted(indices))


def extract_full_code() -> str:
    return BUGGY_FILE.read_text(encoding="utf-8")


# ── Phase 3: AST smart patching ───────────────────────────────────────

def apply_patch(new_code: str) -> bool:
    """
    Safely replace only the modified functions using AST surgery.
    Reads original BEFORE the try-block so rollback is always possible.
    """
    original_source = BUGGY_FILE.read_text(encoding="utf-8")  # always available for rollback
    try:
        # Truncate if the model echoed the prompt back
        for marker in ("FULL BUGGY CODE", "SUSPICIOUS LINES", "SUSPICIOUS SNIPPET", "TRACE:", "ISSUE:"):
            if marker in new_code:
                new_code = new_code[:new_code.index(marker)]
                print(f"  Truncated at prompt-echo marker: '{marker}'")

        # Extract pure Python from LLM output
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

        clean_code = clean_code.replace("⚠️", "").replace("⚠", "").strip()
        clean_code = re.sub(r"^\s*\d+:\s?", "", clean_code, flags=re.MULTILINE)

        if clean_code.count("def") == 0:
            print("❌ Patch rejected: no function definitions found in model output.")
            return False

        # Parse LLM output
        new_ast  = ast.parse(clean_code)
        new_fns  = {n.name: n for n in new_ast.body if isinstance(n, ast.FunctionDef)}
        if not new_fns:
            print("⚠️  Patch rejected: no functions parsed from model output.")
            return False

        # Check if original file has a syntax error
        try:
            orig_ast = ast.parse(original_source)
        except SyntaxError:
            # Original has syntax error — replace entire file with model output
            print("🔧 Original has syntax error — replacing full file with model output")
            BUGGY_FILE.write_text(clean_code, encoding="utf-8")
            print("✅ Full file replaced.")
            return True

        # Normal AST patch: replace only matching functions
        modified = False
        for i, node in enumerate(orig_ast.body):
            if isinstance(node, ast.FunctionDef) and node.name in new_fns:
                orig_ast.body[i] = new_fns[node.name]
                modified = True
                print(f"🔧 Replaced function: '{node.name}'")

        if not modified:
            print("⚠️  Patch rejected: function names from model didn't match the file.")
            return False

        BUGGY_FILE.write_text(ast.unparse(orig_ast), encoding="utf-8")
        print("✅ AST patch applied.")
        return True

    except SyntaxError as e:
        print(f"❌ Patch rejected: invalid Python syntax — {e}")
        BUGGY_FILE.write_text(original_source, encoding="utf-8")
        return False
    except Exception as e:
        print(f"❌ AST patching failed: {e}")
        BUGGY_FILE.write_text(original_source, encoding="utf-8")
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


def save_model_output(code: str, attempt: int):
    path = ARTIFACTS_DIR / f"model_output_attempt_{attempt}.py"
    path.write_text(code, encoding="utf-8")


# ── Phase 5: Main repair loop ─────────────────────────────────────────

def repair_loop(max_attempts: int = 3, top_k: int = 5) -> bool:
    print("=" * 70)
    print("🚀 TestMate + SBFL + AST Repair Loop")
    print("=" * 70)

    # Snapshot original once — used for final restore if all attempts fail
    original_snapshot = BUGGY_FILE.read_text(encoding="utf-8")

    for attempt in range(1, max_attempts + 1):
        print(f"\n🔄 Repair Attempt {attempt}/{max_attempts}")

        # Snapshot at the START of each attempt so every retry begins from
        # the same clean baseline (avoids building on a half-broken patch
        # left by the previous attempt's failed apply_patch call).
        attempt_snapshot = BUGGY_FILE.read_text(encoding="utf-8")

        # Step 1: Run tests
        passed, test_output = run_tests()
        if passed:
            print("\n" + "=" * 70)
            print("🎉 SUCCESS! All tests passed.")
            print("=" * 70)
            return True

        print("❌ Tests failed. Running SBFL analysis...")

        # Step 2: SBFL
        spectrum, fail_output = collect_spectrum()
        ranked = rank_suspicious_lines(spectrum)

        full_code = extract_full_code()

        if not ranked:
            # SBFL failed — likely a syntax error, get exact message
            print("⚠️  No SBFL data — checking for syntax error...")
            import ast as _ast
            syntax_msg = ""
            try:
                _ast.parse(full_code)
            except SyntaxError as e:
                syntax_msg = f"SyntaxError at line {e.lineno}: {e.msg}\n    {e.text or ''}"
                print(f"🔍 Detected: {syntax_msg}")

            prompt = (
                f"ISSUE: Fix the Python syntax error in this code.\n\n"
                f"ERROR:\n{syntax_msg or fail_output[:400] or 'SyntaxError detected'}\n\n"
                f"BUGGY CODE:\n{full_code}\n\n"
                f"Return ONLY the complete corrected Python file with all functions."
            )
        else:
            suspicious_lines = [ln for ln, _ in ranked[:top_k]]
            print(f"🔍 Top suspicious lines: {suspicious_lines}")

            # Step 3: Build prompt
            snippet = extract_snippet(suspicious_lines, window=5)
            prompt  = build_repair_prompt(fail_output, suspicious_lines, snippet, full_code)

        if attempt > 1 and fail_output:
            prompt += (
                f"\n\n[SYSTEM WARNING - ATTEMPT {attempt}]:\n"
                f"Your previous fix FAILED and produced this error:\n{fail_output[:300]}\n"
                f"Please analyze the root cause deeply and try a COMPLETELY DIFFERENT logical approach. "
                f"Do not repeat the previous code."
            )

        # Step 4: Call model
        print(f"🤖 Calling TestMate model (Attempt {attempt})...")
        try:
            new_code = run_testmate(prompt, attempt=attempt)
        except Exception as e:
            print(f"❌ Model call failed: {e}")
            return False
        # ────────────────────────────────────────────────────────────

        save_model_output(new_code, attempt)

        # Step 5: Apply patch
        print("🔧 Applying smart patch...")
        if not apply_patch(new_code):
            print("❌ Patch failed. Restoring attempt snapshot and trying next attempt...")
            BUGGY_FILE.write_text(attempt_snapshot, encoding="utf-8")
            continue

    print("\n" + "=" * 70)
    print(f"❌ FAILED: max attempts ({max_attempts}) reached.")
    print("🔄 Restoring original file...")
    BUGGY_FILE.write_text(original_snapshot, encoding="utf-8")
    print("✅ Original file restored.")
    print("=" * 70)
    return False

# ── Entry point ───────────────────────────────────────────────────────

if __name__ == "__main__":
    success = repair_loop(max_attempts=3, top_k=5)
    exit(0 if success else 1)