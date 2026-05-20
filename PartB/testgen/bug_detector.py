"""
bug_detector.py — Confidence-scored bug detection for PartB.

Replaces the 3-variant oracle (which was circular: same model confirming same model's tests).
Uses 3 independent evidence sources:

  Gate 3a — Docstring Oracle:
      AST-extract the function's docstring, ask LLM if the test assertion is
      consistent with the documented spec.  Catches hallucinated expected values.

  Gate 3b — Existing Test Cross-Reference:
      Find real tests already in the repo, run them, see if the same function
      is covered and passing.  If existing tests pass → code probably works.

  Oracle 4 — Direct Execution:
      Actually run the function with the call extracted from the failing test.
      Compare real output to the test's asserted value.
      Try auto-mocking (requests, DB, etc.) on failure.
      This is model-independent ground truth.

Confidence scoring (0–100):
  Default: 50
  +25 if docstring agrees with test expectation
  -25 if docstring contradicts test expectation
  +20 if existing tests also fail on same target  (independent confirmation)
  -20 if existing tests pass   (function works for known inputs)
  +30 if real output ≠ test's expected  (code is returning the wrong thing)
  -30 if real output == test's expected (code is right → test is wrong)

Verdict thresholds:
  ≥ 70 → 'confirmed'   → send to PartC for repair
  40–69 → 'suspected'  → flag for human review
  < 40  → 'discarded'  → test is wrong, ignore
"""
from __future__ import annotations

import ast
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ── How long to let a function execute before we give up ─────────────────────
_EXEC_TIMEOUT = 10   # seconds
_PYTEST_TIMEOUT = 30  # seconds for existing-test runs


# ═══════════════════════════════════════════════════════════════════════
# Section 1 — AST helpers: extract call + assertion from a test function
# ═══════════════════════════════════════════════════════════════════════

def _extract_calls_and_assertions(test_code: str, func_name: str) -> list[dict]:
    """
    Parse a test function and return a list of dicts:
        {
          "call_src": "func_name(arg1, arg2)",   # source of the call
          "args_src":  "(arg1, arg2)",
          "expected":  <Python value> | None,     # RHS of assert ==
          "raises":    "ValueError" | None,       # for pytest.raises blocks
          "assert_src": "assert result == 5",
        }
    Handles common patterns:
      - assert f(a,b) == expected
      - result = f(a,b); assert result == expected
      - with pytest.raises(Exc): f(a,b)
    Returns [] if nothing extractable.
    """
    results = []
    try:
        tree = ast.parse(textwrap.dedent(test_code))
    except SyntaxError:
        return []

    # Collect all assert nodes and pytest.raises context managers
    for node in ast.walk(tree):

        # ── assert f(...) == val ──────────────────────────────────────
        if isinstance(node, ast.Assert):
            cmp = node.test
            if isinstance(cmp, ast.Compare) and len(cmp.ops) == 1 and isinstance(cmp.ops[0], ast.Eq):
                left, right = cmp.left, cmp.comparators[0]
                call_node, expected_node = None, None
                # Pattern: assert call == val  or  assert val == call
                if isinstance(left, ast.Call) and _call_matches(left, func_name):
                    call_node, expected_node = left, right
                elif isinstance(right, ast.Call) and _call_matches(right, func_name):
                    call_node, expected_node = right, left
                if call_node:
                    try:
                        expected_val = ast.literal_eval(expected_node)
                    except Exception:
                        expected_val = None
                    results.append({
                        "call_src":   ast.unparse(call_node),
                        "args_src":   _args_src(call_node),
                        "expected":   expected_val,
                        "raises":     None,
                        "assert_src": ast.unparse(node),
                    })

        # ── with pytest.raises(Exc): f(...) ──────────────────────────
        if isinstance(node, ast.With):
            exc_name = _extract_raises_exc(node)
            if exc_name:
                for item in ast.walk(node):
                    if isinstance(item, ast.Call) and _call_matches(item, func_name):
                        results.append({
                            "call_src":   ast.unparse(item),
                            "args_src":   _args_src(item),
                            "expected":   None,
                            "raises":     exc_name,
                            "assert_src": f"pytest.raises({exc_name})",
                        })
                        break

    return results


def _call_matches(node: ast.Call, func_name: str) -> bool:
    """Return True if the Call node looks like it calls func_name."""
    if isinstance(node.func, ast.Name):
        return node.func.id == func_name
    if isinstance(node.func, ast.Attribute):
        return node.func.attr == func_name
    return False


def _args_src(call_node: ast.Call) -> str:
    parts = [ast.unparse(a) for a in call_node.args]
    parts += [f"{kw.arg}={ast.unparse(kw.value)}" for kw in call_node.keywords]
    return "(" + ", ".join(parts) + ")"


def _extract_raises_exc(with_node: ast.With) -> Optional[str]:
    for item in with_node.items:
        ctx = item.context_expr
        if isinstance(ctx, ast.Call):
            fn = ctx.func
            name = fn.attr if isinstance(fn, ast.Attribute) else (fn.id if isinstance(fn, ast.Name) else "")
            if name == "raises" and ctx.args:
                try:
                    return ast.unparse(ctx.args[0])
                except Exception:
                    pass
    return None


# ═══════════════════════════════════════════════════════════════════════
# Section 2 — Oracle 4: Direct execution
# ═══════════════════════════════════════════════════════════════════════

_MOCK_PREAMBLE = """
import sys, types, unittest.mock as _mock

# ── Auto-mock common external dependencies ────────────────────────────
_MOCK_MODULES = [
    "requests", "httpx", "aiohttp",
    "django", "flask", "fastapi",
    "sqlalchemy", "pymongo", "redis", "psycopg2",
    "boto3", "botocore",
    "celery", "kafka",
]
for _mod in _MOCK_MODULES:
    if _mod not in sys.modules:
        _parts = _mod.split(".")
        _pkg = types.ModuleType(_parts[0])
        _pkg.__path__ = []
        sys.modules[_parts[0]] = _pkg
        for _depth in range(1, len(_parts)):
            _sub = ".".join(_parts[: _depth + 1])
            _m = types.ModuleType(_sub)
            _m.__path__ = []
            sys.modules[_sub] = _m
            setattr(sys.modules[".".join(_parts[:_depth])], _parts[_depth], _m)
"""

_RUNNER_TEMPLATE = """
import sys, json, traceback

{preamble}

sys.path.insert(0, {source_dir!r})
try:
    from {module_name} import *
except Exception as e:
    print(json.dumps({{"error": f"import failed: {{e}}", "actual": None, "raised": None}}))
    sys.exit(0)

try:
    _result = {call_src}
    print(json.dumps({{"actual": repr(_result), "error": None, "raised": None}}))
except Exception as _e:
    print(json.dumps({{"actual": None, "error": None, "raised": type(_e).__name__}}))
"""


def execute_function_oracle(
    target_file: str,
    module_name: str,
    call_src: str,
    expected: Any,
    raises: Optional[str],
) -> dict:
    """
    Execute `call_src` by importing `module_name` from `target_file`'s directory.
    Returns:
        {
          "executed":               bool,
          "actual":                 str | None,   # repr of real return value
          "expected":               str | None,   # repr of what test expected
          "actual_matches_expected":bool | None,
          "raises_matches":         bool | None,
          "skip_reason":            str | None,
        }
    """
    source_dir = str(Path(target_file).parent)
    script = _RUNNER_TEMPLATE.format(
        preamble=_MOCK_PREAMBLE,
        source_dir=source_dir,
        module_name=module_name,
        call_src=call_src,
    )

    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as tf:
            tf.write(script)
            script_path = tf.name

        proc = subprocess.run(
            [sys.executable, script_path],
            capture_output=True, text=True, timeout=_EXEC_TIMEOUT,
        )
        raw = proc.stdout.strip()
        if not raw:
            return {"executed": False, "skip_reason": f"no output (stderr: {proc.stderr[:200]})"}

        data = json.loads(raw)

        if data.get("error"):
            return {"executed": False, "skip_reason": data["error"]}

        actual_repr = data.get("actual")
        raised_exc  = data.get("raised")

        # pytest.raises case
        if raises:
            raises_matches = (raised_exc == raises) if raised_exc else False
            return {
                "executed":               True,
                "actual":                 raised_exc,
                "expected":               raises,
                "actual_matches_expected": raises_matches,
                "raises_matches":         raises_matches,
                "skip_reason":            None,
            }

        # Normal return-value case
        # Compare repr of actual to repr of expected
        expected_repr = repr(expected) if expected is not None else None
        matches: Optional[bool] = None
        if actual_repr is not None and expected_repr is not None:
            matches = (actual_repr.strip() == expected_repr.strip())

        return {
            "executed":               True,
            "actual":                 actual_repr,
            "expected":               expected_repr,
            "actual_matches_expected": matches,
            "raises_matches":         None,
            "skip_reason":            None,
        }

    except subprocess.TimeoutExpired:
        return {"executed": False, "skip_reason": "execution timed out"}
    except json.JSONDecodeError as e:
        return {"executed": False, "skip_reason": f"bad JSON from runner: {e}"}
    except Exception as e:
        return {"executed": False, "skip_reason": str(e)}
    finally:
        try:
            os.unlink(script_path)
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════════
# Section 3 — Gate 3a: Docstring Oracle
# ═══════════════════════════════════════════════════════════════════════

def _extract_docstring(func_source: str) -> Optional[str]:
    """Extract the docstring from the first function in func_source."""
    try:
        tree = ast.parse(textwrap.dedent(func_source))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                return ast.get_docstring(node)
    except Exception:
        pass
    return None


def check_docstring_oracle(
    func_source: str,
    test_code: str,
    model: Any,
    tokenizer: Any,
) -> dict:
    """
    Ask the LLM: given the function's docstring, is the test's expected
    behaviour consistent with the spec?

    Returns:
        {
          "has_docstring": bool,
          "consistent":    bool | None,  # None if no docstring
          "confidence":    float,        # 0-1
          "reasoning":     str,
        }
    """
    docstring = _extract_docstring(func_source)
    if not docstring or len(docstring.strip()) < 10:
        return {"has_docstring": False, "consistent": None, "confidence": 0.5,
                "reasoning": "No docstring found — cannot evaluate spec consistency."}

    prompt = (
        f"You are a test quality auditor. Given the function docstring and a failing test, "
        f"decide: is the test's expected behaviour consistent with what the docstring says?\n\n"
        f"FUNCTION DOCSTRING:\n{docstring}\n\n"
        f"FAILING TEST CODE:\n{test_code}\n\n"
        f"Answer with exactly one line: "
        f"CONSISTENT if the test expects what the docstring describes, "
        f"INCONSISTENT if the test expects something that contradicts the docstring, "
        f"UNCLEAR if the docstring doesn't say enough to decide.\n"
        f"Then on the next line, give one short sentence explaining why."
    )

    try:
        messages = [
            {"role": "system", "content": "You are a precise code quality auditor. Be concise."},
            {"role": "user", "content": prompt},
        ]
        import torch
        chat = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(chat, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=80, do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
        response = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        lines = [l.strip() for l in response.splitlines() if l.strip()]
        verdict_line = lines[0].upper() if lines else ""
        reasoning = lines[1] if len(lines) > 1 else response

        if "INCONSISTENT" in verdict_line:
            return {"has_docstring": True, "consistent": False, "confidence": 0.75, "reasoning": reasoning}
        elif "CONSISTENT" in verdict_line:
            return {"has_docstring": True, "consistent": True, "confidence": 0.75, "reasoning": reasoning}
        else:
            return {"has_docstring": True, "consistent": None, "confidence": 0.5, "reasoning": reasoning}

    except Exception as e:
        logger.warning("Docstring oracle LLM call failed: %s", e)
        return {"has_docstring": True, "consistent": None, "confidence": 0.5,
                "reasoning": f"LLM call failed: {e}"}


# ═══════════════════════════════════════════════════════════════════════
# Section 4 — Gate 3b: Existing Test Cross-Reference
# ═══════════════════════════════════════════════════════════════════════

def _find_existing_tests(target_file: str) -> list[Path]:
    """Walk up from target_file to find test_*.py files covering it."""
    src_stem = Path(target_file).stem
    repo_root = Path(target_file).parent
    # Walk up at most 3 levels to find the project root
    for _ in range(3):
        if (repo_root / "setup.py").exists() or (repo_root / "pyproject.toml").exists():
            break
        repo_root = repo_root.parent

    patterns = [f"test_{src_stem}.py", f"{src_stem}_test.py", f"tests/test_{src_stem}.py"]
    found = []
    for pat in patterns:
        p = repo_root / pat
        if p.exists():
            found.append(p)

    # Also glob for any test files in a tests/ subdirectory
    for tests_dir in (repo_root / "tests", repo_root / "test"):
        if tests_dir.is_dir():
            for tf in tests_dir.glob(f"*{src_stem}*"):
                if tf.suffix == ".py" and tf not in found:
                    found.append(tf)
    return found


def check_existing_tests(target_file: str, func_name: str) -> dict:
    """
    Find real existing tests in the repo, run them, detect if they pass for
    the target function.

    Returns:
        {
          "has_existing":     bool,
          "existing_pass":    bool | None,
          "files_checked":    list[str],
          "relevant_found":   bool,
        }
    """
    test_files = _find_existing_tests(target_file)
    if not test_files:
        return {"has_existing": False, "existing_pass": None,
                "files_checked": [], "relevant_found": False}

    # Filter: only run tests that actually reference this function name
    relevant = [tf for tf in test_files
                if func_name in tf.read_text(encoding="utf-8", errors="ignore")]

    if not relevant:
        return {"has_existing": True, "existing_pass": None,
                "files_checked": [str(tf) for tf in test_files],
                "relevant_found": False}

    # Run relevant test files with pytest -k func_name
    cwd = str(Path(target_file).parent)
    cmd = [
        sys.executable, "-m", "pytest",
        *[str(tf) for tf in relevant],
        "-k", func_name,
        "-q", "--tb=no", "--no-header",
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=_PYTEST_TIMEOUT, cwd=cwd,
        )
        existing_pass = (result.returncode == 0)
        return {
            "has_existing":   True,
            "existing_pass":  existing_pass,
            "files_checked":  [str(tf) for tf in relevant],
            "relevant_found": True,
        }
    except subprocess.TimeoutExpired:
        return {"has_existing": True, "existing_pass": None,
                "files_checked": [str(tf) for tf in relevant],
                "relevant_found": True}
    except Exception as e:
        logger.warning("Existing test run failed: %s", e)
        return {"has_existing": True, "existing_pass": None,
                "files_checked": [str(tf) for tf in relevant],
                "relevant_found": True}


# ═══════════════════════════════════════════════════════════════════════
# Section 5 — Main: compute_bug_confidence()
# ═══════════════════════════════════════════════════════════════════════

def compute_bug_confidence(
    test_code: str,
    target_file: str,
    module_name: str,
    func_name: str,
    source_code: str,
    model: Any,
    tokenizer: Any,
    failure_logs: list,
) -> dict:
    """
    Run all detection gates and return a confidence-scored verdict.

    Args:
        test_code:    The generated pytest code that is failing.
        target_file:  Absolute path to the source file being tested.
        module_name:  Import module name (e.g. 'requests.auth').
        func_name:    Function/method name being tested.
        source_code:  Full source of the target file.
        model:        Pre-loaded Qwen model (for docstring oracle).
        tokenizer:    Pre-loaded tokenizer.
        failure_logs: List of pytest failure output strings.

    Returns:
        {
          "score":     int,          # 0–100
          "verdict":   str,          # 'confirmed' | 'suspected' | 'discarded'
          "evidence":  dict,         # all gate results for logging
          "func_name": str,
          "test_code": str,
          "target_file": str,
          "failure_logs": list,
        }
    """
    score = 50  # neutral starting point
    evidence: dict = {}

    # ── Gate 3a: Docstring Oracle ─────────────────────────────────────
    try:
        doc_result = check_docstring_oracle(source_code, test_code, model, tokenizer)
        evidence["docstring"] = doc_result
        if doc_result["consistent"] is True:
            score += 25  # test agrees with spec → test is probably right
            logger.debug("[BugDetect] +25 docstring consistent")
        elif doc_result["consistent"] is False:
            score -= 25  # test contradicts spec → test is probably wrong
            logger.debug("[BugDetect] -25 docstring inconsistent")
    except Exception as e:
        evidence["docstring"] = {"error": str(e)}
        logger.warning("[BugDetect] Gate 3a failed: %s", e)

    # ── Gate 3b: Existing Test Cross-Reference ────────────────────────
    try:
        ext_result = check_existing_tests(target_file, func_name)
        evidence["existing_tests"] = ext_result
        if ext_result["relevant_found"] and ext_result["existing_pass"] is not None:
            if ext_result["existing_pass"]:
                score -= 20  # existing tests pass → function works → new test probably wrong
                logger.debug("[BugDetect] -20 existing tests pass")
            else:
                score += 20  # existing tests also fail → real bug
                logger.debug("[BugDetect] +20 existing tests fail")
    except Exception as e:
        evidence["existing_tests"] = {"error": str(e)}
        logger.warning("[BugDetect] Gate 3b failed: %s", e)

    # ── Oracle 4: Direct Execution ────────────────────────────────────
    try:
        calls = _extract_calls_and_assertions(test_code, func_name)
        if calls:
            # Use the first extractable call
            call_info = calls[0]
            exec_result = execute_function_oracle(
                target_file=target_file,
                module_name=module_name,
                call_src=call_info["call_src"],
                expected=call_info["expected"],
                raises=call_info["raises"],
            )
            evidence["execution"] = {**exec_result, "call_src": call_info["call_src"]}

            if exec_result["executed"]:
                matches = exec_result.get("actual_matches_expected")
                if matches is False:
                    # Function returned different from what test expects → code is wrong
                    score += 30
                    logger.debug("[BugDetect] +30 function output ≠ test expectation")
                elif matches is True:
                    # Function returned exactly what test expects → test infrastructure is wrong
                    score -= 30
                    logger.debug("[BugDetect] -30 function output == test expectation (test bug)")
            else:
                evidence["execution"]["skip_reason"] = exec_result.get("skip_reason")
                logger.debug("[BugDetect] Oracle 4 skipped: %s", exec_result.get("skip_reason"))
        else:
            evidence["execution"] = {"executed": False, "skip_reason": "no extractable call found in test"}
    except Exception as e:
        evidence["execution"] = {"executed": False, "skip_reason": str(e)}
        logger.warning("[BugDetect] Oracle 4 failed: %s", e)

    # ── Clamp + verdict ───────────────────────────────────────────────
    score = max(0, min(100, score))
    if score >= 70:
        verdict = "confirmed"
    elif score >= 40:
        verdict = "suspected"
    else:
        verdict = "discarded"

    logger.info(
        "[BugDetect] %s → score=%d verdict=%s",
        func_name, score, verdict,
    )

    return {
        "score":        score,
        "verdict":      verdict,
        "evidence":     evidence,
        "func_name":    func_name,
        "test_code":    test_code,
        "target_file":  target_file,
        "failure_logs": failure_logs,
    }


# ═══════════════════════════════════════════════════════════════════════
# Section 6 — Batch audit: replace post_run_audit's 3-variant oracle
# ═══════════════════════════════════════════════════════════════════════

def run_confidence_audit(
    suspicious_tests: list[dict],
    model: Any,
    tokenizer: Any,
) -> list[dict]:
    """
    Replace the old 3-variant oracle with confidence-scored detection.

    Args:
        suspicious_tests: List of dicts from autonomous_loop with keys:
            target_name, target_file, method_name, cls_name, source_code,
            test_code, failure_logs, actual_import, test_file
        model, tokenizer: Pre-loaded Qwen.

    Returns:
        List of confirmed bug dicts (verdict == 'confirmed' or 'suspected').
        Each dict has all evidence attached.
    """
    if not suspicious_tests:
        return []

    print(f"\n🔬 Confidence Audit: {len(suspicious_tests)} suspicious test(s)...")
    confirmed: list[dict] = []

    for case in suspicious_tests:
        func_name = case.get("method_name") or case.get("target_name", "unknown")
        target_file = case.get("target_file", "")
        test_code = case.get("test_code", "")
        source_code = case.get("source_code", "")
        actual_import = case.get("actual_import") or Path(target_file).stem
        failure_logs = case.get("failure_logs", [])

        print(f"  🔍 Auditing: {func_name}")
        t0 = time.time()

        result = compute_bug_confidence(
            test_code=test_code,
            target_file=target_file,
            module_name=actual_import,
            func_name=func_name,
            source_code=source_code,
            model=model,
            tokenizer=tokenizer,
            failure_logs=failure_logs,
        )

        elapsed = round(time.time() - t0, 1)
        icon = {"confirmed": "🐛", "suspected": "⚠️", "discarded": "✅"}[result["verdict"]]
        print(f"  {icon} {func_name}: score={result['score']} verdict={result['verdict']} ({elapsed}s)")

        # Log evidence details
        doc = result["evidence"].get("docstring", {})
        if doc.get("has_docstring"):
            print(f"     Docstring: {'✅ consistent' if doc.get('consistent') else '❌ contradicts' if doc.get('consistent') is False else '❓ unclear'} — {doc.get('reasoning', '')[:80]}")

        ext = result["evidence"].get("existing_tests", {})
        if ext.get("relevant_found"):
            print(f"     Existing tests: {'✅ pass' if ext.get('existing_pass') else '❌ fail'}")

        exe = result["evidence"].get("execution", {})
        if exe.get("executed"):
            print(f"     Execution: actual={exe.get('actual')} expected={exe.get('expected')} match={exe.get('actual_matches_expected')}")
        elif exe.get("skip_reason"):
            print(f"     Execution skipped: {exe['skip_reason']}")

        if result["verdict"] in ("confirmed", "suspected"):
            confirmed.append({**case, **result})

    print(f"\n✅ Confidence audit complete: {len(confirmed)}/{len(suspicious_tests)} passed ({sum(1 for c in confirmed if c['verdict']=='confirmed')} confirmed, {sum(1 for c in confirmed if c['verdict']=='suspected')} suspected)\n")
    return confirmed
