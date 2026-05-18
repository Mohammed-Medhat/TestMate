"""
LangChain Tool Wrappers for TestMate
=====================================
Wraps TestMate's existing functions (pytest, coverage, mutation, RAG, quality check)
as LangChain Tools so they can be invoked by the LangGraph agent.
"""

import os
import json
from typing import Optional
from langchain_core.tools import tool


# ---------------------------------------------------------------------------
# Tool 1: Run pytest
# ---------------------------------------------------------------------------
@tool
def run_pytest_tool(test_file: str, target_file: str = "") -> str:
    """Run pytest on a generated test file. Returns (passed: bool, output: str) as JSON."""
    from main import run_pytest
    try:
        passed, output = run_pytest(test_file, target_file or None)
        return json.dumps({"passed": passed, "output": output[:2000]})
    except Exception as e:
        return json.dumps({"passed": False, "output": f"Error: {str(e)}"})


# ---------------------------------------------------------------------------
# Tool 2: Run pytest with coverage
# ---------------------------------------------------------------------------
@tool
def run_coverage_tool(test_file: str, source_file: str, cov_module: str = "") -> str:
    """Run pytest with line + branch coverage. Returns coverage metrics as JSON."""
    from main import run_pytest_with_coverage
    try:
        passed, output, line_cov, branch_cov, covered_lines = run_pytest_with_coverage(
            test_file, source_file, cov_module or None
        )
        return json.dumps({
            "passed": passed,
            "line_coverage": line_cov,
            "branch_coverage": branch_cov,
            "covered_lines": sorted(list(covered_lines))[:50],
            "output": output[:1500],
        })
    except Exception as e:
        return json.dumps({"passed": False, "error": str(e)})


# ---------------------------------------------------------------------------
# Tool 3: Run mutation testing
# ---------------------------------------------------------------------------
@tool
def run_mutation_tool(target_file: str, test_file: str) -> str:
    """Run lightweight mutation testing. Returns (all_killed: bool, details: str) as JSON."""
    from main import run_mutation_testing
    try:
        all_killed, feedback = run_mutation_testing(target_file, test_file)
        return json.dumps({"all_killed": all_killed, "feedback": feedback})
    except Exception as e:
        return json.dumps({"all_killed": False, "feedback": f"Error: {str(e)}"})


# ---------------------------------------------------------------------------
# Tool 4: Quality check (syntax + composite score)
# ---------------------------------------------------------------------------
@tool
def quality_check_tool(test_code: str) -> str:
    """Run quick quality check + composite score on generated test code. Returns JSON."""
    from main import quick_quality_check, compute_composite_score
    try:
        ok, reason = quick_quality_check(test_code)
        composite = compute_composite_score(test_code, per_test=True)
        return json.dumps({
            "syntax_ok": ok,
            "syntax_reason": reason,
            "composite_score": composite["composite"],
            "breakdown": composite,
        })
    except Exception as e:
        return json.dumps({"syntax_ok": False, "syntax_reason": str(e)})


# ---------------------------------------------------------------------------
# Tool 5: Probe a method's runtime return value
# ---------------------------------------------------------------------------
@tool
def probe_method_tool(
    actual_import: str, cls_name: str, method_name: str,
    init_params: str = "[]", source_code: str = ""
) -> str:
    """Probe a method at runtime to discover its real return value."""
    from main import probe_method
    import json as _json
    try:
        params = _json.loads(init_params)
    except Exception:
        params = []
    result = probe_method(actual_import, cls_name, method_name, params, source_code)
    return result if result else "probe_failed"


# ---------------------------------------------------------------------------
# Tool 6: RAG retrieval (good + bad examples)
# ---------------------------------------------------------------------------
@tool
def rag_retrieve_tool(method_name: str, class_name: str = "", source_file: str = "") -> str:
    """Retrieve similar good examples and bad examples from RAG store."""
    from rag_store import retrieve_similar, retrieve_bad_examples
    try:
        good = retrieve_similar(method_name, class_name, top_k=2, source_file=source_file)
        bad = retrieve_bad_examples(method_name, class_name)
        return json.dumps({
            "good_examples": [
                {"signature": g["target_signature"],
                 "test_code": g["test_code"][:500],
                 "quality": g["quality_score"]}
                for g in good
            ],
            "bad_examples": [
                {"reason": b["failure_reason"],
                 "category": b.get("failure_category", "unknown")}
                for b in bad
            ],
        })
    except Exception as e:
        return json.dumps({"good_examples": [], "bad_examples": [], "error": str(e)})


# ---------------------------------------------------------------------------
# Convenience: get all tools as a list
# ---------------------------------------------------------------------------
def get_all_tools():
    """Return all TestMate tools for use in a LangGraph agent."""
    return [
        run_pytest_tool,
        run_coverage_tool,
        run_mutation_tool,
        quality_check_tool,
        probe_method_tool,
        rag_retrieve_tool,
    ]
