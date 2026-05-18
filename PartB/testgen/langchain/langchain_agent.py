"""
LangGraph Agent for TestMate — with Human-in-the-Loop
======================================================
Replaces the monolithic autonomous_loop() with a LangGraph StateGraph.

Human-in-the-loop (HITL):
  - After ALL tests are generated, the agent pauses for human review.
  - User can: ✅ Approve all | ✏️ Edit tests | ❌ Reject & regenerate | ⏭️ Skip review
  - User can also toggle "auto-approve" to skip the review entirely.

Architecture:
  [START] → [init] → [generate_all_tests] → [human_review] → [finalize] → [END]
                                                  ↑
                                            INTERRUPT HERE
                                       (if auto_approve is off)
"""

import os
import sys
import re
import ast
import json
import time
import threading
from typing import TypedDict, Annotated, Literal, Optional, Any
from operator import add

from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver

# ---------------------------------------------------------------------------
# Agent State
# ---------------------------------------------------------------------------

class TestGenState(TypedDict):
    """Full state for the test generation agent."""

    # ── Inputs ──
    target_file: str
    import_path: str  # e.g. 'requests.auth' or ''
    source_code: str
    deep_scan: bool
    max_retries: int
    auto_approve: bool  # HITL toggle: if True, skip human review

    # ── Computed once during init ──
    targets: list  # [(cls_name, method_name), ...]
    module_name: str
    actual_import: str
    import_statement: str
    test_file: str
    ctx: dict  # AST context
    framework_header: str
    repo_ctx: dict

    # ── Loop state ──
    accumulated_code: str
    accumulated_tests: int
    covered_summary: list
    failed_targets: list
    current_target_idx: int

    # ── Results ──
    final_coverage: float
    final_branch_coverage: float
    mutation_score: float
    composite_score: dict
    test_results_summary: str

    # ── HITL ──
    human_decision: str  # 'approve', 'edit', 'reject', 'skip', or ''
    human_edited_code: str  # if user edited the test code
    review_requested: bool  # True when waiting for human input

    # ── Streaming ──
    log_messages: list  # log messages for SSE


# ---------------------------------------------------------------------------
# Helper: emit log (thread-safe, works with GUI's log_queue)
# ---------------------------------------------------------------------------

_log_callback = None


def set_log_callback(callback):
    """Set a callback function for log messages (e.g., GUI's emit_log)."""
    global _log_callback
    _log_callback = callback


def _log(msg: str, level: str = "info"):
    """Log a message. If callback is set, use it; otherwise print."""
    if _log_callback:
        _log_callback(msg, level)
    else:
        print(msg)


# ---------------------------------------------------------------------------
# Node: Initialize
# ---------------------------------------------------------------------------

def node_init(state: TestGenState) -> dict:
    """Initialize the agent: parse AST, extract targets, build imports."""

    # Import main.py functions (lazy import to avoid circular deps)
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from main import (
        extract_context_from_code, build_call_graph,
        get_method_line_count, get_stateful_setup_chain,
        detect_repo_context, detect_framework_setup,
        probe_all_targets,
    )
    from rag_store import init_db

    init_db()

    target_file = state["target_file"]
    import_path = state.get("import_path", "")
    deep_scan = state.get("deep_scan", False)

    with open(target_file, "r") as f:
        source_code = f.read()

    target_name = os.path.basename(target_file)
    module_name = target_name.replace('.py', '')
    actual_import = import_path if import_path else module_name

    # Detect private functions
    private_names = []
    try:
        tree = ast.parse(source_code)
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name.startswith("_") and not node.name.startswith("__"):
                    private_names.append(node.name)
    except SyntaxError:
        pass

    # Build import statement
    if module_name[0].isdigit():
        abs_target = os.path.abspath(target_file)
        import_statement = (
            f"import importlib.util, sys, os\n"
            f'_spec = importlib.util.spec_from_file_location('
            f'"{module_name}", r"{abs_target}")\n'
            f"_mod = importlib.util.module_from_spec(_spec)\n"
            f'sys.modules["{module_name}"] = _mod\n'
            f"_spec.loader.exec_module(_mod)\n"
            f"globals().update({{k: v for k, v in _mod.__dict__.items() "
            f"if not k.startswith('__')}})"
        )
    elif private_names:
        private_imports = ", ".join(private_names)
        import_statement = (
            f"from {actual_import} import *\n"
            f"from {actual_import} import {private_imports}"
        )
    else:
        import_statement = f"from {actual_import} import *"

    # Test file location
    if import_path:
        gen_tests_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "generated_tests")
        os.makedirs(gen_tests_dir, exist_ok=True)
        test_file = os.path.join(gen_tests_dir, f"test_{target_name}")
    else:
        test_file = os.path.join(
            os.path.dirname(os.path.abspath(target_file)),
            f"test_{target_name}"
        )

    # Extract AST context
    ctx = extract_context_from_code(source_code)

    # Build targets list
    targets = []
    for cls_info in ctx["classes"]:
        cls_name = cls_info["name"]
        if not cls_info["methods"]:
            targets.append([cls_name, "__init__"])
        else:
            for method in cls_info["methods"]:
                targets.append([cls_name, method])
    try:
        tree = ast.parse(source_code)
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                targets.append([None, node.name])
    except SyntaxError:
        pass

    # Sort by dependency order
    G = build_call_graph(source_code)

    def _dep_sort_key(target):
        cls, method = target
        label = f"{cls}.{method}" if cls else method
        if method == "__init__":
            return (0, 0)
        try:
            deps = get_stateful_setup_chain(G, label)
            n_deps = len(deps)
        except Exception:
            n_deps = 0
        lines = get_method_line_count(source_code, cls, method)
        return (n_deps, -lines)

    targets.sort(key=_dep_sort_key)

    # Detect framework
    framework_header = detect_framework_setup(source_code)

    # Detect repo context
    repo_ctx = detect_repo_context(target_file, deep_scan=deep_scan,
                                    source_files=[target_file])

    # Build initial accumulated code
    accumulated_code = (
        f"{framework_header}"
        f"import pytest\n"
        f"from unittest.mock import MagicMock, patch\n"
        f"{import_statement}\n"
    )

    _log(f"📄 Target: {target_name}")
    _log(f"🎯 Found {len(targets)} test targets")
    _log(f"🔀 CFG paths: {len(ctx.get('cfg_paths', []))}")

    return {
        "source_code": source_code,
        "targets": targets,
        "module_name": module_name,
        "actual_import": actual_import,
        "import_statement": import_statement,
        "test_file": test_file,
        "ctx": ctx,
        "framework_header": framework_header,
        "repo_ctx": repo_ctx,
        "accumulated_code": accumulated_code,
        "accumulated_tests": 0,
        "covered_summary": [],
        "failed_targets": [],
        "current_target_idx": 0,
        "review_requested": False,
        "human_decision": "",
        "human_edited_code": "",
    }


# ---------------------------------------------------------------------------
# Node: Generate All Tests (calls existing autonomous_loop logic)
# ---------------------------------------------------------------------------

def node_generate_all_tests(state: TestGenState) -> dict:
    """Run the core test generation loop for all targets.

    This delegates to the existing autonomous_loop() in main.py, which handles:
    - Per-method prompt building
    - LLM generation via your Qwen2.5 + LoRA model
    - Quality gates, RAG, self-correction retries
    - Mutation testing
    """
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from main import load_model, autonomous_loop

    target_file = state["target_file"]
    import_path = state.get("import_path", "")
    deep_scan = state.get("deep_scan", False)
    max_retries = state.get("max_retries", 3)

    _log("🧠 Loading model...")
    model, tokenizer = load_model()
    _log("✅ Model loaded")

    _log("🚀 Starting test generation loop...")
    success = autonomous_loop(
        model, tokenizer, target_file,
        import_path=import_path or None,
        deep_scan=deep_scan,
        max_retries=max_retries,
    )

    # Read the generated test file
    test_file = state["test_file"]
    accumulated_code = ""
    if os.path.exists(test_file):
        with open(test_file, "r") as f:
            accumulated_code = f.read()

    n_tests = len(re.findall(r'def test_\w+', accumulated_code))

    _log(f"{'='*50}")
    _log(f"🏁 Generation complete: {n_tests} tests generated")

    return {
        "accumulated_code": accumulated_code,
        "accumulated_tests": n_tests,
        "test_results_summary": f"Generated {n_tests} tests (success={success})",
    }


# ---------------------------------------------------------------------------
# Node: Human Review (HITL INTERRUPT)
# ---------------------------------------------------------------------------

def node_human_review(state: TestGenState) -> dict:
    """Pause for human review if auto_approve is off.

    Uses LangGraph's interrupt() to pause the graph and wait for user input.
    The GUI sends the user's decision via Command(resume=...).
    """
    auto_approve = state.get("auto_approve", False)
    accumulated_code = state.get("accumulated_code", "")
    n_tests = state.get("accumulated_tests", 0)

    if auto_approve:
        _log("⏭️  Auto-approve enabled — skipping human review")
        return {
            "human_decision": "approve",
            "review_requested": False,
        }

    if n_tests == 0:
        _log("⚠️  No tests to review — skipping")
        return {
            "human_decision": "skip",
            "review_requested": False,
        }

    # ── INTERRUPT: Wait for human input ──
    _log("⏸️  Pausing for human review...")
    _log(f"📝 {n_tests} tests ready for review")

    # This call PAUSES the graph until Command(resume=...) is called
    human_input = interrupt({
        "type": "review_request",
        "test_code": accumulated_code,
        "num_tests": n_tests,
        "message": "Review the generated tests. You can approve, edit, or reject them.",
    })

    # human_input is the value passed via Command(resume=value)
    decision = human_input.get("decision", "approve")
    edited_code = human_input.get("edited_code", "")

    _log(f"👤 Human decision: {decision}")

    if decision == "edit" and edited_code:
        _log("✏️  Using human-edited test code")
        return {
            "human_decision": "edit",
            "human_edited_code": edited_code,
            "accumulated_code": edited_code,
            "review_requested": False,
        }

    return {
        "human_decision": decision,
        "review_requested": False,
    }


# ---------------------------------------------------------------------------
# Node: Finalize (coverage + mutation on final tests)
# ---------------------------------------------------------------------------

def node_finalize(state: TestGenState) -> dict:
    """Run final coverage + mutation testing and save results."""
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from main import (
        run_pytest_with_coverage, run_mutation_testing,
        compute_composite_score, save_to_flywheel,
    )

    decision = state.get("human_decision", "approve")
    accumulated_code = state.get("accumulated_code", "")
    target_file = state["target_file"]
    test_file = state["test_file"]
    actual_import = state.get("actual_import", "")

    if decision == "reject":
        _log("❌ Tests rejected by human — discarding")
        # Clean up test file
        if os.path.exists(test_file):
            os.remove(test_file)
        return {
            "test_results_summary": "Tests rejected by human reviewer",
            "final_coverage": 0.0,
            "final_branch_coverage": 0.0,
            "mutation_score": 0.0,
            "composite_score": {},
        }

    if decision == "skip" or not accumulated_code.strip():
        _log("⏭️  Skipping finalization")
        return {
            "test_results_summary": "Skipped",
            "final_coverage": 0.0,
            "final_branch_coverage": 0.0,
            "mutation_score": 0.0,
            "composite_score": {},
        }

    # Write final test code (may have been edited by human)
    with open(test_file, "w") as f:
        f.write(accumulated_code)

    n_tests = len(re.findall(r'def test_\w+', accumulated_code))
    _log(f"📊 Running final evaluation on {n_tests} tests...")

    # Coverage
    try:
        _, _, line_cov, branch_cov, _ = run_pytest_with_coverage(
            test_file, target_file, actual_import, need_lines=False
        )
    except Exception as e:
        _log(f"⚠️  Coverage failed: {e}")
        line_cov, branch_cov = 0.0, 0.0

    # Mutation testing
    mutation_score = 0.0
    try:
        mutants_killed, mutation_feedback = run_mutation_testing(target_file, test_file)
        if mutants_killed:
            mutation_score = 100.0
        else:
            m = re.search(r'(\d+)%', mutation_feedback)
            if m:
                mutation_score = float(m.group(1))
        _log(f"🧬 Mutation: {mutation_feedback}")
    except Exception as e:
        _log(f"⚠️  Mutation testing failed: {e}")

    # Composite score
    composite = compute_composite_score(
        test_code=accumulated_code,
        line_coverage_pct=line_cov,
        branch_coverage_pct=branch_cov,
        mutation_score=mutation_score,
        per_test=False,
    )

    _log(f"📈 Final composite: {composite['composite']:.0f}/100")
    _log(f"   Line cov:   {line_cov:.1f}%")
    _log(f"   Branch cov: {branch_cov:.1f}%")
    _log(f"   Mutation:   {mutation_score:.0f}%")

    # Flywheel
    source_code = state.get("source_code", "")
    if line_cov >= 25 and mutation_score >= 20:
        save_to_flywheel(target_file, source_code, accumulated_code, line_cov, mutation_score)

    summary = (
        f"✅ {n_tests} tests | Line: {line_cov:.1f}% | "
        f"Branch: {branch_cov:.1f}% | Mutation: {mutation_score:.0f}% | "
        f"Composite: {composite['composite']:.0f}/100"
    )
    _log(f"🏁 {summary}")

    return {
        "final_coverage": line_cov,
        "final_branch_coverage": branch_cov,
        "mutation_score": mutation_score,
        "composite_score": composite,
        "test_results_summary": summary,
    }


# ---------------------------------------------------------------------------
# Edge: route after human review
# ---------------------------------------------------------------------------

def route_after_review(state: TestGenState) -> Literal["finalize", "generate_all_tests"]:
    """Route based on human decision."""
    decision = state.get("human_decision", "approve")
    if decision == "reject":
        # User wants to retry — go back to generation
        return "generate_all_tests"
    # approve, edit, skip → finalize
    return "finalize"


# ---------------------------------------------------------------------------
# Build the LangGraph
# ---------------------------------------------------------------------------

def build_agent_graph() -> StateGraph:
    """Build and return the TestMate LangGraph agent with HITL."""

    graph = StateGraph(TestGenState)

    # Add nodes
    graph.add_node("init", node_init)
    graph.add_node("generate_all_tests", node_generate_all_tests)
    graph.add_node("human_review", node_human_review)
    graph.add_node("finalize", node_finalize)

    # Add edges
    graph.add_edge(START, "init")
    graph.add_edge("init", "generate_all_tests")
    graph.add_edge("generate_all_tests", "human_review")

    # Conditional edge after human review
    graph.add_conditional_edges(
        "human_review",
        route_after_review,
        {
            "finalize": "finalize",
            "generate_all_tests": "generate_all_tests",
        },
    )
    graph.add_edge("finalize", END)

    return graph


def create_agent():
    """Create a compiled LangGraph agent with memory checkpointing."""
    graph = build_agent_graph()
    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)


# ---------------------------------------------------------------------------
# Convenience: run agent (CLI or programmatic use)
# ---------------------------------------------------------------------------

def run_agent(
    target_file: str,
    import_path: str = "",
    auto_approve: bool = True,
    deep_scan: bool = False,
    max_retries: int = 3,
    thread_id: str = "default",
) -> dict:
    """Run the TestMate agent.

    Args:
        target_file: Path to the Python file to test.
        import_path: Package import path (e.g. 'requests.auth').
        auto_approve: If True, skip human review.
        deep_scan: Enable deep scan mode.
        max_retries: Max retries per target.
        thread_id: Unique thread ID for checkpointing.

    Returns:
        Final state dict with results.
    """
    agent = create_agent()

    initial_state = {
        "target_file": target_file,
        "import_path": import_path,
        "source_code": "",
        "deep_scan": deep_scan,
        "max_retries": max_retries,
        "auto_approve": auto_approve,
        "targets": [],
        "module_name": "",
        "actual_import": "",
        "import_statement": "",
        "test_file": "",
        "ctx": {},
        "framework_header": "",
        "repo_ctx": {},
        "accumulated_code": "",
        "accumulated_tests": 0,
        "covered_summary": [],
        "failed_targets": [],
        "current_target_idx": 0,
        "final_coverage": 0.0,
        "final_branch_coverage": 0.0,
        "mutation_score": 0.0,
        "composite_score": {},
        "test_results_summary": "",
        "human_decision": "",
        "human_edited_code": "",
        "review_requested": False,
        "log_messages": [],
    }

    config = {"configurable": {"thread_id": thread_id}}

    # Run the graph
    final_state = None
    for event in agent.stream(initial_state, config=config, stream_mode="updates"):
        for node_name, updates in event.items():
            if isinstance(updates, dict):
                final_state = updates

    return final_state or {}


# ---------------------------------------------------------------------------
# Resume agent after human review
# ---------------------------------------------------------------------------

def resume_agent(
    thread_id: str,
    decision: str,
    edited_code: str = "",
    agent=None,
) -> dict:
    """Resume the agent after human review.

    Args:
        thread_id: Same thread_id used in run_agent.
        decision: 'approve', 'edit', 'reject', or 'skip'.
        edited_code: If decision=='edit', the modified test code.
        agent: The compiled agent (must be the same instance).

    Returns:
        Final state dict with results.
    """
    config = {"configurable": {"thread_id": thread_id}}

    resume_value = {
        "decision": decision,
        "edited_code": edited_code,
    }

    final_state = None
    for event in agent.stream(
        Command(resume=resume_value),
        config=config,
        stream_mode="updates",
    ):
        for node_name, updates in event.items():
            if isinstance(updates, dict):
                final_state = updates

    return final_state or {}


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="TestMate LangGraph Agent")
    parser.add_argument("--target", required=True, help="Python file to test")
    parser.add_argument("--import-as", default="", help="Package import path")
    parser.add_argument("--auto-approve", action="store_true",
                        help="Skip human review (auto-approve all)")
    parser.add_argument("--deep-scan", action="store_true")
    parser.add_argument("--max-retries", type=int, default=3)
    args = parser.parse_args()

    print("=" * 60)
    print("  🤖 TestMate LangGraph Agent (with HITL)")
    print("=" * 60)

    result = run_agent(
        target_file=args.target,
        import_path=args.import_as,
        auto_approve=args.auto_approve,
        deep_scan=args.deep_scan,
        max_retries=args.max_retries,
    )

    print(f"\n📊 Result: {result.get('test_results_summary', 'No results')}")
