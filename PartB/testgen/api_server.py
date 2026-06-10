"""
TestMate — FastAPI Backend
===========================
Serves API endpoints for the React desktop GUI.
Wraps core logic from main.py and docker_runner.py.

Usage:
  python api_server.py              # starts at http://localhost:8000
  python api_server.py --port 9000  # custom port
"""

import os
import sys
import json
import re
import ast
import time
import queue
import threading
import multiprocessing
import subprocess
import argparse
import asyncio
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.responses import StreamingResponse

# ── Paths ──
TESTGEN_DIR = Path(__file__).parent
REPOS_DIR = TESTGEN_DIR / "repos"
RESULTS_DIR = TESTGEN_DIR / "results"
GEN_TESTS_DIR = TESTGEN_DIR / "generated_tests"

app = FastAPI(title="TestMate", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static assets (styles.css, app.js) and the dashboard HTML
_STATIC_DIR = TESTGEN_DIR / "static"
_TEMPLATES_DIR = TESTGEN_DIR / "templates"
if _STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

# ── Global State ──
log_queue = multiprocessing.Queue()
review_queue = multiprocessing.Queue()  # parent→child: HITL decisions
plan_queue = multiprocessing.Queue()    # parent→child: plan review decisions
current_run = {"status": "idle", "results": None}

# HITL tracking (parent-side only)
review_pending = False
_review_queue = review_queue  # child-side alias (overwritten in subprocess)
_plan_queue = plan_queue      # child-side alias (overwritten in subprocess)

# Singleton model — NOT used when subprocess isolation is active
_model_holder = {"model": None, "tokenizer": None, "loading": False}
_model_lock = threading.Lock()

# Track the child process for run_evaluation
_run_process = None

# ============================================================
# FILE DISCOVERY
# ============================================================

SKIP_FILES = {
    "__init__.py", "setup.py", "setup.cfg", "conftest.py",
    "manage.py", "wsgi.py", "asgi.py", "__main__.py",
    "__version__.py", "_version.py", "version.py",
}
SKIP_DIRS = {
    "test", "tests", "testing", "__pycache__", ".git", ".github",
    "docs", "doc", "examples", "example", "scripts", "benchmarks",
    "migrations", "node_modules", ".tox", ".eggs", "build", "dist",
    ".mypy_cache", ".pytest_cache", "venv", "env", ".venv",
}
MAX_FILE_LINES = 400
MIN_FILE_LINES = 10


def auto_discover_files(repo_dir: str, max_files: int = 15) -> list[dict]:
    discovered = []
    package_name = None
    package_root = repo_dir  # directory containing the package

    # Look for top-level package (dir with __init__.py)
    for item in sorted(os.listdir(repo_dir)):
        pkg_dir = os.path.join(repo_dir, item)
        if (os.path.isdir(pkg_dir) and
            os.path.exists(os.path.join(pkg_dir, "__init__.py")) and
            not item.startswith(".") and item not in SKIP_DIRS):
            package_name = item
            break

    # src-layout: check src/<package>/ if no top-level package found
    if not package_name:
        src_dir = os.path.join(repo_dir, "src")
        if os.path.isdir(src_dir):
            for item in sorted(os.listdir(src_dir)):
                pkg_dir = os.path.join(src_dir, item)
                if (os.path.isdir(pkg_dir) and
                    os.path.exists(os.path.join(pkg_dir, "__init__.py")) and
                    not item.startswith(".") and item not in SKIP_DIRS):
                    package_name = item
                    package_root = src_dir  # import paths are relative to src/
                    break

    for root, dirs, files in os.walk(repo_dir):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith(".")]
        for fname in sorted(files):
            if not fname.endswith(".py") or fname in SKIP_FILES:
                continue
            if fname.startswith("test_") or fname.endswith("_test.py"):
                continue
            fpath = os.path.join(root, fname)
            rel_path = os.path.relpath(fpath, repo_dir).replace("\\", "/")
            try:
                with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                    lines = f.readlines()
            except Exception:
                continue
            num_lines = len(lines)
            if num_lines < MIN_FILE_LINES or num_lines > MAX_FILE_LINES:
                continue
            source = "".join(lines)
            try:
                tree = ast.parse(source)
            except SyntaxError:
                continue
            functions = [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
            classes = [n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
            if len(functions) + len(classes) < 1:
                continue

            # Compute import_path relative to the package root
            import_path = None
            if package_name:
                rel_to_pkg_root = os.path.relpath(fpath, package_root).replace("\\", "/")
                if rel_to_pkg_root.startswith(package_name + "/"):
                    import_path = rel_to_pkg_root.replace("/", ".").replace(".py", "")

            discovered.append({
                "path": rel_path, "abs_path": fpath, "import_path": import_path,
                "lines": num_lines, "functions": len(functions), "classes": len(classes),
                "func_names": functions[:8], "class_names": classes[:5],
            })
    discovered.sort(key=lambda x: x["functions"] + x["classes"], reverse=True)
    return discovered[:max_files]


def clone_repo(url: str, branch: str = None) -> tuple[str, str]:
    name = url.rstrip("/").split("/")[-1].replace(".git", "")
    repo_dir = str(REPOS_DIR / name)
    if os.path.exists(repo_dir):
        return repo_dir, name
    REPOS_DIR.mkdir(exist_ok=True)
    cmd = ["git", "clone", "--depth=1"]
    if branch:
        cmd += ["--branch", branch]
    cmd += [url, repo_dir]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(f"Clone failed: {result.stderr[:300]}")
    return repo_dir, name


# ============================================================
# SSE HELPERS
# ============================================================

def emit(data: dict):
    log_queue.put(json.dumps(data))

def emit_log(msg: str, level: str = "info"):
    emit({"type": "log", "level": level, "message": msg})

def emit_progress(current: int, total: int, file_name: str):
    emit({"type": "progress", "current": current, "total": total, "file": file_name})

def emit_pipeline_stage(stage: str):
    emit({"type": "pipeline_stage", "stage": stage})

def emit_ai_status(status: str, detail: str = "", target: str = ""):
    emit({"type": "ai_status", "status": status, "detail": detail, "target": target})

def emit_code_stream(code: str, filename: str = "", target: str = "", is_retry: bool = False):
    emit({"type": "code_stream", "code": code, "filename": filename, "target": target, "is_retry": is_retry})

def emit_code_clear():
    emit({"type": "code_clear"})

def emit_result(results: dict):
    emit({"type": "result", "data": results})

def emit_complete():
    emit({"type": "complete"})


# ============================================================
# MODEL MANAGEMENT
# ============================================================

def get_model():
    """Lazy-load the model (singleton).
    
    Initializes CUDA context in the calling thread/process before loading.
    """
    with _model_lock:
        if _model_holder["model"] is not None:
            return _model_holder["model"], _model_holder["tokenizer"]
        if _model_holder["loading"]:
            return None, None
        _model_holder["loading"] = True

    try:
        sys.path.insert(0, str(TESTGEN_DIR))
        from main import load_model
        model, tokenizer = load_model()
        with _model_lock:
            _model_holder["model"] = model
            _model_holder["tokenizer"] = tokenizer
            _model_holder["loading"] = False
        return model, tokenizer
    except Exception as e:
        with _model_lock:
            _model_holder["loading"] = False
        raise


# ============================================================
# CFG EXTRACTION
# ============================================================

def extract_cfg_paths(source_path: str) -> list[dict]:
    try:
        with open(source_path, encoding="utf-8", errors="replace") as f:
            source = f.read()
        tree = ast.parse(source)
    except (SyntaxError, IOError):
        return []
    functions = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        paths = []
        _trace_cfg(node, [], paths, 0)
        if paths:
            functions.append({"name": node.name, "line": node.lineno, "paths": paths[:20]})
    return functions


def _trace_cfg(node, current: list, all_paths: list, depth: int):
    if depth > 15 or len(all_paths) >= 20:
        return
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        current = [f"def {node.name}()"]
        for child in ast.iter_child_nodes(node):
            _trace_cfg(child, current.copy(), all_paths, depth + 1)
        if current and not all_paths:
            all_paths.append(current + ["return"])
    elif isinstance(node, ast.If):
        true_path = current + [f"if (L{node.lineno})"]
        for child in node.body:
            _trace_cfg(child, true_path, all_paths, depth + 1)
        if not any(isinstance(c, (ast.If, ast.For, ast.While, ast.Return)) for c in node.body):
            all_paths.append(true_path + ["continue"])
        if node.orelse:
            if isinstance(node.orelse[0], ast.If):
                _trace_cfg(node.orelse[0], current + [f"elif (L{node.orelse[0].lineno})"], all_paths, depth + 1)
            else:
                all_paths.append(current + [f"else (L{node.lineno})", "continue"])
        else:
            all_paths.append(current + [f"skip if (L{node.lineno})"])
    elif isinstance(node, ast.For):
        for child in node.body:
            _trace_cfg(child, current + [f"for (L{node.lineno})"], all_paths, depth + 1)
    elif isinstance(node, ast.While):
        for child in node.body:
            _trace_cfg(child, current + [f"while (L{node.lineno})"], all_paths, depth + 1)
    elif isinstance(node, ast.Try):
        for child in node.body:
            _trace_cfg(child, current + [f"try (L{node.lineno})"], all_paths, depth + 1)
        for handler in node.handlers:
            exc = handler.type.id if handler.type and hasattr(handler.type, 'id') else "Exception"
            all_paths.append(current + [f"except {exc} (L{handler.lineno})", "handle"])
    elif isinstance(node, ast.Return):
        all_paths.append(current + ["return"])
    elif isinstance(node, ast.Raise):
        all_paths.append(current + ["raise"])


# ============================================================
# EVALUATION RUNNER
# ============================================================

def run_evaluation(repo_url: str, branch: str, selected_files: list[dict],
                   use_docker: bool = False, deep_scan: bool = False,
                   max_retries: int = 3, hitl: bool = False, intense: bool = False,
                   plan_mode: bool = False):
    global current_run
    current_run["status"] = "running"
    current_run["results"] = None

    try:
        # Step 1: Clone
        emit_pipeline_stage("scan")
        is_local = (os.path.isdir(repo_url) or
                    (len(repo_url) >= 2 and repo_url[1] == ":") or
                    repo_url.startswith("/") or repo_url.startswith("\\"))

        if is_local:
            repo_dir = os.path.abspath(repo_url)
            repo_name = os.path.basename(repo_dir)
            emit_log(f"📂 Using local path: {repo_dir}")
        else:
            if not re.match(r"https?://", repo_url):
                repo_url = "https://github.com/" + repo_url
            emit_log(f"📥 Cloning {repo_url}...")
            repo_dir, repo_name = clone_repo(repo_url, branch or None)
            emit_log(f"✅ Cloned to repos/{repo_name}/", "success")

        # Step 2: Load model
        emit_pipeline_stage("model")
        emit_ai_status("loading", "Loading Qwen2.5-Coder-7B + LoRA into GPU...")
        emit_log("🧠 Loading model... (~30s on RTX 4060)")

        sys.path.insert(0, str(TESTGEN_DIR))
        from main import autonomous_loop

        if intense:
            from intense_mode import intense_mode as run_intense

        model, tokenizer = get_model()
        emit_ai_status("loading", "Model loaded — ready to generate")
        emit_log("✅ Model loaded on GPU", "success")

        # Step 3: Generate tests
        emit_pipeline_stage("gen")
        GEN_TESTS_DIR.mkdir(exist_ok=True)
        bug_reports_path = GEN_TESTS_DIR / "bug_reports.jsonl"
        if bug_reports_path.exists():
            bug_reports_path.unlink()
        discarded_path = GEN_TESTS_DIR / "discarded_reports.jsonl"
        if discarded_path.exists():
            discarded_path.unlink()

        test_files = []
        total = len(selected_files)

        for i, file_info in enumerate(selected_files, 1):
            src_path = file_info["abs_path"]
            import_path = file_info.get("import_path")
            basename = os.path.basename(src_path)

            emit_progress(i, total, basename)
            emit_ai_status("analyzing", f"Preparing context for {basename}", basename)
            emit_log(f"\n{'─'*50}")
            emit_log(f"🤖 [{i}/{total}] Testing: {file_info['path']}")

            t0 = time.time()
            try:
                if intense:
                    result = run_intense(
                        target_file=src_path, import_path=import_path,
                        max_iterations=5, max_retries=max_retries, log_callback=emit_log,
                    )
                    success = result.get("best_score", 0) > 0
                else:
                    def _gui_callback(event_type, *args):
                        if event_type == "ai_status":
                            emit_ai_status(*args)
                        elif event_type == "code_stream":
                            # Stream code line-by-line for typewriter effect
                            code = args[0] if args else ""
                            rest_args = args[1:] if len(args) > 1 else ()
                            lines = code.split('\n')
                            for line in lines:
                                emit_code_stream(line, *rest_args)
                                time.sleep(0.05)  # 50ms per line for visible streaming
                        elif event_type == "code_clear":
                            emit_code_clear()
                        elif event_type == "plan_request":
                            # Plan mode: send plan to frontend for review/edit
                            plan_text = args[0] if args else ""
                            target_label = args[1] if len(args) > 1 else ""
                            test_file = args[2] if len(args) > 2 else ""
                            method_name = args[3] if len(args) > 3 else ""
                            class_name = args[4] if len(args) > 4 else ""
                            emit({"type": "plan_request",
                                  "plan": plan_text,
                                  "target": target_label,
                                  "test_file": test_file,
                                  "method_name": method_name,
                                  "class_name": class_name})
                            emit_log(f"   📋 Plan ready — waiting for review...")
                            # Wait for user to approve/edit the plan
                            try:
                                plan_response = _plan_queue.get(timeout=600)
                            except Exception:
                                plan_response = {"plan": plan_text}  # timeout → use original
                            # Update the plan in autonomous_loop via return
                            # We use a mutable container to pass the edited plan back
                            _gui_callback._edited_plan = plan_response.get("plan", plan_text)
                            emit_log(f"   📋 Plan approved — generating tests...", "success")

                    _gui_callback._edited_plan = None  # init mutable attr

                    success = autonomous_loop(
                        model, tokenizer, src_path,
                        import_path=import_path, deep_scan=deep_scan,
                        max_retries=max_retries, log_callback=_gui_callback,
                        plan_mode=plan_mode,
                    )

                elapsed = time.time() - t0
                icon = "✅" if success else "⚠️"
                emit_log(f"   {icon} Completed in {elapsed:.0f}s", "success" if success else "warning")
            except Exception as e:
                emit_log(f"   ❌ Failed: {str(e)[:200]}", "error")
                continue

            test_name = f"test_{basename}"
            test_path = str(GEN_TESTS_DIR / test_name)
            if not os.path.exists(test_path):
                test_path = os.path.join(os.path.dirname(src_path), test_name)
            if os.path.exists(test_path):
                test_files.append(test_path)

            # ── Per-file HITL review (loops so 'revise' can re-present) ──
            while hitl and os.path.exists(test_path):
                try:
                    with open(test_path, "r", encoding="utf-8") as f:
                        file_test_code = f.read()
                except Exception:
                    file_test_code = ""

                n_tests = file_test_code.count("def test_")
                emit_log(f"⏸️  Pausing for human review of {basename}...")
                emit({"type": "review_request",
                      "test_code": file_test_code,
                      "test_file": test_name,
                      "num_tests": n_tests,
                      "target_file": basename,
                      "file_index": i,
                      "file_total": total})

                # Wait for decision from parent via review_queue
                # (parent's /api/review puts the decision here)
                try:
                    decision_data = _review_queue.get(timeout=600)
                except Exception:
                    decision_data = {"decision": "approve"}

                decision = decision_data.get("decision", "approve")
                emit_log(f"👤 Decision for {basename}: {decision}", "success")

                if decision == "revise":
                    # Natural-language change request → regenerate, then re-review.
                    instruction = decision_data.get("edited_code", "").strip()
                    if not instruction:
                        continue
                    emit_log(f"   💬 Revising {basename}: {instruction[:80]}")
                    try:
                        from testgen_api import refine_test_file
                        refine_test_file(
                            target_file=src_path, test_path=test_path,
                            test_code=file_test_code, instruction=instruction,
                            model=model, tokenizer=tokenizer,
                            max_retries=max_retries, log_callback=_gui_callback,
                            use_docker=use_docker,
                        )
                    except Exception as _re:
                        emit_log(f"   ⚠️ Revision failed: {str(_re)[:160]}", "warning")
                    continue  # re-present the (possibly updated) test for review

                if decision == "reject":
                    # Remove this test file and skip it
                    if test_path in test_files:
                        test_files.remove(test_path)
                    if os.path.exists(test_path):
                        os.remove(test_path)
                    emit_log(f"   🗑️ Test for {basename} rejected and removed")
                elif decision == "edit":
                    edited = decision_data.get("edited_code", "").strip()
                    if edited:
                        with open(test_path, "w", encoding="utf-8") as f:
                            f.write(edited)
                        emit_log(f"   ✏️ Test for {basename} updated with edits")
                # "approve"/"edit"/"reject" → done reviewing this file
                break

        # Step 5: Evaluate
        emit_pipeline_stage("eval")
        if test_files:
            from docker_runner import run_tests_locally, run_tests_in_docker
            full_source_paths = [f["abs_path"] for f in selected_files]
            cov_package = None
            for f in selected_files:
                if f.get("import_path"):
                    cov_package = f["import_path"].split(".")[0]
                    break

            if use_docker:
                emit_log(f"🐳 Running tests in Docker...")
                results = run_tests_in_docker(repo_dir, test_files, full_source_paths,
                                              repo_name=cov_package or repo_name)
            else:
                emit_log(f"📊 Running evaluation locally...")
                results = run_tests_locally(repo_dir, test_files, full_source_paths,
                                            cov_package=cov_package)

            results["repo_name"] = repo_name
            results["repo_url"] = repo_url
            results["num_source_files"] = len(selected_files)
            results["num_test_files"] = len(test_files)

            RESULTS_DIR.mkdir(exist_ok=True)
            with open(RESULTS_DIR / f"{repo_name}.json", "w") as f:
                json.dump(results, f, indent=2)

            # Fetch per-line coverage data computed by docker_runner
            try:
                from docker_runner import get_coverage_data
                cov_data = get_coverage_data()
                if cov_data:
                    # Attach per-line coverage maps directly to results
                    # so the frontend receives them through SSE
                    coverage_map = {}
                    for fname, fdata in cov_data.items():
                        coverage_map[fname] = {
                            "covered_lines": fdata.get("covered_lines", []),
                            "uncovered_lines": fdata.get("uncovered_lines", []),
                            "excluded_lines": fdata.get("excluded_lines", []),
                            "line_coverage_pct": fdata.get("line_coverage_pct", 0),
                            "num_statements": fdata.get("num_statements", 0),
                            "source": fdata.get("source", ""),
                        }
                    results["coverage_map"] = coverage_map
            except Exception:
                pass

            # Also read generated test code for display
            test_code_map = {}
            for tf in test_files:
                try:
                    with open(tf, "r", encoding="utf-8", errors="replace") as f:
                        test_code_map[os.path.basename(tf)] = f.read()
                except Exception:
                    pass
            if test_code_map:
                results["test_code_map"] = test_code_map

            emit_log(f"🏁 EVALUATION COMPLETE", "success")
            emit_log(f"   Tests: {results.get('passed_tests', 0)}/{results.get('total_tests', 0)} passed")
            emit_log(f"   Coverage: {results.get('line_coverage', 0):.1f}%")
            emit_result(results)
        else:
            emit_log("⚠️ No test files were generated", "warning")
            emit_result({"error": "No tests generated"})

        current_run["status"] = "done"
    except Exception as e:
        emit_log(f"❌ Fatal error: {str(e)}", "error")
        import traceback
        emit_log(traceback.format_exc(), "error")
        current_run["status"] = "error"

    emit_complete()


def _run_evaluation_subprocess(shared_queue, shared_review_queue, shared_plan_queue,
                                repo_url, branch, selected_files,
                                use_docker=False, deep_scan=False,
                                max_retries=3, hitl=False, intense=False,
                                plan_mode=False):
    """Subprocess entry point for run_evaluation.
    
    Runs in a separate process to isolate CUDA model inference.
    If the model segfaults, only this process dies — the API server survives.
    """
    global log_queue, _review_queue, _plan_queue
    # Replace module-level queues with the shared multiprocessing queues
    log_queue = shared_queue
    _review_queue = shared_review_queue
    _plan_queue = shared_plan_queue

    # Set CUDA allocator config BEFORE any PyTorch/CUDA import side-effects
    import os
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
        "expandable_segments:True,garbage_collection_threshold:0.6"
    )

    try:
        run_evaluation(repo_url, branch, selected_files,
                       use_docker, deep_scan, max_retries, hitl, intense,
                       plan_mode)
    except Exception as e:
        import traceback
        emit_log(f"❌ Fatal subprocess error: {e}", "error")
        emit_log(traceback.format_exc(), "error")
        emit_complete()


# ============================================================
# ROUTES
# ============================================================

@app.get("/")
async def index():
    """Serve the dashboard HTML for the Electron desktop app."""
    html = _TEMPLATES_DIR / "index_v2.html"
    if not html.exists():
        html = _TEMPLATES_DIR / "index.html"
    return FileResponse(str(html))


@app.get("/api/status")
async def api_status():
    return current_run


@app.post("/api/discover")
async def api_discover(request: Request):
    data = await request.json()
    url = data.get("url", "").strip()
    branch = data.get("branch", "").strip()
    if not url:
        return JSONResponse({"error": "No URL provided"}, status_code=400)
    try:
        is_local = (os.path.isdir(url) or
                    (len(url) >= 2 and url[1] == ":") or
                    url.startswith("/") or url.startswith("\\"))
        if is_local:
            repo_dir = os.path.abspath(url)
            if not os.path.isdir(repo_dir):
                return JSONResponse({"error": f"Folder not found: {repo_dir}"}, status_code=400)
            repo_name = os.path.basename(repo_dir)
            source_type = "local"
        else:
            if not re.match(r"https?://", url):
                url = "https://github.com/" + url
            repo_dir, repo_name = clone_repo(url, branch or None)
            source_type = "github"

        files = auto_discover_files(repo_dir, max_files=15)
        return {"repo_name": repo_name, "repo_dir": repo_dir,
                "source_type": source_type, "files": files, "total_discovered": len(files)}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/run")
async def api_run(request: Request):
    global current_run, _run_process
    if current_run["status"] == "running":
        return JSONResponse({"error": "Already running"}, status_code=409)
    data = await request.json()
    files = data.get("files", [])
    if not files:
        return JSONResponse({"error": "No files selected"}, status_code=400)

    # Drain any leftover messages from previous runs
    while True:
        try: log_queue.get_nowait()
        except: break

    current_run["status"] = "running"
    current_run["results"] = None

    # Use multiprocessing.Process to isolate CUDA model inference.
    # If the model segfaults (0xC0000005), only the child process dies —
    # the API server stays alive and can report the error.
    # Drain any leftover review/plan decisions
    while True:
        try: review_queue.get_nowait()
        except: break
    while True:
        try: plan_queue.get_nowait()
        except: break

    _run_process = multiprocessing.Process(
        target=_run_evaluation_subprocess,
        args=(log_queue, review_queue, plan_queue,
              data.get("url", ""), data.get("branch", ""), files,
              data.get("docker", False), data.get("deep_scan", False),
              max(1, min(int(data.get("max_retries", 3)), 5)),
              data.get("hitl", False), data.get("intense", False),
              data.get("plan_mode", False)),
        daemon=True,
    )
    _run_process.start()

    # Monitor the process in a background thread
    def _monitor():
        global current_run
        _run_process.join()
        exit_code = _run_process.exitcode
        if exit_code == 0:
            # Normal completion — status already set via SSE events
            if current_run["status"] == "running":
                current_run["status"] = "done"
        elif exit_code is not None:
            # Process crashed (segfault, OOM, etc.)
            error_msg = f"Worker process crashed with exit code {exit_code}"
            if exit_code == -11 or exit_code == 3221225477:  # SIGSEGV / Windows AV
                error_msg += " (GPU memory access violation — try restarting)"
            elif exit_code == -9:  # SIGKILL
                error_msg += " (killed by OS — likely out of memory)"
            log_queue.put(json.dumps({"type": "log", "level": "error",
                                       "message": f"❌ {error_msg}"}))
            log_queue.put(json.dumps({"type": "complete"}))
            current_run["status"] = "error"

    monitor_thread = threading.Thread(target=_monitor, daemon=True)
    monitor_thread.start()

    return {"status": "started"}


@app.get("/api/stream")
async def api_stream():
    async def generate():
        while True:
            try:
                msg = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: log_queue.get(timeout=30))
                yield f"data: {msg}\n\n"
                if '"type": "complete"' in msg or '"type":"complete"' in msg:
                    break
            except (queue.Empty, Exception):
                yield f"data: {json.dumps({'type': 'keepalive'})}\n\n"
    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/api/review")
async def api_review(request: Request):
    data = await request.json()
    decision = data.get("decision", "approve")
    edited_code = data.get("edited_code", "")
    # Put the decision on the review_queue so the child process picks it up
    review_queue.put({"decision": decision, "edited_code": edited_code})
    return {"status": "ok", "decision": decision}


@app.post("/api/plan_review")
async def api_plan_review(request: Request):
    data = await request.json()
    plan_text = data.get("plan", "")
    # Put the (possibly edited) plan on the plan_queue so the child process picks it up
    plan_queue.put({"plan": plan_text})
    return {"status": "ok"}

@app.get("/api/coverage/{filename}")
async def api_coverage(filename: str):
    # Try docker_runner coverage data
    try:
        from docker_runner import get_coverage_data
        cov_data = get_coverage_data()
        for key, val in cov_data.items():
            if filename == os.path.basename(key) or filename == key:
                return {
                    "filename": filename,
                    "covered_lines": val.get("covered_lines", []),
                    "uncovered_lines": val.get("uncovered_lines", []),
                    "excluded_lines": val.get("excluded_lines", []),
                    "line_coverage_pct": val.get("line_coverage_pct", 0),
                    "num_statements": val.get("num_statements", 0),
                    "source": val.get("source", ""),
                }
    except Exception:
        pass

    # Fallback: read source directly
    try:
        for root, dirs, files_list in os.walk(str(REPOS_DIR)):
            if filename in files_list:
                source_path = os.path.join(root, filename)
                with open(source_path, encoding="utf-8", errors="replace") as f:
                    source_text = f.read()
                lines = source_text.strip().split('\n')
                return {
                    "filename": filename,
                    "covered_lines": list(range(1, len(lines) + 1)),
                    "uncovered_lines": [],
                    "excluded_lines": [],
                    "line_coverage_pct": 100.0,
                    "num_statements": len(lines),
                    "source": source_text,
                }
    except Exception:
        pass
    return {"error": f"Source file '{filename}' not found"}


@app.get("/api/testcode/{filename}")
async def api_testcode(filename: str):
    basename = os.path.basename(filename)
    test_name = f"test_{basename}" if not basename.startswith("test_") else basename
    test_path = GEN_TESTS_DIR / test_name
    if not test_path.exists():
        test_path_r = RESULTS_DIR / test_name
        if test_path_r.exists():
            test_path = test_path_r
        else:
            return {"error": f"Test file not found: {test_name}"}
    with open(test_path, "r", encoding="utf-8", errors="replace") as f:
        source = f.read()
    return {"filename": test_name, "source": source}


@app.get("/api/cfg/{filename}")
async def api_cfg(filename: str):
    source_path = None
    try:
        for root, dirs, files in os.walk(str(REPOS_DIR)):
            if filename in files:
                source_path = os.path.join(root, filename)
                break
    except Exception:
        pass
    if source_path and os.path.exists(source_path):
        try:
            functions = extract_cfg_paths(source_path)
            return {"filename": filename, "functions": functions}
        except Exception:
            pass
    return {"filename": filename, "functions": []}


@app.get("/api/discarded/{filename}")
async def api_discarded(filename: str):
    discarded_path = GEN_TESTS_DIR / "discarded_reports.jsonl"
    if not discarded_path.exists():
        return {"reports": [], "count": 0}
    reports = []
    stem = filename.replace('.py', '') if filename.endswith('.py') else filename
    try:
        with open(discarded_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    if entry.get("source_file", "").replace('.py', '') == stem:
                        reports.append(entry)
                except json.JSONDecodeError:
                    continue
    except Exception:
        pass
    return {"reports": reports, "count": len(reports)}


@app.post("/api/chat")
async def api_chat(request: Request):
    data = await request.json()
    message = data.get("message", "").strip()
    if not message:
        return JSONResponse({"error": "No message"}, status_code=400)

    async def generate():
        try:
            model, tokenizer = get_model()
            if model is None:
                yield f"data: {json.dumps({'type': 'chat_token', 'token': 'Model is loading, please wait...'})}\n\n"
                yield f"data: {json.dumps({'type': 'chat_done'})}\n\n"
                return

            sys.path.insert(0, str(TESTGEN_DIR))
            from main import generate_test

            messages = [
                {"role": "system", "content": "You are TestMate AI, an expert Python testing assistant. Help users with test strategies, code analysis, and debugging. Be concise and practical."},
                {"role": "user", "content": message},
            ]
            response = await asyncio.get_event_loop().run_in_executor(
                None, lambda: generate_test(model, tokenizer, messages, max_tokens=512)
            )
            yield f"data: {json.dumps({'type': 'chat_token', 'token': response})}\n\n"
            yield f"data: {json.dumps({'type': 'chat_done'})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'chat_token', 'token': f'Error: {str(e)}'})}\n\n"
            yield f"data: {json.dumps({'type': 'chat_done'})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    # CRITICAL for Windows: must use 'spawn' start method and freeze_support
    # to prevent recursive process spawning and enable proper CUDA isolation
    multiprocessing.freeze_support()
    multiprocessing.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser(description="TestMate FastAPI Server")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    REPOS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    GEN_TESTS_DIR.mkdir(exist_ok=True)

    print("=" * 60)
    print("  TestMate — FastAPI Server")
    print(f"  http://localhost:{args.port}")
    print("=" * 60)

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
