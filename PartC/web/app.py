"""
app.py — TestMate APR Flask Web UI
SSE-streaming pipeline with live logs, SBFL viewer, code diff, coverage, and mutation panels.
"""
import sys
import os
import json
import queue
import threading
import subprocess
import ast
import re
import copy
import tempfile
from pathlib import Path
from flask import Flask, render_template, jsonify, request, Response, stream_with_context

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

BUGGY_FILE = PROJECT_ROOT / "buggy_code.py"
TEST_FILE  = PROJECT_ROOT / "test_logic.py"
ARTIFACTS  = PROJECT_ROOT / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)

app = Flask(__name__, template_folder="templates")

# ── Shared pipeline state ─────────────────────────────────────────────
state = {
    "running":       False,
    "step":          0,
    "done":          False,
    "success":       False,
    "attempt":       0,
    "logs":          [],
    "suspicious":    [],
    "attempts":      [],
    "original_code": "",
    "fixed_code":    "",
}
log_q = queue.Queue()


def emit(typ: str, msg: str):
    entry = {"type": typ, "msg": msg}
    state["logs"].append(entry)
    log_q.put(entry)
    print(f"[{typ.upper()}] {msg}")


# ── API: current code snapshot ────────────────────────────────────────
@app.route("/api/code")
def api_code():
    current = BUGGY_FILE.read_text(encoding="utf-8") if BUGGY_FILE.exists() else "# buggy_code.py not found"
    fixed = ""
    if state["done"] and state["success"]:
        fixed = state["fixed_code"]
    elif state["done"] and not state["success"]:
        patches = sorted(ARTIFACTS.glob("model_output_attempt_*.py"))
        fixed = patches[-1].read_text(encoding="utf-8") if patches else ""

    return jsonify({
        "buggy":   state["original_code"] or current,
        "current": current,
        "fixed":   fixed,
    })


# ── API: run pytest and return results ────────────────────────────────
@app.route("/api/tests")
def api_tests():
    r = subprocess.run(
         [sys.executable, "-m", "pytest", str(TEST_FILE), "-v", "--tb=short", "--no-header"],
        capture_output=True, text=True, cwd=str(PROJECT_ROOT)
    )
    out = r.stdout + r.stderr
    tests = []
    for line in out.splitlines():
        if "::" in line and (" PASSED" in line or " FAILED" in line):
            name   = line.split("::")[-1].split(" ")[0]
            status = "pass" if " PASSED" in line else "fail"
            tests.append({"name": name, "status": status})
    trace = "\n".join(
        l for l in out.splitlines()
        if l.startswith("E ") or "AssertionError" in l or ("Error" in l and "error" in l.lower())
    )
    return jsonify({
        "total":  len(tests),
        "passed": sum(1 for t in tests if t["status"] == "pass"),
        "failed": sum(1 for t in tests if t["status"] == "fail"),
        "tests":  tests,
        "trace":  trace[:600],
        "raw":    out,
    })


# ── API: SBFL suspicious lines ────────────────────────────────────────
@app.route("/api/sbfl")
def api_sbfl():
    if state["suspicious"]:
        return jsonify(state["suspicious"])
    try:
        from core.run_and_collect import collect_spectrum
        from core.sbfl_localiser  import rank_suspicious_lines
        spectrum, _ = collect_spectrum()
        ranked  = rank_suspicious_lines(spectrum)
        lines   = BUGGY_FILE.read_text(encoding="utf-8").splitlines()
        result  = [
            {"line": ln, "score": round(sc, 3),
             "code": lines[ln - 1] if 0 < ln <= len(lines) else ""}
            for ln, sc in ranked[:5]
        ]
        return jsonify(result)
    except Exception as e:
        return jsonify([])


# ── API: all logs (SSE fallback) ──────────────────────────────────────
@app.route("/api/logs")
def api_logs():
    return jsonify({"logs": state["logs"]})


# ── API: pipeline status polling ──────────────────────────────────────
@app.route("/api/status")
def api_status():
    """Return current pipeline state — used by the frontend to poll progress."""
    return jsonify({k: state[k] for k in
        ["running", "step", "done", "success", "attempt", "suspicious", "attempts"]})


# ── SSE: live log stream ──────────────────────────────────────────────
@app.route("/api/stream")
def api_stream():
    def generate():
        yield 'data: {"type":"connected","msg":"Stream ready"}\n\n'
        try:
            while True:
                try:
                    entry = log_q.get(timeout=25)
                    yield f"data: {json.dumps(entry)}\n\n"
                    if entry.get("type") == "done":
                        break
                except queue.Empty:
                    yield 'data: {"type":"ping"}\n\n'
        except GeneratorExit:
            # Client disconnected — drain the queue so the pipeline thread
            # never blocks on a full queue waiting for a dead consumer.
            print("⚠️  SSE client disconnected — draining log queue.")
            while not log_q.empty():
                try:
                    log_q.get_nowait()
                except queue.Empty:
                    break
    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── API: start the pipeline ───────────────────────────────────────────
@app.route("/api/run", methods=["POST"])
def api_run():
    if state["running"]:
        return jsonify({"error": "Pipeline already running"}), 409
    state.update({
        "running": True, "step": 0, "done": False,
        "success": False, "attempt": 0,
        "logs": [], "suspicious": [], "attempts": [],
        "original_code": BUGGY_FILE.read_text(encoding="utf-8") if BUGGY_FILE.exists() else "",
        "fixed_code": "",
    })
    while not log_q.empty():
        try: log_q.get_nowait()
        except: pass
    threading.Thread(target=_pipeline_thread, daemon=True).start()
    return jsonify({"status": "started"})


# ── Pipeline thread ───────────────────────────────────────────────────
def _pipeline_thread():
    MAX_ATTEMPTS = 3
    TOP_K = 5
    try:
        from core.run_and_collect import collect_spectrum
        from core.sbfl_localiser  import rank_suspicious_lines
        from core.model_runner    import run_testmate

        emit("info", "🚀 TestMate APR pipeline started")
        emit("info", f"📂 Target file: {BUGGY_FILE.name}")

        # Snapshot original so we can fully restore if all attempts fail
        original_snapshot = BUGGY_FILE.read_text(encoding="utf-8") if BUGGY_FILE.exists() else ""

        for attempt in range(1, MAX_ATTEMPTS + 1):
            state["attempt"] = attempt
            emit("info", f"━━━ Attempt {attempt}/{MAX_ATTEMPTS} ━━━")

            # Step 1: Run tests
            state["step"] = 1
            emit("info", "🧪 Step 1 — Running pytest…")
            r = subprocess.run(
                ["pytest", str(TEST_FILE), "-v", "--tb=short", "--no-header"],
                capture_output=True, text=True, cwd=str(PROJECT_ROOT)
            )
            test_out = r.stdout + r.stderr

            if r.returncode == 0:
                emit("success", "✅ All tests PASSED — bug is fixed!")
                state["step"] = 6
                state["success"] = True
                state["fixed_code"] = BUGGY_FILE.read_text(encoding="utf-8")
                state["attempts"].append({"n": attempt, "status": "success",
                    "patch": "Patch applied", "result": "All tests passed ✅"})
                break

            failed_n = test_out.count(" FAILED")
            emit("warn", f"❌ {failed_n} test(s) failed")

            # Step 2: SBFL
            state["step"] = 2
            emit("info", "🔍 Step 2 — SBFL fault localisation…")
            try:
                spectrum, fail_out = collect_spectrum()
                ranked = rank_suspicious_lines(spectrum)
            except Exception as e:
                emit("warn", f"⚠️  SBFL error: {e}")
                fail_out = test_out
                ranked   = []

            if ranked:
                suspicious_lines = [ln for ln, _ in ranked[:TOP_K]]
                code_lines = BUGGY_FILE.read_text(encoding="utf-8").splitlines()
                state["suspicious"] = [
                    {"line": ln, "score": round(sc, 3),
                     "code": code_lines[ln - 1] if 0 < ln <= len(code_lines) else ""}
                    for ln, sc in ranked[:TOP_K]
                ]
                emit("info", f"🔍 Suspicious lines: {suspicious_lines}")
            else:
                fail_out = test_out
                emit("warn", "⚠️  No SBFL data — sending full code to model")

            # Step 3: Build prompt
            state["step"] = 3
            emit("info", "📝 Step 3 — Building repair prompt…")
            failure_lines = [
                l.strip() for l in fail_out.split("\n")
                if "E   " in l or "FAILED" in l or "Error" in l
            ]
            trace_text = "\n".join(failure_lines[:4]) or "AssertionError: Logic verification failed."
            full_code  = BUGGY_FILE.read_text(encoding="utf-8")
            prompt = (
                f"ISSUE: Fix the logic bug causing the test failure.\n\n"
                f"TRACE:\n{trace_text}\n\n"
                f"BUGGY:\n{full_code}"
            )

            # Step 4: Call model
            state["step"] = 4
            emit("info", "🤖 Step 4 — Calling TestMate model…")
            try:
                new_code = run_testmate(prompt)
            except Exception as e:
                emit("error", f"❌ Model error: {e}")
                state["attempts"].append({"n": attempt, "status": "fail",
                    "patch": "Model call failed", "result": str(e)})
                continue

            if not new_code.strip():
                emit("error", "❌ Model returned empty output")
                state["attempts"].append({"n": attempt, "status": "fail",
                    "patch": "Empty output", "result": "No code generated"})
                continue

            (ARTIFACTS / f"model_output_attempt_{attempt}.py").write_text(new_code, encoding="utf-8")
            emit("info", f"💾 Saved: artifacts/model_output_attempt_{attempt}.py")

            # Step 5: AST patch
            state["step"] = 5
            emit("info", "🔧 Step 5 — Applying AST patch…")
            if not _apply_ast_patch(new_code):
                emit("error", f"❌ Patch {attempt} rejected")
                state["attempts"].append({"n": attempt, "status": "fail",
                    "patch": "AST patch failed", "result": "Patch rejected"})
                continue

            # Step 6: Verify
            state["step"] = 6
            emit("info", "🔁 Step 6 — Re-running tests to verify patch…")
            r2 = subprocess.run(
                ["pytest", str(TEST_FILE), "-v", "--tb=short", "--no-header"],
                capture_output=True, text=True, cwd=str(PROJECT_ROOT)
            )
            if r2.returncode == 0:
                emit("success", "✅ All tests PASSED — bug is fixed!")
                state["success"] = True
                state["fixed_code"] = BUGGY_FILE.read_text(encoding="utf-8")
                state["attempts"].append({"n": attempt, "status": "success",
                    "patch": "Patch applied & verified", "result": "All tests passed ✅"})
                break
            else:
                n2 = (r2.stdout + r2.stderr).count(" FAILED")
                emit("warn", f"⚠️  Patch applied but {n2} test(s) still failing")
                state["attempts"].append({"n": attempt, "status": "fail",
                    "patch": "Patch applied", "result": f"{n2} test(s) still failing"})
        else:
            emit("error", f"❌ All {MAX_ATTEMPTS} attempts failed")
            if original_snapshot:
                BUGGY_FILE.write_text(original_snapshot, encoding="utf-8")
                emit("warn", "🔄 Original file restored — no partial patches left")
            state["success"] = False

    except Exception as e:
        import traceback
        emit("error", f"💥 Pipeline crashed: {e}")
        emit("error", traceback.format_exc())
    finally:
        state["step"] = 6
        state["done"]    = True
        state["running"] = False
        emit("done", "Pipeline finished")


# ── AST patch helper ──────────────────────────────────────────────────
def _apply_ast_patch(new_code: str) -> bool:
    # Read BEFORE the try block so rollback is always available
    original = BUGGY_FILE.read_text(encoding="utf-8")
    try:
        m = re.search(r"```(?:python)?(.*?)```", new_code, re.DOTALL | re.IGNORECASE)
        if m:
            clean = m.group(1)
        else:
            lines = new_code.split("\n")
            start = 0
            for i, l in enumerate(lines):
                if l.strip().startswith(("def ", "class ", "@")):
                    start = i
                    break
            clean = "\n".join(lines[start:])
        clean = clean.replace("⚠️", "").replace("⚠", "").strip()

        new_ast = ast.parse(clean)
        new_fns = {n.name: n for n in new_ast.body
                   if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
        if not new_fns:
            emit("warn", "⚠️  No functions/classes in patch")
            return False

        orig_ast = ast.parse(original)
        modified = False
        for i, node in enumerate(orig_ast.body):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in new_fns:
                orig_ast.body[i] = new_fns[node.name]
                modified = True
                emit("info", f"   🔧 Replaced '{node.name}'")

        if not modified:
            emit("warn", "⚠️  Function names didn't match — patch rejected")
            return False

        BUGGY_FILE.write_text(ast.unparse(orig_ast), encoding="utf-8")
        emit("success", "✅ Patch written to buggy_code.py")
        return True

    except SyntaxError as e:
        emit("error", f"❌ Invalid syntax in patch: {e}")
        BUGGY_FILE.write_text(original, encoding="utf-8")
        return False
    except Exception as e:
        emit("error", f"❌ Patch error: {e}")
        BUGGY_FILE.write_text(original, encoding="utf-8")
        return False


# ── API: code coverage ───────────────────────────────────────────────
@app.route("/api/coverage")
def api_coverage():
    """Run pytest with coverage.py and return per-function + overall stats."""
    cov_json = PROJECT_ROOT / "coverage.json"
    try:
        r = subprocess.run(
            [
                sys.executable, "-m", "pytest", str(TEST_FILE),
                f"--cov={BUGGY_FILE.stem}",
                "--cov-report=json",
                "--cov-report=term-missing",
                "-q", "--no-header",
                f"--rootdir={PROJECT_ROOT}",
            ],
            capture_output=True, text=True, cwd=str(PROJECT_ROOT)
        )
        if not cov_json.exists():
            return jsonify({"error": "coverage.json not generated", "raw": r.stdout + r.stderr}), 500

        data = json.loads(cov_json.read_text(encoding="utf-8"))
        file_data = data.get("files", {}).get("buggy_code.py", {})
        summary   = file_data.get("summary", {})
        functions = file_data.get("functions", {})

        fn_results = []
        for fn_name, fn_info in functions.items():
            s = fn_info.get("summary", {})
            fn_results.append({
                "name":    fn_name,
                "covered": s.get("covered_lines", 0),
                "total":   s.get("num_statements", 0),
                "pct":     round(s.get("percent_covered", 0), 1),
                "missing": fn_info.get("missing_lines", []),
            })

        return jsonify({
            "overall_pct":   round(summary.get("percent_covered", 0), 1),
            "covered_lines": summary.get("covered_lines", 0),
            "total_lines":   summary.get("num_statements", 0),
            "missing_lines": file_data.get("missing_lines", []),
            "functions":     fn_results,
            "raw":           r.stdout,
        })
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


# ── Mutation testing (AST-based, Windows-compatible) ──────────────────
class _MutantCollector(ast.NodeTransformer):
    """Collect all mutation opportunities without applying them."""
    BINOP_MUTATIONS = {
        ast.Add:      [ast.Sub, ast.Mult],
        ast.Sub:      [ast.Add, ast.Mult],
        ast.Mult:     [ast.Add, ast.Sub],
        ast.Div:      [ast.FloorDiv, ast.Mult],
        ast.FloorDiv: [ast.Div, ast.Add],
        ast.Mod:      [ast.Add, ast.Mult],
    }
    CMP_MUTATIONS = {
        ast.Lt:    [ast.LtE, ast.Gt,  ast.GtE, ast.Eq],
        ast.LtE:   [ast.Lt,  ast.Gt,  ast.GtE, ast.Eq],
        ast.Gt:    [ast.GtE, ast.Lt,  ast.LtE, ast.Eq],
        ast.GtE:   [ast.Gt,  ast.Lt,  ast.LtE, ast.Eq],
        ast.Eq:    [ast.NotEq, ast.Lt, ast.Gt],
        ast.NotEq: [ast.Eq,   ast.Lt, ast.Gt],
    }

    def __init__(self):
        self.mutations = []  # list of (node_id, description, mutated_source)
        self._src_lines = []
        self._orig_tree = None

    def collect(self, source: str):
        self._src_lines = source.splitlines()
        self._orig_tree = ast.parse(source)
        self.visit(self._orig_tree)

    def _make_mutant(self, orig_node, replacement_node, label: str):
        """Clone the tree, swap node, unparse to source."""
        try:
            tree_copy = copy.deepcopy(self._orig_tree)
            # Walk copy to find matching node by position
            for node in ast.walk(tree_copy):
                if (
                    type(node) is type(orig_node)
                    and getattr(node, 'lineno', None) == getattr(orig_node, 'lineno', None)
                    and getattr(node, 'col_offset', None) == getattr(orig_node, 'col_offset', None)
                ):
                    if isinstance(node, ast.BinOp) and isinstance(orig_node, ast.BinOp):
                        node.op = replacement_node()
                        break
                    elif isinstance(node, ast.Compare) and isinstance(orig_node, ast.Compare):
                        node.ops = [type(replacement_node)() if type(o) is type(replacement_node) else o
                                    for o in node.ops]
                        # Simpler: replace first matching op type
                        for idx, op in enumerate(node.ops):
                            if type(op) is type(orig_node.ops[0]) if hasattr(orig_node, 'ops') else False:
                                node.ops[idx] = replacement_node()
                                break
                        else:
                            # Replace all ops
                            node.ops = [replacement_node() if type(o).__name__ == type(orig_node.ops[0] if hasattr(orig_node, 'ops') else op).__name__ else o for o in node.ops]
                        break
            mutant_src = ast.unparse(tree_copy)
            ln = getattr(orig_node, 'lineno', '?')
            orig_line = self._src_lines[ln - 1].strip() if isinstance(ln, int) and 0 < ln <= len(self._src_lines) else ''
            self.mutations.append({
                "label":     label,
                "line":      ln,
                "orig_line": orig_line,
                "source":    mutant_src,
            })
        except Exception:
            pass

    def visit_BinOp(self, node):
        self.generic_visit(node)
        for orig_type, replacements in self.BINOP_MUTATIONS.items():
            if isinstance(node.op, orig_type):
                for rep in replacements:
                    self._make_mutant(node, rep, f"BinOp {orig_type.__name__}→{rep.__name__} (line {node.lineno})")
        return node

    def visit_Compare(self, node):
        self.generic_visit(node)
        for op in node.ops:
            for orig_type, replacements in self.CMP_MUTATIONS.items():
                if isinstance(op, orig_type):
                    for rep in replacements:
                        self._make_mutant(node, rep(), f"Cmp {orig_type.__name__}→{rep.__name__} (line {node.lineno})")
        return node

    def visit_UnaryOp(self, node):
        self.generic_visit(node)
        if isinstance(node.op, ast.USub):
            try:
                tree_copy = copy.deepcopy(self._orig_tree)
                for n in ast.walk(tree_copy):
                    if (
                        isinstance(n, ast.UnaryOp)
                        and isinstance(n.op, ast.USub)
                        and getattr(n, 'lineno', None) == getattr(node, 'lineno', None)
                        and getattr(n, 'col_offset', None) == getattr(node, 'col_offset', None)
                    ):
                        # Replace -x with x
                        parent_info = [(p, f) for p in ast.walk(tree_copy)
                                       for f, v in ast.iter_fields(p)
                                       if v is n]
                        break
                mutant_src = ast.unparse(tree_copy)
                ln = getattr(node, 'lineno', '?')
                orig_line = self._src_lines[ln-1].strip() if isinstance(ln, int) and 0 < ln <= len(self._src_lines) else ''
                self.mutations.append({
                    "label":     f"UnaryOp USub→UAdd (line {ln})",
                    "line":      ln,
                    "orig_line": orig_line,
                    "source":    mutant_src,
                })
            except Exception:
                pass
        return node


def _run_mutation_tests(mutant_src: str) -> bool:
    """Write mutant to a temp file replacing buggy_code.py, run tests, restore."""
    original = BUGGY_FILE.read_text(encoding="utf-8")
    try:
        BUGGY_FILE.write_text(mutant_src, encoding="utf-8")
        r = subprocess.run(
            [sys.executable, "-m", "pytest", str(TEST_FILE), "-x", "-q", "--no-header", "--tb=no"],
            capture_output=True, text=True, cwd=str(PROJECT_ROOT), timeout=30
        )
        return r.returncode != 0  # True = mutant was KILLED (tests caught it)
    except subprocess.TimeoutExpired:
        return True  # timeout = consider killed
    finally:
        BUGGY_FILE.write_text(original, encoding="utf-8")


@app.route("/api/mutation")
def api_mutation():
    """AST-based mutation testing. Returns mutation score and per-mutant results."""
    if not BUGGY_FILE.exists():
        return jsonify({"error": "buggy_code.py not found"}), 404
    try:
        source = BUGGY_FILE.read_text(encoding="utf-8")
        collector = _MutantCollector()
        collector.collect(source)
        mutants = collector.mutations

        if not mutants:
            return jsonify({"score": 100.0, "killed": 0, "survived": 0, "total": 0, "mutants": []})

        results = []
        killed = 0
        for i, m in enumerate(mutants):
            is_killed = _run_mutation_tests(m["source"])
            status = "killed" if is_killed else "survived"
            if is_killed:
                killed += 1
            results.append({
                "id":       i + 1,
                "label":    m["label"],
                "line":     m["line"],
                "orig":     m["orig_line"],
                "status":   status,
            })

        total    = len(results)
        survived = total - killed
        score    = round((killed / total) * 100, 1) if total > 0 else 0.0

        return jsonify({
            "score":    score,
            "killed":   killed,
            "survived": survived,
            "total":    total,
            "mutants":  results,
        })
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


# ── Main page ─────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("  TestMate APR — Web UI")
    print(f"  ROOT  : {PROJECT_ROOT}")
    print(f"  buggy : {BUGGY_FILE}  {'✅' if BUGGY_FILE.exists() else '❌ NOT FOUND'}")
    print(f"  tests : {TEST_FILE}  {'✅' if TEST_FILE.exists() else '❌ NOT FOUND'}")
    print("  URL   : http://localhost:5000")
    print("=" * 55 + "\n")
    app.run(debug=False, port=5000, threaded=True, use_reloader=False)