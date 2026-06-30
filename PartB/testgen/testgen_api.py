"""
Part B Test Generation Pipeline API
Wraps autonomous_loop into a clean callable function.
"""
from __future__ import annotations

import re
import sys
import logging
from pathlib import Path
from typing import Any, Callable, Optional

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

logger = logging.getLogger(__name__)

GEN_TESTS_DIR = _HERE / "generated_tests"

# ── Stop-words for coverage keyword extraction ────────────────────────────
_COVERAGE_NOISE = {
    "the", "system", "shall", "must", "accept", "return", "provide",
    "support", "allow", "when", "that", "from", "with", "this",
    "have", "will", "each", "only", "than", "into", "and", "for",
    "not", "any", "all", "its", "may", "can", "are", "was", "has",
    "been", "also", "does", "use", "used", "set", "get", "via",
}


def check_requirement_coverage(requirements: list[dict], test_code: str) -> dict:
    """
    Measure how many SRS requirements are touched by the generated test code.

    A requirement is "covered" if at least 2 of its meaningful keywords appear
    in the test code. Returns a summary dict suitable for the UI.
    """
    req_only = [r for r in requirements if isinstance(r, dict) and r.get("label") == 1]
    if not req_only or not test_code:
        return {"total": len(req_only), "covered": 0, "pct": 0.0, "uncovered_count": 0}

    test_lower = test_code.lower()
    covered_count = 0

    for r in req_only:
        text = r.get("text", "")
        # Extract identifiers: camelCase, snake_case, PascalCase, error names
        keywords = re.findall(r"\b[A-Za-z][a-zA-Z_]{2,}\b", text)
        keywords = [k.lower() for k in keywords if k.lower() not in _COVERAGE_NOISE]
        keywords = keywords[:6]
        if not keywords:
            continue
        hits = sum(1 for kw in keywords if kw in test_lower)
        if hits >= min(2, len(keywords)):
            covered_count += 1

    total = len(req_only)
    return {
        "total":           total,
        "covered":         covered_count,
        "pct":             round(covered_count / total * 100, 1) if total > 0 else 0.0,
        "uncovered_count": total - covered_count,
    }


def execute_part_b(
    target_file: str,
    requirements: Optional[list[dict]] = None,
    import_path: Optional[str] = None,
    deep_scan: bool = False,
    max_retries: int = 3,
    log_callback: Optional[Callable] = None,
    plan_mode: bool = False,
    use_base_only: bool = False,
    quality_mode: str = "fast",
    priming_examples: str = "",
    model: Any = None,
    tokenizer: Any = None,
    auto_repair: bool = False,
    use_docker: bool = False,
    skip_bes_gate: bool = False,   # keep gate active; threshold is calibrated in main.py
    max_seconds: float = 300.0,    # hard 5-min per-file timeout prevents audit hangs
) -> dict:
    """
    Run the self-correcting test generation loop for a single target file.

    Args:
        target_file:    Absolute path to the Python file to test.
        requirements:   Part A's labeled requirements.
        import_path:    Package import path (e.g. 'requests.auth').
        deep_scan:      Scan broader repo context.
        max_retries:    Max self-correction iterations.
        log_callback:   GUI callback(event_type, *args).
        plan_mode:      Generate a test plan before writing code.
        use_base_only:  Skip LoRA — raw 4-bit base Qwen.
        auto_repair:    If True, invoke PartC on confirmed/suspected bugs
                        detected after test generation (opt-in).
        use_docker:     If True, build a per-repo Docker image and run every
                        pytest invocation inside it (env isolation — solves
                        host-package version mismatches).  Falls back to host
                        pytest if Docker is unavailable or the build fails.
        model/tokenizer: Pre-loaded model (loaded here if None).
    """
    from main import autonomous_loop, load_model as _load_model

    _model = model
    _tokenizer = tokenizer
    _loaded_here = False

    if _model is None or _tokenizer is None:
        logger.info("[B] Loading model (use_lora=%s)", not use_base_only)
        _model, _tokenizer = _load_model(use_lora=not use_base_only)
        _loaded_here = True

    req_count = sum(1 for r in (requirements or []) if isinstance(r, dict) and r.get("label") == 1)
    logger.info("[B] %s: plan_mode=%s, auto_repair=%s, use_docker=%s, %d/%d SRS requirements",
                Path(target_file).name, plan_mode, auto_repair, use_docker, req_count, len(requirements or []))

    # Build (or reuse cached) Docker image when use_docker=True so the inner
    # loop can mount the test file at runtime without per-iteration builds.
    docker_image: Optional[str] = None
    if use_docker:
        try:
            from docker_runner import ensure_repo_image  # type: ignore[import]
            # Walk up from target_file until __init__.py disappears — that's
            # the repo root pytest sees.
            _root = str(Path(target_file).parent.resolve())
            while (Path(_root) / "__init__.py").exists():
                _parent = str(Path(_root).parent)
                if _parent == _root:
                    break
                _root = _parent
            _repo_name = Path(_root).name or "project"
            docker_image = ensure_repo_image(_root, _repo_name)
            logger.info("[B] Docker enabled: image=%s repo=%s", docker_image, _root)
        except Exception as _docker_err:
            logger.warning("[B] Docker setup failed (%s) — falling back to host pytest", _docker_err)
            docker_image = None

    try:
        GEN_TESTS_DIR.mkdir(exist_ok=True)

        success = autonomous_loop(
            _model,
            _tokenizer,
            target_file,
            import_path=import_path,
            deep_scan=deep_scan,
            max_retries=max_retries,
            log_callback=log_callback,
            plan_mode=plan_mode,
            srs_requirements=requirements,
            priming_examples=priming_examples,
            auto_repair=False,  # repair handled by orchestrator via adapter swap
            docker_image=docker_image,
            skip_bes_gate=skip_bes_gate,
            max_seconds=max_seconds,
        )

        # Locate generated test file.
        basename = Path(target_file).stem
        candidates = [
            GEN_TESTS_DIR / f"test_{basename}_testmate.py",
            Path(target_file).parent / f"test_{basename}_testmate.py",
            GEN_TESTS_DIR / f"test_{basename}.py",
            Path(target_file).parent / f"test_{basename}.py",
        ]
        test_path = next((p for p in candidates if p.exists()), None)
        test_filename = test_path.name if test_path else f"test_{basename}_testmate.py"
        test_code = test_path.read_text(encoding="utf-8") if test_path else ""

        # Read bug reports (repair is handled by the orchestrator after adapter swap)
        _bug_report_path_gen = GEN_TESTS_DIR / "bug_reports.jsonl"
        _bug_report_path_src = Path(target_file).parent / "bug_reports.jsonl"
        bug_report_path = _bug_report_path_src if _bug_report_path_src.exists() else _bug_report_path_gen
        bug_reports: list[dict] = []
        if bug_report_path.exists():
            for line in bug_report_path.read_text(encoding="utf-8").splitlines():
                try:
                    entry = __import__("json").loads(line)
                    if entry.get("source_file") == Path(target_file).name:
                        bug_reports.append(entry)
                except Exception:
                    pass

        coverage = check_requirement_coverage(requirements or [], test_code)
        logger.info("[B] SRS coverage: %d/%d (%.0f%%)",
                    coverage["covered"], coverage["total"], coverage["pct"])

        # ── Real line coverage via coverage.py ──────────────────────────────
        line_coverage: float = 0.0
        coverage_map: dict = {}
        if test_path and test_path.exists():
            try:
                import subprocess as _sp, json as _json, os as _os
                src_path = Path(target_file)
                work_dir = src_path.parent          # always run from source dir
                cov_out  = work_dir / "_testmate_cov.json"
                # Make source dir importable so the test can find the module
                _env = _os.environ.copy()
                _env["PYTHONPATH"] = str(work_dir) + _os.pathsep + _env.get("PYTHONPATH", "")
                r = _sp.run(
                    [
                        sys.executable, "-m", "pytest", str(test_path),
                        f"--cov={src_path.stem}",
                        f"--cov-report=json:{cov_out}",
                        "-q", "--no-header", "--tb=no",
                    ],
                    capture_output=True, text=True,
                    cwd=str(work_dir), env=_env, timeout=60
                )
                logger.debug("[B] coverage stdout: %s", r.stdout[-500:] if r.stdout else "")
                logger.debug("[B] coverage stderr: %s", r.stderr[-300:] if r.stderr else "")
                if cov_out.exists():
                    data     = _json.loads(cov_out.read_text(encoding="utf-8"))
                    # Find our file's entry
                    fdata: dict = {}
                    for fname, finfo in data.get("files", {}).items():
                        if Path(fname).stem == src_path.stem:
                            fdata = finfo; break
                    if not fdata and data.get("files"):
                        fdata = next(iter(data["files"].values()))
                    if fdata:
                        summ          = fdata.get("summary", {})
                        line_coverage = round(summ.get("percent_covered", 0.0), 1)
                        source_text   = ""
                        try: source_text = src_path.read_text(encoding="utf-8")
                        except Exception: pass
                        coverage_map[src_path.name] = {
                            "source":            source_text,
                            "covered_lines":     sorted(fdata.get("executed_lines", [])),
                            "uncovered_lines":   sorted(fdata.get("missing_lines", [])),
                            "line_coverage_pct": line_coverage,
                            "num_statements":    summ.get("num_statements", 0),
                        }
                    cov_out.unlink(missing_ok=True)
            except Exception as _cov_err:
                logger.warning("[B] Coverage measurement failed: %s", _cov_err)

        # ── AST mutation score ───────────────────────────────────────────────
        mutation_score: float = 0.0
        mutation_survivors: list[dict] = []
        if test_path and test_path.exists():
            try:
                import ast as _ast, copy as _copy, subprocess as _sp
                src_path   = Path(target_file)
                original   = src_path.read_text(encoding="utf-8")
                BINOP_MAP  = {
                    _ast.Add: [_ast.Sub, _ast.Mult], _ast.Sub: [_ast.Add, _ast.Mult],
                    _ast.Mult: [_ast.Add, _ast.Sub], _ast.Div: [_ast.FloorDiv, _ast.Mult],
                    _ast.FloorDiv: [_ast.Div, _ast.Add], _ast.Mod: [_ast.Add, _ast.Mult],
                }
                CMP_MAP = {
                    _ast.Lt: [_ast.LtE, _ast.Gt], _ast.LtE: [_ast.Lt, _ast.GtE],
                    _ast.Gt: [_ast.GtE, _ast.Lt], _ast.GtE: [_ast.Gt, _ast.LtE],
                    _ast.Eq: [_ast.NotEq, _ast.Lt], _ast.NotEq: [_ast.Eq, _ast.Gt],
                }
                src_lines = original.splitlines()
                orig_tree = _ast.parse(original)
                mutants: list[dict] = []

                def _collect(orig_node, tc, label):
                    try:
                        ln = getattr(orig_node, "lineno", "?")
                        orig_line = src_lines[ln - 1].strip() if isinstance(ln, int) and 0 < ln <= len(src_lines) else ""
                        mutants.append({"source": _ast.unparse(tc), "line": ln,
                                        "label": label, "orig": orig_line})
                    except Exception: pass

                for node in _ast.walk(orig_tree):
                    if isinstance(node, _ast.BinOp):
                        for ot, reps in BINOP_MAP.items():
                            if isinstance(node.op, ot):
                                for rep in reps:
                                    tc = _copy.deepcopy(orig_tree)
                                    for n in _ast.walk(tc):
                                        if (isinstance(n, _ast.BinOp) and isinstance(n.op, ot)
                                                and getattr(n, "lineno", None) == getattr(node, "lineno", None)
                                                and getattr(n, "col_offset", None) == getattr(node, "col_offset", None)):
                                            n.op = rep(); break
                                    _collect(node, tc, f"BinOp {ot.__name__}→{rep.__name__} (line {node.lineno})")
                    if isinstance(node, _ast.Compare):
                        for op in node.ops:
                            for ot, reps in CMP_MAP.items():
                                if isinstance(op, ot):
                                    for rep in reps:
                                        tc = _copy.deepcopy(orig_tree)
                                        for n in _ast.walk(tc):
                                            if (isinstance(n, _ast.Compare)
                                                    and getattr(n, "lineno", None) == getattr(node, "lineno", None)
                                                    and getattr(n, "col_offset", None) == getattr(node, "col_offset", None)):
                                                n.ops = [rep() if type(o) is ot else o for o in n.ops]; break
                                        _collect(node, tc, f"Cmp {ot.__name__}→{rep.__name__} (line {node.lineno})")

                import os as _os
                _mut_env = _os.environ.copy()
                _mut_env["PYTHONPATH"] = str(src_path.parent) + _os.pathsep + _mut_env.get("PYTHONPATH", "")
                killed = 0
                for m in mutants:
                    try:
                        src_path.write_text(m["source"], encoding="utf-8")
                        r = _sp.run(
                            [sys.executable, "-m", "pytest", str(test_path),
                             "-x", "-q", "--no-header", "--tb=no"],
                            capture_output=True, text=True,
                            cwd=str(src_path.parent), env=_mut_env, timeout=30
                        )
                        is_killed = (r.returncode != 0)
                    except _sp.TimeoutExpired:
                        is_killed = True
                    except Exception:
                        is_killed = False
                    finally:
                        src_path.write_text(original, encoding="utf-8")

                    if is_killed:
                        killed += 1
                    elif m not in mutation_survivors:
                        # Record surviving mutants as bug-report entries for MutationsView
                        mutation_survivors.append({
                            "bug_type":    "mutation_survivor",
                            "target":      f"line {m['line']}",
                            "description": m["label"],
                            "evidence":    m["orig"],
                            "confidence":  "Medium",
                            "source_file": src_path.name,
                        })

                total         = len(mutants)
                mutation_score = round((killed / total) * 100, 1) if total > 0 else 0.0
                logger.info("[B] Mutation score: %.1f%% (%d/%d killed)", mutation_score, killed, total)
            except Exception as _mut_err:
                logger.warning("[B] Mutation testing failed: %s", _mut_err)

        return {
            "success":        success,
            "test_file":      test_filename,
            "test_code":      test_code,
            "test_path":      str(test_path) if test_path else "",
            "target":         target_file,
            "srs_coverage":   coverage,
            "bug_reports":    bug_reports + mutation_survivors,
            "repairs":        [b.get("repair", {}) for b in bug_reports if b.get("repair")],
            # ── New: real line coverage & mutation score ──
            "line_coverage":  line_coverage,
            "coverage_map":   coverage_map,
            "mutation_score": mutation_score,
        }

    finally:
        if _loaded_here:
            import gc, torch
            del _model, _tokenizer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()


# ── Interactive chat refinement ─────────────────────────────────────────────

_REFINE_SYSTEM = (
    "You are TestMate's test editor. You revise an EXISTING pytest file according "
    "to the user's instruction.\n"
    "RULES:\n"
    "- Output the COMPLETE updated test file wrapped in <test>...</test> tags.\n"
    "- On the line BEFORE the <test> block, write ONE short sentence describing what you changed.\n"
    "- Keep every existing passing test unless the instruction says to change or remove it.\n"
    "- Use pytest. Import the module under test exactly as the current file does. "
    "Assert concrete expected VALUES (never `assert True`).\n"
    "- If the user's message is a QUESTION rather than an edit request, do NOT output a "
    "<test> block — reply in one short sentence telling them what concrete change you can make.\n"
)


def refine_test_file(
    target_file: str,
    test_path: str,
    test_code: str,
    instruction: str,
    model: Any,
    tokenizer: Any,
    history: Optional[list] = None,
    max_retries: int = 3,
    log_callback: Optional[Callable] = None,
    use_docker: bool = False,
) -> dict:
    """Apply a single natural-language edit instruction to an existing test file.

    Reuses the same model + helpers as the generation loop: builds an edit
    prompt, generates a rewrite, runs it through `quick_quality_check` and
    `run_pytest`, and feeds any failure back for up to `max_retries` repair
    rounds. On success the file at `test_path` is overwritten; on total failure
    the original content is restored and the file is left unchanged.

    Returns a dict shaped like `execute_part_b()` plus:
        edited:     bool — whether the on-disk file was changed.
        chat_reply: str  — one-line assistant message for the chat UI.
    """
    from main import (
        generate_test, extract_test_code, quick_quality_check, run_pytest,
    )

    def _status(status: str, detail: str = "") -> None:
        if log_callback:
            try:
                log_callback("ai_status", status, detail)
            except Exception:
                pass

    target = Path(target_file)
    tpath = Path(test_path) if test_path else None

    try:
        source = target.read_text(encoding="utf-8") if target.exists() else ""
    except Exception:
        source = ""
    src_ctx = source if len(source) <= 6000 else source[:6000] + "\n# … (truncated)"

    messages: list[dict] = [{"role": "system", "content": _REFINE_SYSTEM}]
    # Prior turns for multi-turn context (cap each to keep the prompt small).
    for h in (history or []):
        if isinstance(h, dict) and h.get("role") in ("user", "assistant") and h.get("content"):
            messages.append({"role": h["role"], "content": str(h["content"])[:2000]})
    messages.append({
        "role": "user",
        "content": (
            f"Module under test ({target.name}):\n```python\n{src_ctx}\n```\n\n"
            f"Current test file ({tpath.name if tpath else 'test.py'}):\n"
            f"```python\n{test_code}\n```\n\n"
            f"Instruction: {instruction}"
        ),
    })

    # Optional Docker isolation (mirrors execute_part_b's image build).
    docker_image: Optional[str] = None
    if use_docker:
        try:
            from docker_runner import ensure_repo_image  # type: ignore[import]
            _root = str(target.parent.resolve())
            while (Path(_root) / "__init__.py").exists():
                _parent = str(Path(_root).parent)
                if _parent == _root:
                    break
                _root = _parent
            docker_image = ensure_repo_image(_root, Path(_root).name or "project")
        except Exception as _e:
            logger.warning("[chat] Docker setup failed (%s) — host pytest", _e)
            docker_image = None

    def _result(success: bool, edited: bool, code: str, reply: str, error: str = "") -> dict:
        return {
            "success":   success,
            "edited":    edited,
            "chat_reply": reply,
            "test_file": tpath.name if tpath else (f"test_{target.stem}_testmate.py"),
            "test_code": code,
            "test_path": str(tpath) if tpath else "",
            "target":    target_file,
            "error":     error,
        }

    _status("Thinking", "revising the tests")
    raw = generate_test(model, tokenizer, messages, temperature=0.2)
    reply = (raw.split("<test>")[0].strip() if "<test>" in raw else raw.strip())[:300]

    # No edit produced — treat as a question/refusal, leave the file untouched.
    # Don't surface the model's free-form reply here: a local 7B answering a
    # testing question directly reads as hallucination (it may be wrong) and
    # breaks the "edits only, no Q&A" guarantee. Always redirect instead.
    if "<test>" not in raw:
        return _result(False, False, test_code,
                       "I only edit tests — tell me what to change.")

    candidate = extract_test_code(raw)
    original = test_code
    last_error = ""

    for attempt in range(max_retries + 1):
        ok_q, reason = quick_quality_check(candidate)
        if not ok_q:
            last_error = f"quality check: {reason}"
            if attempt < max_retries:
                _status("Repairing", reason)
                messages.append({"role": "assistant", "content": raw})
                messages.append({"role": "user", "content":
                    f"That test has a problem: {reason}. "
                    f"Fix it and output the COMPLETE file in <test>...</test>."})
                raw = generate_test(model, tokenizer, messages, temperature=0.3)
                candidate = extract_test_code(raw)
                continue
            break

        if not tpath:
            # Nowhere to validate against — quality passed, return as-is.
            return _result(True, False, candidate, reply or "Updated the tests.")

        try:
            tpath.write_text(candidate, encoding="utf-8")
        except Exception as e:
            return _result(False, False, original, f"Could not write the file: {e}", str(e))

        _status("Validating", f"running pytest (attempt {attempt + 1})")
        try:
            passed, output = run_pytest(str(tpath), str(target), docker_image=docker_image)
        except Exception as e:
            passed, output = False, str(e)

        if passed:
            return _result(True, True, candidate, reply or "Updated the tests — they pass.")

        last_error = (output or "tests failed")[-1500:]
        if attempt < max_retries:
            _status("Repairing", "the edited test failed — fixing")
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content":
                f"The test failed when run:\n{last_error}\n"
                f"Fix it and output the COMPLETE file in <test>...</test>."})
            raw = generate_test(model, tokenizer, messages, temperature=0.4)
            candidate = extract_test_code(raw)

    # All attempts failed — restore the original file so nothing is broken.
    if tpath:
        try:
            tpath.write_text(original, encoding="utf-8")
        except Exception:
            pass
    return _result(
        False, False, original,
        (reply + " — but that edit didn't pass, so I left the test unchanged.").strip(),
        last_error,
    )
