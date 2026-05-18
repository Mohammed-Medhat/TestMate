"""
Combined pipeline orchestrator — Part A → Part B in sequence.

Part A runs ONCE per project (SRS + README) → produces requirements + scenarios.
Part B runs PER FILE — for each selected target, the most-relevant requirements
(via hybrid keyword + semantic matcher) are seeded into the RAG memory store
before generation.

The GPU is flushed between Part A's model unload and Part B's model load.
"""
from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).parent.parent
_PART_A_SRS = _ROOT / "PartA" / "srs_pipeline"
_PART_A_README = _ROOT / "PartA" / "readme_extractor"
_PART_B_TESTGEN = _ROOT / "PartB" / "testgen"


def _log(msg: str, callback: Optional[Callable] = None, level: str = "info") -> None:
    getattr(logger, level)(msg)
    if callback:
        callback("log", level, msg)


def execute_combined(
    srs_path: Optional[str],
    target_files: list[dict],
    readme_content: Optional[str] = None,
    user_input: Optional[dict] = None,
    deep_scan: bool = False,
    max_retries: int = 3,
    plan_mode: bool = False,
    top_k_requirements: int = 10,
    use_base_only: bool = False,
    quality_mode: str = "fast",          # fast | balanced | best
    log_callback: Optional[Callable] = None,
) -> dict:
    """
    Run Part A once, then Part B for every selected file.

    Args:
        srs_path: Path to SRS PDF/DOCX (may be None if only README mode).
        target_files: List of file dicts. Each item must have at least
            { "abs_path": str, "path": str }; optional "import_path".
            (Same shape as Part B's discover endpoint.)
        readme_content: Raw README text for Part A README extractor (optional).
        user_input: Dict with description/problems/expected/edge_cases for scenarios.
        deep_scan: Enable deep repo scan in Part B.
        max_retries: Part B max correction iterations.
        plan_mode: Part B plan-then-generate mode.
        top_k_requirements: How many requirements to seed RAG with per file (hybrid match).
        log_callback: Optional callback(event_type, *args) for SSE streaming.

    Returns:
        {
          "part_a":  { "requirements": [...], "scenarios": [...], "stats": {...} },
          "part_b":  [ { "file": "...", "success": bool, "test_code": "...", ... }, ... ],
          "summary": { "total_files": N, "succeeded": M, ... },
        }
    """
    from model_lifecycle import part_a_model, part_b_model
    from requirement_matcher import match_requirements_to_file

    sys.path.insert(0, str(_PART_A_SRS))
    sys.path.insert(0, str(_PART_A_README))
    sys.path.insert(0, str(_PART_B_TESTGEN))

    if not target_files:
        raise ValueError("No target files provided — pass at least one.")

    a_result: dict = {"requirements": [], "stats": {}, "scenarios": [], "features": {}}

    # ── Phase 1: Part A (runs once for the whole project) ────────────────
    _log("=== Phase 1: Part A — Requirement Extraction ===", log_callback)

    # 1a. SRS pipeline (no GPU needed)
    if srs_path:
        _log(f"  SRS extraction: {Path(srs_path).name}", log_callback)
        from srs_api import execute_part_a_srs  # type: ignore[import]
        srs_result = execute_part_a_srs(srs_path)
        a_result.update(srs_result)
        _log(
            f"  SRS done — {srs_result['stats'].get('requirements', 0)} requirements found",
            log_callback,
        )

    # 1b. README extractor — runs in a SUBPROCESS so VRAM is fully freed
    #     before Part B's model loads. BitsAndBytes 4-bit models on Windows
    #     don't release VRAM cleanly with del+empty_cache in the same process.
    if readme_content:
        _log("  README extraction (subprocess, ~2-3 min)...", log_callback)
        _log("    Extracting project features (LLM 1/2)...", log_callback)
        _log("    Generating test scenarios (LLM 2/2)...", log_callback)

        _extractor_script = _PART_A_README / "extractor_subprocess.py"
        _input_payload = json.dumps({
            "content":    readme_content,
            "repo_name":  "combined_pipeline",
            "user_input": user_input or {},
        }, ensure_ascii=False)

        try:
            proc = subprocess.run(
                [sys.executable, str(_extractor_script)],
                input=_input_payload,
                capture_output=True,
                text=True,
                timeout=600,
                env={**__import__("os").environ,
                     "PYTHONIOENCODING": "utf-8",
                     "PYTHONUTF8": "1"},
            )
            if proc.returncode != 0:
                _log(f"  README extraction failed (exit {proc.returncode}): "
                     f"{proc.stderr[-500:]}", log_callback, "warning")
            else:
                readme_result = json.loads(proc.stdout)
                a_result["scenarios"] = readme_result.get("test_scenarios", [])
                a_result["features"]  = readme_result.get("features", {})
                _log(
                    f"  README done — {len(a_result['scenarios'])} scenarios generated",
                    log_callback,
                )
        except subprocess.TimeoutExpired:
            _log("  README extraction timed out (>600s) — skipping", log_callback, "warning")
        except Exception as exc:
            _log(f"  README extraction error: {exc}", log_callback, "warning")

    # GPU is now FREE — subprocess exited, OS reclaimed all VRAM

    # ── Emit Part A results NOW (don't wait for Part B to finish) ─────────
    if log_callback:
        log_callback("result", {
            "part_a": a_result,
            "part_b": [],          # empty — will fill in per-file
            "summary": {},
            "phase": "part_a_complete",
        })

    # ── Phase 2 prep: filter out non-testable files (empty, __init__, etc.) ─
    try:
        from testable_filter import filter_testable  # type: ignore[import]
        keep, skipped = filter_testable(target_files)
        if skipped:
            _log(
                f"  Skipping {len(skipped)} non-testable file(s) "
                f"(empty / no public API / test files)",
                log_callback,
            )
            for sk in skipped:
                _log(f"     - {sk.get('path', '?')}: {sk['skip_reason']}", log_callback)
            target_files = keep
    except Exception as _fe:
        logger.debug("Testable filter failed (non-critical): %s", _fe)

    if not target_files:
        _log("No testable files remain after filtering — exiting Part B.", log_callback, "warning")
        return {
            "part_a":       a_result,
            "part_b":       [],
            "summary":      {"total_files": 0, "succeeded": 0, "failed": 0,
                             "requirements_total": len(a_result["requirements"]),
                             "scenarios_total": len(a_result["scenarios"]),
                             "quality_mode": quality_mode},
            "srs_coverage": {"total": 0, "covered": 0, "gaps": [], "coverage_pct": 0.0},
        }

    # ── Phase 2: Part B (per-file, hybrid-matched requirements) ──────────
    total = len(target_files)
    _log(f"=== Phase 2: Part B — Test Generation ({total} files) ===", log_callback)
    _log(
        f"  Project-level context: {len(a_result['requirements'])} requirements"
        f" + {len(a_result['scenarios'])} scenarios",
        log_callback,
    )

    # Auto-priming: scan repo for existing passing tests to use as style examples
    priming_examples = ""
    if target_files:
        repo_dir = str(Path(target_files[0]["abs_path"]).parent.parent)
        try:
            sys.path.insert(0, str(_PART_B_TESTGEN))
            from existing_test_scanner import get_priming_examples  # type: ignore[import]
            priming_examples = get_priming_examples(repo_dir, total_examples=3)
            if priming_examples:
                _log(f"  Auto-priming: found existing test examples to use as style reference", log_callback)
        except Exception as _pe:
            logger.debug("Auto-priming failed (non-critical): %s", _pe)

    per_file_results: list[dict] = []

    with part_b_model(use_lora=not use_base_only) as (model, tokenizer):
        from testgen_api import execute_part_b  # type: ignore[import]

        for i, file_info in enumerate(target_files, 1):
            abs_path     = file_info["abs_path"]
            rel_path     = file_info.get("path", Path(abs_path).name)
            import_path  = file_info.get("import_path")

            _log(f"\n  [{i}/{total}] {rel_path}", log_callback)
            if log_callback:
                log_callback("progress", i, total, rel_path)

            # Hybrid match: pick the most relevant requirements for THIS file
            try:
                with open(abs_path, encoding="utf-8", errors="ignore") as f:
                    file_source = f.read()
            except OSError as exc:
                _log(f"     ❌ Could not read file: {exc}", log_callback, "error")
                per_file_results.append({"file": rel_path, "success": False, "error": str(exc)})
                continue

            matched_reqs = match_requirements_to_file(
                file_source=file_source,
                requirements=a_result.get("requirements", []),
                top_k=top_k_requirements,
            )
            _log(
                f"     Matched {len(matched_reqs)} relevant requirements"
                f" (of {len(a_result['requirements'])} project-wide)",
                log_callback,
            )

            try:
                b_result = execute_part_b(
                    target_file       = abs_path,
                    requirements      = a_result.get("requirements", []),
                    import_path       = import_path,
                    deep_scan         = deep_scan,
                    max_retries       = max_retries,
                    log_callback      = log_callback,
                    plan_mode         = True,
                    use_base_only     = use_base_only,
                    priming_examples  = priming_examples,
                    model             = model,
                    tokenizer         = tokenizer,
                )
                b_result["file"] = rel_path
                b_result["matched_requirements_count"] = len(matched_reqs)
                per_file_results.append(b_result)
                status_str = "success" if b_result.get("success") else "finished with warnings"
                _log(f"     ✓ {status_str}", log_callback)
                # Stream this file's result immediately to the frontend
                if log_callback:
                    log_callback("file_result", {**b_result, "file": rel_path})
            except Exception as exc:
                logger.exception("Part B generation failed for %s", rel_path)
                _log(f"     ❌ {exc}", log_callback, "error")
                failed_entry = {"file": rel_path, "success": False, "error": str(exc)[:300]}
                per_file_results.append(failed_entry)
                if log_callback:
                    log_callback("file_result", failed_entry)

    # ── Summary ───────────────────────────────────────────────────────────
    succeeded = sum(1 for r in per_file_results if r.get("success"))
    _log(f"=== Phase 2 complete — {succeeded}/{total} files succeeded ===", log_callback)

    # ── Phase 3 (balanced / best): SRS coverage gap analysis ─────────────
    srs_coverage = {"total": 0, "covered": 0, "gaps": [], "coverage_pct": 0.0}
    if quality_mode in ("balanced", "best") and a_result.get("requirements"):
        _log("=== Phase 3: SRS Coverage Gap Analysis ===", log_callback)
        from coverage_analyzer import compute_srs_coverage, load_generated_tests  # type: ignore[import]
        test_contents = load_generated_tests(per_file_results)
        srs_coverage  = compute_srs_coverage(a_result["requirements"], test_contents)
        _log(
            f"  Coverage: {srs_coverage['covered']}/{srs_coverage['total']} requirements"
            f" ({srs_coverage['coverage_pct']}%), {len(srs_coverage['gaps'])} gaps found",
            log_callback,
        )
        # Attach srs_coverage to each file's result (combined total)
        for f in per_file_results:
            f["srs_coverage"] = {
                "total":   srs_coverage["total"],
                "covered": srs_coverage["covered"],
            }

    # ── Phase 4 (best only): gap-fill targeted generation ────────────────
    if quality_mode == "best" and srs_coverage.get("gaps"):
        _log(f"=== Phase 4: Gap-Fill Pass ({len(srs_coverage['gaps'])} uncovered reqs) ===",
             log_callback)
        gap_results = _run_gap_fill(
            gaps=srs_coverage["gaps"],
            target_files=target_files,
            use_base_only=use_base_only,
            log_callback=log_callback,
        )
        if gap_results:
            _log(f"  Gap-fill: {len(gap_results)} supplemental tests generated", log_callback)
            for gr in gap_results:
                per_file_results.append(gr)
                if log_callback:
                    log_callback("file_result", gr)

    summary = {
        "total_files": total,
        "succeeded":   succeeded,
        "failed":      total - succeeded,
        "requirements_total":   len(a_result["requirements"]),
        "scenarios_total":      len(a_result["scenarios"]),
        "srs_covered":          srs_coverage.get("covered", 0),
        "srs_coverage_pct":     srs_coverage.get("coverage_pct", 0.0),
        "quality_mode":         quality_mode,
    }

    return {
        "part_a":       a_result,
        "part_b":       per_file_results,
        "summary":      summary,
        "srs_coverage": srs_coverage,
    }


def _build_gap_fill_hint(gaps: list[dict], file_source: str) -> str:
    """
    Build a focused priming block for gap-fill generation.
    Forces the model to target specific uncovered requirements
    and produce concrete expected values.
    """
    import re as _re
    if not gaps:
        return ""
    # Extract function names from source to ground the model
    func_names = _re.findall(r"def ([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", file_source)
    func_names = [f for f in func_names if not f.startswith("_")][:8]

    lines = ["\n🎯 GAP-FILL MODE — these requirements have NO tests yet:"]
    for i, g in enumerate(gaps[:5], 1):
        text = g.get("text", "").strip()
        text = _re.sub(r"^REQ-[\d.]+\s*", "", text)[:100]
        lines.append(f"  {i}. {text}")

    lines.append("\nRULES for gap-fill tests:")
    lines.append("  • Write EXACTLY ONE pytest function per uncovered requirement above")
    lines.append("  • Use CONCRETE expected values — never placeholders like 'expected_result'")
    lines.append("  • If a function returns a specific value, assert THAT EXACT value")
    if func_names:
        lines.append(f"  • Available functions to test: {', '.join(func_names)}")
    lines.append("  • NEVER write: assert True, assert x == x, assert result == result")
    lines.append("")

    return "\n".join(lines)


def _run_gap_fill(
    gaps: list[dict],
    target_files: list[dict],
    use_base_only: bool,
    log_callback: Optional[Callable],
) -> list[dict]:
    """
    For each uncovered SRS requirement, generate one targeted test using
    the most relevant target file as context. Keeps only tests that pass.
    """
    if not gaps or not target_files:
        return []

    _log(f"  Loading model for gap-fill ({len(gaps)} requirements)...", log_callback)

    gap_results = []
    try:
        with part_b_model(use_lora=not use_base_only) as (model, tokenizer):
            from testgen_api import execute_part_b  # type: ignore[import]
            from requirement_matcher import match_requirements_to_file  # type: ignore[import]

            # Group gaps by best matching target file
            for file_info in target_files:
                abs_path = file_info["abs_path"]
                rel_path = file_info.get("path", Path(abs_path).name)

                try:
                    with open(abs_path, encoding="utf-8", errors="ignore") as f:
                        file_source = f.read()
                except OSError:
                    continue

                # Which gaps are relevant to this file?
                file_gaps = [
                    g for g in gaps
                    if match_requirements_to_file(file_source, [g], top_k=1)
                ]
                if not file_gaps:
                    continue

                _log(f"  Gap-fill: {len(file_gaps)} reqs → {rel_path}", log_callback)

                # Build a focused gap-fill priming hint with concrete-value forcing
                gap_hint = _build_gap_fill_hint(file_gaps, file_source)

                result = execute_part_b(
                    target_file      = abs_path,
                    requirements     = file_gaps,
                    max_retries      = 2,
                    log_callback     = log_callback,
                    plan_mode        = True,
                    use_base_only    = use_base_only,
                    priming_examples = gap_hint,
                    model            = model,
                    tokenizer        = tokenizer,
                )
                result["file"]   = f"gap_fill_{rel_path}"
                result["is_gap_fill"] = True
                result["gap_count"]   = len(file_gaps)
                gap_results.append(result)

    except Exception as exc:
        _log(f"  Gap-fill error: {exc}", log_callback, "warning")

    return gap_results
