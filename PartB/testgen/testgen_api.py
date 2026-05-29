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
            auto_repair=auto_repair,
            docker_image=docker_image,
        )

        # Locate generated test file.
        # Prefer the _testmate.py naming (PartB's new convention) then fall
        # back to the legacy test_<stem>.py name for backward compatibility.
        basename = Path(target_file).stem
        candidates = [
            GEN_TESTS_DIR / f"test_{basename}_testmate.py",  # new
            Path(target_file).parent / f"test_{basename}_testmate.py",
            GEN_TESTS_DIR / f"test_{basename}.py",           # legacy
            Path(target_file).parent / f"test_{basename}.py",
        ]
        test_path = next((p for p in candidates if p.exists()), None)
        test_filename = test_path.name if test_path else f"test_{basename}_testmate.py"
        test_code = test_path.read_text(encoding="utf-8") if test_path else ""

        # Read bug reports produced during this run
        bug_report_path = GEN_TESTS_DIR / "bug_reports.jsonl"
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

        return {
            "success":      success,
            "test_file":    test_filename,
            "test_code":    test_code,
            "test_path":    str(test_path) if test_path else "",
            "target":       target_file,
            "srs_coverage": coverage,
            "bug_reports":  bug_reports,
            "repairs":      [b.get("repair", {}) for b in bug_reports if b.get("repair")],
        }

    finally:
        if _loaded_here:
            import gc, torch
            del _model, _tokenizer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
