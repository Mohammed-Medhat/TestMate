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
) -> dict:
    """
    Run the self-correcting test generation loop for a single target file.

    Args:
        target_file:    Absolute path to the Python file to test.
        requirements:   Part A's labeled requirements (all of them for this project).
                        They are passed into autonomous_loop's plan prompt directly —
                        top-12 most relevant will be injected at plan level.
        import_path:    Package import path (e.g. 'requests.auth').
        deep_scan:      Scan broader repo context.
        max_retries:    Max self-correction iterations.
        log_callback:   GUI callback(event_type, *args).
        plan_mode:      Generate a test plan before writing code (recommended for SRS).
        use_base_only:  Skip LoRA — raw 4-bit base Qwen.
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

    # Filter to just requirement-labeled sentences for logging
    req_count = sum(1 for r in (requirements or []) if isinstance(r, dict) and r.get("label") == 1)
    logger.info("[B] %s: plan_mode=%s, %d/%d SRS requirements available",
                Path(target_file).name, plan_mode, req_count, len(requirements or []))

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
        )

        # Locate generated test file
        basename = Path(target_file).stem
        test_filename = f"test_{basename}.py"
        candidates = [
            GEN_TESTS_DIR / test_filename,
            Path(target_file).parent / test_filename,
        ]
        test_path = next((p for p in candidates if p.exists()), None)
        test_code = test_path.read_text(encoding="utf-8") if test_path else ""

        # Requirement coverage check
        coverage = check_requirement_coverage(requirements or [], test_code)
        logger.info("[B] SRS coverage: %d/%d (%.0f%%)",
                    coverage["covered"], coverage["total"], coverage["pct"])

        return {
            "success":    success,
            "test_file":  test_filename,
            "test_code":  test_code,
            "target":     target_file,
            "srs_coverage": coverage,
        }

    finally:
        if _loaded_here:
            import gc, torch
            del _model, _tokenizer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
