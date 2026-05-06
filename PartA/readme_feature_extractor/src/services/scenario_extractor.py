# src/scenario_generator/feature_extractor.py
"""
Feature extractor for the Test Scenario Generator.

Combines fast deterministic heuristics with an LLM call to produce a
validated ProjectFeatures instance.

Strategy:
  1. Run heuristics (instant, no LLM) → tech_stack, has_tests, license_type
  2. Ask LLM for high-level fields     → project_name, description, installation_commands
  3. Merge both results safely         → prefer heuristics for detection fields
  4. Validate final object via Pydantic

Public API:
  extract_project_features(combined_text, model) → ProjectFeatures
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from ..utils.heuristics import detect_license, detect_tests, extract_tech_stack
from .llm_engine import call_llm
from ..models.models import ProjectFeatures

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LLM prompt template
# ---------------------------------------------------------------------------

_EXTRACTION_SYSTEM = (
    "You are a strict software-project analyst. "
    "Extract factual information ONLY from the provided text. "
    "Do NOT invent or hallucinate missing data — use null or [] instead. "
    "Return valid JSON only, no markdown fences, no prose."
)

_EXTRACTION_PROMPT = """\
Analyse the following project text and extract these fields:

- project_name      : (string | null)  The software project name
- description       : (string | null)  One or two sentence summary of what the project does
- installation_commands : (list[string]) Terminal commands needed to install the project

Rules:
- project_name     → exact name only; null if genuinely unclear
- description      → ≤ 2 sentences, ≤ 300 chars; null if absent
- installation_commands → shell commands only (pip install …, npm install, etc.)
- Do NOT include field "tech_stack", "has_tests", or "license_type" — those are handled elsewhere

TEXT:
{text}

Return JSON like:
{{
  "project_name": "...",
  "description":  "...",
  "installation_commands": ["...", "..."]
}}
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_project_features(
    combined_text: str,
    model: Any,
) -> ProjectFeatures:
    """
    Extract structured project features from the combined README + user-input text.

    Args:
        combined_text: Output of preprocess.combine_inputs().
        model:         A llama-cpp-python Llama instance (or None for heuristics-only).

    Returns:
        A validated ProjectFeatures instance.
    """
    logger.debug("feature_extractor: starting extraction (%d chars)", len(combined_text))

    # ── Step 1: Fast deterministic heuristics ───────────────────────────────
    tech_heuristic   = extract_tech_stack(combined_text)
    tests_heuristic  = detect_tests(combined_text)
    license_heuristic = detect_license(combined_text)

    logger.debug(
        "feature_extractor: heuristics → tech=%s, tests=%s, license=%s",
        tech_heuristic, tests_heuristic, license_heuristic,
    )

    # ── Step 2: LLM extraction for complex fields ────────────────────────────
    prompt = _EXTRACTION_PROMPT.format(text=combined_text[:6_000])  # cap to keep tokens low

    llm_result: ProjectFeatures = call_llm(
        prompt=prompt,
        model=model,
        response_model=ProjectFeatures,
        system_prompt=_EXTRACTION_SYSTEM,
        max_tokens=800,
        temperature=0.15,   # low temperature = less hallucination
    )

    logger.debug(
        "feature_extractor: LLM → name=%r, description=%r, commands=%s",
        llm_result.project_name,
        (llm_result.description or "")[:60],
        llm_result.installation_commands,
    )

    # ── Step 3: Merge — heuristics win for detection fields ─────────────────
    # tech_stack: union of both (heuristics are reliable, LLM may add extra)
    merged_tech = list(dict.fromkeys(llm_result.tech_stack + tech_heuristic))

    # has_tests: OR — if either source says True, trust it
    merged_tests = tests_heuristic or llm_result.has_tests

    # license: heuristic takes priority (regex is more precise than the LLM here)
    merged_license = license_heuristic or llm_result.license_type

    # ── Step 4: Build final validated model ─────────────────────────────────
    features = ProjectFeatures(
        project_name=_sanitise_str(llm_result.project_name),
        description=_sanitise_str(llm_result.description),
        tech_stack=merged_tech,
        installation_commands=llm_result.installation_commands,
        has_tests=merged_tests,
        license_type=_sanitise_str(merged_license),
    )

    logger.debug("feature_extractor: final features = %s", features.model_dump())
    return features


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _sanitise_str(value: Optional[str]) -> Optional[str]:
    """Strip whitespace and return None for empty/placeholder strings."""
    if not value:
        return None
    cleaned = value.strip()
    # Reject obvious LLM placeholders
    placeholders = {"n/a", "null", "none", "unknown", "not found", "not specified", ""}
    if cleaned.lower() in placeholders:
        return None
    return cleaned


def get_heuristic_hits(combined_text: str) -> dict[str, bool]:
    """
    Return a dict indicating which fields were detected by heuristics alone.
    Used downstream by the confidence scorer.

    Args:
        combined_text: The full combined project text.

    Returns:
        e.g. {"tech_stack": True, "has_tests": False, "license_type": True}
    """
    return {
        "tech_stack":    bool(extract_tech_stack(combined_text)),
        "has_tests":     detect_tests(combined_text),
        "license_type":  detect_license(combined_text) is not None,
    }