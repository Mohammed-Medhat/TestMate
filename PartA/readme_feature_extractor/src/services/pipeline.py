# src/services/pipeline.py
from __future__ import annotations

import logging
import time
from typing import Any

# 1. Corrected imports pointing to the new folders
from src.services.confidence import compute_confidence, get_low_confidence_fields
from src.services.scenario_extractor import extract_project_features, get_heuristic_hits
from src.models.models import FinalOutput, ProjectFeatures, UserIntent
from src.utils.preprocess import build_user_context, combine_inputs, extract_important_sections
from src.services.scenario_generator import generate_scenarios

logger = logging.getLogger(__name__)


def _build_user_intent(user_input: dict[str, Any]) -> UserIntent:
    return UserIntent(
        main_problem=(user_input.get("problems") or "").strip(),
        expected_behavior=(user_input.get("expected") or "").strip(),
        edge_cases=(user_input.get("edge_cases") or "").strip(),
    )


def run_pipeline(
    cleaned_readme: Any,
    user_input: dict[str, Any],
    model: Any,
) -> FinalOutput:
    t0 = time.perf_counter()
    logger.info("pipeline: starting run")

    # ── Stage 1: Preprocessing ──────────────────────────────────────────────
    if isinstance(cleaned_readme, dict):
        readme_text = extract_important_sections(cleaned_readme)
    else:
        readme_text = str(cleaned_readme) if cleaned_readme else ""

    user_context = build_user_context(user_input)
    combined     = combine_inputs(readme_text, user_context)

    # ── Stage 2: Feature extraction ─────────────────────────────────────────
    features: ProjectFeatures = extract_project_features(combined, model)

    # ── Stage 3: User intent ────────────────────────────────────────────────
    user_intent: UserIntent = _build_user_intent(user_input)

    # ── Stage 4: Confidence scoring ─────────────────────────────────────────
    heuristic_hits  = get_heuristic_hits(combined)
    confidence      = compute_confidence(features, combined, heuristic_hits)
    low_conf_fields = get_low_confidence_fields(confidence, threshold=0.50)

    # ── Stage 5: Scenario generation ────────────────────────────────────────
    scenarios = generate_scenarios(features, user_intent, model)

    # ── Stage 6: Assemble output ────────────────────────────────────────────
    output = FinalOutput(
        features=features,
        user_intent=user_intent,
        test_scenarios=scenarios,
        confidence=confidence,
        low_confidence_fields=low_conf_fields,
    )

    elapsed = time.perf_counter() - t0
    logger.info("pipeline: completed in %.2fs | scenarios=%d", elapsed, len(scenarios))
    return output