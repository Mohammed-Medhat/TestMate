# src/scenario_generator/confidence.py
"""
Dynamic confidence scoring for extracted project features.

Each field is scored independently using a set of signals:
  - Presence / non-emptiness
  - Whether the value appears verbatim in the source text
  - Length / richness of the value
  - Whether it was found by a deterministic heuristic (bonus)

Public API:
  compute_confidence(features, full_text, heuristic_hits) → ConfidenceScores
  get_low_confidence_fields(scores, threshold)            → List[str]
"""

from __future__ import annotations

import logging
import math
from typing import Any, Optional

from ..models.models import ConfidenceScores, ProjectFeatures

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def _value_in_text(value: Any, text: str) -> bool:
    """Check whether a scalar value appears (case-insensitively) in text."""
    if not value or not text:
        return False
    return str(value).lower() in text.lower()


def _score_string_field(
    value: Optional[str],
    full_text: str,
    *,
    heuristic_hit: bool = False,
    ideal_length: int = 40,
) -> float:
    """
    Score a single optional string field.

    Signals used:
      - Missing → 0.20
      - Very short (< 3 chars) → 0.35
      - Appears in source text → +0.25
      - Heuristic also found it → +0.15
      - Length bonus (log-scaled, caps at ideal_length) → up to +0.20

    Result is clamped to [0.20, 0.97].
    """
    if not value:
        return 0.20

    base = 0.45

    # Presence in source
    if _value_in_text(value, full_text):
        base += 0.25

    # Heuristic corroboration
    if heuristic_hit:
        base += 0.15

    # Length bonus (information richness)
    length_ratio = min(len(value) / ideal_length, 1.0)
    base += 0.15 * math.log1p(length_ratio * (math.e - 1))

    return round(min(max(base, 0.20), 0.97), 3)


def _score_list_field(
    value: list,
    full_text: str,
    *,
    heuristic_hit: bool = False,
    ideal_count: int = 4,
) -> float:
    """
    Score a list field.

    Signals:
      - Empty → 0.20
      - Count bonus (log-scaled, caps at ideal_count) → up to 0.40
      - At least one item appears in source → +0.20
      - Heuristic also found items → +0.15

    Result clamped to [0.20, 0.97].
    """
    if not value:
        return 0.20

    base = 0.40

    # Count richness
    count_ratio = min(len(value) / ideal_count, 1.0)
    base += 0.30 * math.log1p(count_ratio * (math.e - 1))

    # Source verification (any item found in text?)
    if any(_value_in_text(item, full_text) for item in value):
        base += 0.20

    if heuristic_hit:
        base += 0.15

    return round(min(max(base, 0.20), 0.97), 3)


def _score_bool_field(value: bool, *, heuristic_hit: bool = False) -> float:
    """
    Score a boolean field.

    True + heuristic corroboration → high confidence.
    True without corroboration     → moderate.
    False (not detected)           → lower but not rock-bottom.
    """
    if value and heuristic_hit:
        return 0.92
    if value:
        return 0.72
    # False can still be correct — we're moderately confident it's absent
    return 0.60


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_confidence(
    features: ProjectFeatures,
    full_text: str,
    heuristic_hits: Optional[dict[str, bool]] = None,
) -> ConfidenceScores:
    """
    Compute per-field confidence scores for a ProjectFeatures instance.

    Args:
        features:       Extracted features to score.
        full_text:      The combined README + user-input text used for extraction.
        heuristic_hits: Dict indicating which fields were also found by
                        deterministic heuristics.  Keys:
                          'tech_stack', 'has_tests', 'license_type'
                        Missing keys are treated as False.

    Returns:
        A ConfidenceScores instance.
    """
    hits = heuristic_hits or {}

    scores = ConfidenceScores(
        project_name=_score_string_field(
            features.project_name,
            full_text,
            ideal_length=30,
        ),
        description=_score_string_field(
            features.description,
            full_text,
            ideal_length=120,
        ),
        tech_stack=_score_list_field(
            features.tech_stack,
            full_text,
            heuristic_hit=bool(hits.get("tech_stack")),
            ideal_count=4,
        ),
        installation_commands=_score_list_field(
            features.installation_commands,
            full_text,
            ideal_count=3,
        ),
        has_tests=_score_bool_field(
            features.has_tests,
            heuristic_hit=bool(hits.get("has_tests")),
        ),
        license_type=_score_string_field(
            features.license_type,
            full_text,
            heuristic_hit=bool(hits.get("license_type")),
            ideal_length=15,
        ),
    )

    logger.debug("confidence: scores = %s", scores.model_dump())
    return scores


def get_low_confidence_fields(
    scores: ConfidenceScores,
    threshold: float = 0.50,
) -> list[str]:
    """
    Return the names of fields whose confidence is below `threshold`.

    Args:
        scores:    ConfidenceScores instance.
        threshold: Fields with score < threshold are flagged (default 0.50).

    Returns:
        Sorted list of field names.
    """
    low = [
        field
        for field, score in scores.model_dump().items()
        if score < threshold
    ]
    if low:
        logger.debug("confidence: low-confidence fields: %s", low)
    return sorted(low)