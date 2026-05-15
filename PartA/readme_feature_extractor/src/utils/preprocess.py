# src/scenario_generator/preprocess.py
"""
Preprocessing layer for the Test Scenario Generator.

Responsibilities:
  - extract_important_sections  → distil cleaned README into a single ranked text block
  - build_user_context          → normalise & validate the user_input dict
  - combine_inputs              → merge README text + user context into a prompt-ready string
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Section priorities: higher = more useful for scenario generation
# ---------------------------------------------------------------------------
_SECTION_PRIORITY = [
    "usage",        # how the project is used  → most valuable
    "testing",      # test infrastructure hints
    "installation", # setup steps
    "tech",         # technology section
    "license",      # usually low-signal, but keep it
]


def extract_important_sections(cleaned_readme: dict[str, Any]) -> str:
    """
    Distil a cleaned README dict (output of cleaner.clean_readme) into a
    single, LLM-friendly text block.

    Priority order: usage → testing → installation → tech → license → prose.
    Code blocks are appended last (they're noisy but useful for tech detection).

    Args:
        cleaned_readme: dict with keys 'prose', 'code_blocks', 'sections' (optional).

    Returns:
        A single string ready to be embedded in a prompt.
    """
    parts: list[str] = []

    # ── 1. Named sections (highest signal) ──────────────────────────────────
    sections: dict[str, str] = cleaned_readme.get("sections", {})
    for key in _SECTION_PRIORITY:
        value = sections.get(key, "").strip()
        if value:
            parts.append(f"[{key.upper()}]\n{value}")
            logger.debug("preprocess: added section '%s' (%d chars)", key, len(value))

    # ── 2. Prose fallback ────────────────────────────────────────────────────
    prose: str = cleaned_readme.get("prose", "").strip()
    if prose:
        # cap prose to 5 000 chars — anything longer is usually boilerplate
        capped = prose[:5_000]
        parts.append(f"[README_PROSE]\n{capped}")
        logger.debug("preprocess: added prose (%d chars → capped to %d)", len(prose), len(capped))

    # ── 3. Code blocks (top-3 only — noisy beyond that) ────────────────────
    code_blocks: list[str] = cleaned_readme.get("code_blocks", [])
    if code_blocks:
        snippet = "\n---\n".join(block.strip() for block in code_blocks[:3])
        parts.append(f"[CODE_SNIPPETS]\n{snippet[:1_500]}")
        logger.debug("preprocess: added %d code block(s)", min(len(code_blocks), 3))

    result = "\n\n".join(parts)
    logger.debug("preprocess: extract_important_sections → %d total chars", len(result))
    return result


def build_user_context(user_input: dict[str, Any]) -> str:
    """
    Normalise the user_input dict into a structured text block.

    Handles missing keys gracefully — any absent field is silently skipped.

    Args:
        user_input: dict with optional keys:
            'description', 'problems', 'expected', 'edge_cases'

    Returns:
        A structured string block, or empty string if all fields are blank.
    """
    mapping = {
        "DESCRIPTION":    user_input.get("description", ""),
        "MAIN_PROBLEM":   user_input.get("problems", ""),
        "EXPECTED_BEHAVIOR": user_input.get("expected", ""),
        "EDGE_CASES":     user_input.get("edge_cases", ""),
    }

    parts: list[str] = []
    for label, value in mapping.items():
        stripped = (value or "").strip()
        if stripped:
            parts.append(f"[USER_{label}]\n{stripped}")
            logger.debug("preprocess: user context field '%s' → %d chars", label, len(stripped))

    result = "\n\n".join(parts)
    logger.debug("preprocess: build_user_context → %d total chars", len(result))
    return result


def combine_inputs(readme_text: str, user_context: str) -> str:
    """
    Merge README text and user context into a single string for the LLM.

    The README always comes first so that project facts ground the prompt.
    User context follows so the model focuses on the user's specific concerns.

    Args:
        readme_text:  Output of extract_important_sections().
        user_context: Output of build_user_context().

    Returns:
        Combined string, with a clear separator between the two blocks.
    """
    separator = "\n" + ("─" * 60) + "\n"

    blocks: list[str] = []
    if readme_text.strip():
        blocks.append(readme_text.strip())
    if user_context.strip():
        blocks.append(user_context.strip())

    combined = separator.join(blocks) if blocks else ""
    logger.debug("preprocess: combine_inputs → %d total chars", len(combined))
    return combined