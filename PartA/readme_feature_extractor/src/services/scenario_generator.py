# src/scenario_generator/scenario_generator.py
from __future__ import annotations

import json
import logging
import re
from typing import Any

from .llm_engine import call_llm
from ..models.models import ProjectFeatures, TestScenario, UserIntent

logger = logging.getLogger(__name__)

_TARGET_SCENARIO_COUNT = 10
_MIN_MIX: dict[str, int] = {"positive": 2, "negative": 2, "edge": 2, "boundary": 1}

_SCENARIO_SYSTEM = (
    "You are a senior QA engineer. "
    "Generate descriptive test SCENARIOS only — no code, no assertions. "
    "Return a JSON array only. No markdown fences, no prose outside the JSON."
)

_SCENARIO_PROMPT = """\
Generate {count} test scenarios for the following software project.

PROJECT FACTS:
- Name:              {name}
- Description:       {description}
- Tech Stack:        {tech_stack}
- Has Tests:         {has_tests}
- License:           {license_type}

USER INTENT:
- Main Problem:      {main_problem}
- Expected Behavior: {expected_behavior}
- Edge Cases noted:  {edge_cases}

RULES:
- Include a mix of: positive, negative, edge, boundary
- Do NOT generate code or test functions
- Each scenario must be standalone and self-explanatory
- Priority: "high" for core flows, "medium" for secondary, "low" for optional

Return JSON array:
[
  {{
    "name":            "...",
    "type":            "positive|negative|edge|boundary",
    "description":     "...",
    "expected_result": "...",
    "priority":        "high|medium|low"
  }}
]
"""


def _build_fallback_scenarios(features: ProjectFeatures, user_intent: UserIntent) -> list[TestScenario]:
    name     = features.project_name or "the application"
    desc     = features.description  or "the system"
    expected = user_intent.expected_behavior or f"{name} works as documented"
    problem  = user_intent.main_problem      or "the main functionality"

    scenarios: list[TestScenario] = [
        TestScenario(name="Happy Path — Core Functionality", type="positive",
            description=f"Verify that {desc} performs its primary use case with valid inputs.",
            expected_result=f"System completes the operation successfully. {expected}", priority="high"),
        TestScenario(name="Invalid Input Rejection", type="negative",
            description=f"Submit malformed or out-of-range input to {name}.",
            expected_result="System returns a clear error without exposing internals.", priority="high"),
        TestScenario(name="Empty / Null Input Handling", type="edge",
            description=f"Provide null or empty values to {name}'s primary inputs.",
            expected_result="System rejects or handles gracefully without crashing.", priority="medium"),
        TestScenario(name="Boundary Value — Maximum Input Size", type="boundary",
            description=f"Submit the largest allowable input to {name}.",
            expected_result="System processes without memory errors or truncation.", priority="medium"),
        TestScenario(name="Concurrent Requests", type="edge",
            description=f"Send multiple simultaneous requests to {name}.",
            expected_result="Each request is handled independently; no data corruption.", priority="medium"),
    ]

    if user_intent.main_problem:
        scenarios.append(TestScenario(
            name=f"Reproduce Reported Problem: {problem[:50]}", type="negative",
            description=f"Replicate the conditions that trigger: {problem}.",
            expected_result="The previously reported problem no longer occurs.", priority="high"))

    tech_lower = [t.lower() for t in features.tech_stack]
    if any(db in tech_lower for db in ["postgresql", "mysql", "mongodb"]):
        scenarios.append(TestScenario(name="Database Connection Failure", type="negative",
            description="Simulate a DB connection failure and verify graceful handling.",
            expected_result="User-friendly error returned; no connection strings exposed.", priority="high"))

    if "docker" in tech_lower:
        scenarios.append(TestScenario(name="Container Startup Verification", type="positive",
            description="Start from a clean image; verify all services pass health checks.",
            expected_result="All containers start; health endpoints return 200 OK.", priority="high"))

    if features.has_tests:
        scenarios.append(TestScenario(name="Existing Test Suite Passes", type="positive",
            description="Run the existing test suite against the latest codebase.",
            expected_result="All tests pass; coverage does not regress.", priority="high"))

    return scenarios


def _parse_llm_scenarios(raw: str) -> list[TestScenario]:
    """
    FIX #8: raw is now always a real string from call_llm (response_model=None).
    """
    clean = re.sub(r"```json\s*", "", raw)
    clean = re.sub(r"```\s*", "", clean).strip()

    match = re.search(r"\[.*\]", clean, re.DOTALL)
    if not match:
        logger.warning("scenario_generator: no JSON array found in LLM output")
        return []

    try:
        items = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        logger.warning("scenario_generator: JSON decode error: %s", exc)
        return []

    scenarios: list[TestScenario] = []
    for item in items:
        try:
            scenarios.append(TestScenario(**item))
        except Exception as ve:
            logger.warning("scenario_generator: skipping invalid item: %s | %s", ve, item)
    return scenarios


def _deduplicate(scenarios: list[TestScenario]) -> list[TestScenario]:
    seen: set[str] = set()
    result: list[TestScenario] = []
    for s in scenarios:
        key = s.name.lower().strip()
        if key not in seen:
            seen.add(key)
            result.append(s)
    return result


def _ensure_mix(scenarios, features, user_intent):
    from collections import Counter
    counts = Counter(s.type for s in scenarios)
    fallbacks = _build_fallback_scenarios(features, user_intent)
    by_type: dict[str, list[TestScenario]] = {"positive": [], "negative": [], "edge": [], "boundary": []}
    for s in fallbacks:
        by_type[s.type].append(s)
    for stype, minimum in _MIN_MIX.items():
        deficit = minimum - counts.get(stype, 0)
        if deficit > 0:
            scenarios.extend(by_type.get(stype, [])[:deficit])
    return scenarios


def generate_scenarios(features: ProjectFeatures, user_intent: UserIntent, model: Any) -> list[TestScenario]:
    prompt = _SCENARIO_PROMPT.format(
        count=_TARGET_SCENARIO_COUNT,
        name=features.project_name or "Unknown Project",
        description=features.description or "No description available",
        tech_stack=", ".join(features.tech_stack) if features.tech_stack else "Unknown",
        has_tests="Yes" if features.has_tests else "No",
        license_type=features.license_type or "Not specified",
        main_problem=user_intent.main_problem or "Not specified",
        expected_behavior=user_intent.expected_behavior or "Not specified",
        edge_cases=user_intent.edge_cases or "Not specified",
    )

    llm_scenarios: list[TestScenario] = []

    if model is not None:
        # FIX #8: response_model=None → call_llm returns raw string → parse here
        raw_result = call_llm(
            prompt=prompt,
            model=model,
            response_model=None,
            system_prompt=_SCENARIO_SYSTEM,
            max_tokens=2_000,
            temperature=0.3,
        )
        # raw_result is now guaranteed to be a string
        raw_str = raw_result if isinstance(raw_result, str) else json.dumps(raw_result)
        llm_scenarios = _parse_llm_scenarios(raw_str)

    if not llm_scenarios:
        logger.warning("scenario_generator: LLM produced no valid scenarios — using fallback")
        llm_scenarios = _build_fallback_scenarios(features, user_intent)

    scenarios = _deduplicate(llm_scenarios)
    scenarios = _ensure_mix(scenarios, features, user_intent)
    scenarios = _deduplicate(scenarios)

    logger.debug("scenario_generator: final count = %d", len(scenarios))
    return scenarios