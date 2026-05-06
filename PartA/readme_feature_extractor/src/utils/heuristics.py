# src/scenario_generator/heuristics.py
"""
Fast, deterministic heuristics for extracting project metadata.

All functions are pure (no side effects, no LLM calls) and run in O(n)
time relative to the input text length.

Functions:
  - extract_tech_stack(text)  → List[str]
  - detect_tests(text)        → bool
  - detect_license(text)      → Optional[str]
"""

from __future__ import annotations

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tech-stack catalogue
# ---------------------------------------------------------------------------
# Each entry: (canonical_name, list_of_patterns_to_match_case_insensitively)
_TECH_CATALOGUE: list[tuple[str, list[str]]] = [
    # ── Languages ────────────────────────────────────────────────────────────
    ("Python",       ["python"]),
    ("JavaScript",   ["javascript", r"\bjs\b"]),
    ("TypeScript",   ["typescript", r"\bts\b"]),
    ("Go",           [r"\bgolang\b", r"\bgo\b"]),
    ("Rust",         [r"\brust\b"]),
    ("Java",         [r"\bjava\b"]),
    ("C#",           [r"\bc#\b", r"\bdotnet\b", r"\.net\b"]),
    ("C++",          [r"c\+\+"]),
    ("Ruby",         [r"\bruby\b"]),
    ("PHP",          [r"\bphp\b"]),
    # ── Web frameworks ───────────────────────────────────────────────────────
    ("Django",       ["django"]),
    ("Flask",        [r"\bflask\b"]),
    ("FastAPI",      ["fastapi"]),
    ("Express",      [r"\bexpress\b"]),
    ("Rails",        [r"\brails\b", "ruby on rails"]),
    ("Spring Boot",  ["spring boot", "springboot"]),
    ("Laravel",      ["laravel"]),
    # ── Frontend ─────────────────────────────────────────────────────────────
    ("React",        [r"\breact\b"]),
    ("Vue",          [r"\bvue\.?js\b", r"\bvuejs\b"]),
    ("Angular",      [r"\bangular\b"]),
    ("Next.js",      [r"next\.js", r"\bnextjs\b"]),
    ("Svelte",       [r"\bsvelte\b"]),
    ("Tailwind CSS", ["tailwind"]),
    # ── Runtime / backend ────────────────────────────────────────────────────
    ("Node.js",      [r"node\.js", r"\bnodejs\b", r"\bnode\b"]),
    ("Deno",         [r"\bdeno\b"]),
    # ── ML / data ────────────────────────────────────────────────────────────
    ("TensorFlow",   ["tensorflow"]),
    ("PyTorch",      ["pytorch", "torch"]),
    ("scikit-learn", ["scikit-learn", "sklearn"]),
    ("Pandas",       [r"\bpandas\b"]),
    ("NumPy",        [r"\bnumpy\b"]),
    ("Hugging Face", ["huggingface", "hugging face", "transformers"]),
    # ── Databases ────────────────────────────────────────────────────────────
    ("PostgreSQL",   ["postgresql", "postgres"]),
    ("MySQL",        [r"\bmysql\b"]),
    ("MongoDB",      ["mongodb", "mongo"]),
    ("Redis",        [r"\bredis\b"]),
    ("SQLite",       ["sqlite"]),
    ("Elasticsearch",["elasticsearch"]),
    # ── Infrastructure ───────────────────────────────────────────────────────
    ("Docker",       ["docker"]),
    ("Kubernetes",   ["kubernetes", r"\bk8s\b"]),
    ("Terraform",    ["terraform"]),
    ("AWS",          [r"\baws\b", "amazon web services"]),
    ("GCP",          [r"\bgcp\b", "google cloud"]),
    ("Azure",        [r"\bazure\b"]),
    # ── Testing ──────────────────────────────────────────────────────────────
    ("pytest",       ["pytest"]),
    ("Jest",         [r"\bjest\b"]),
    ("Cypress",      ["cypress"]),
    ("Playwright",   ["playwright"]),
    # ── Other tooling ────────────────────────────────────────────────────────
    ("GraphQL",      ["graphql"]),
    ("gRPC",         [r"\bgrpc\b"]),
    ("Celery",       [r"\bcelery\b"]),
    ("RabbitMQ",     ["rabbitmq"]),
    ("Kafka",        [r"\bkafka\b"]),
]

# Pre-compile patterns once at import time for speed
_COMPILED_TECH: list[tuple[str, list[re.Pattern]]] = [
    (name, [re.compile(p, re.IGNORECASE) for p in patterns])
    for name, patterns in _TECH_CATALOGUE
]


# ---------------------------------------------------------------------------
# License catalogue
# ---------------------------------------------------------------------------
_LICENSE_PATTERNS: list[tuple[str, list[str]]] = [
    ("MIT",              [r"\bmit\b"]),
    ("Apache 2.0",       [r"apache[\s\-]*2\.?0?", r"\bapache\b"]),
    ("GPL-3.0",          [r"gpl[\s\-]*3", r"gnu general public"]),
    ("GPL-2.0",          [r"gpl[\s\-]*2"]),
    ("LGPL",             [r"\blgpl\b"]),
    ("BSD-2-Clause",     [r"bsd[\s\-]*2"]),
    ("BSD-3-Clause",     [r"bsd[\s\-]*3", r"\bbsd\b"]),
    ("Mozilla 2.0",      [r"mozilla", r"\bmpl\b"]),
    ("ISC",              [r"\bisc\b"]),
    ("Creative Commons", [r"creative commons", r"\bcc[\s\-]by\b"]),
    ("Unlicense",        [r"\bunlicense\b"]),
    ("AGPL-3.0",         [r"\bagpl\b"]),
]

_COMPILED_LICENSE: list[tuple[str, list[re.Pattern]]] = [
    (name, [re.compile(p, re.IGNORECASE) for p in patterns])
    for name, patterns in _LICENSE_PATTERNS
]


# ---------------------------------------------------------------------------
# Test-infrastructure keywords
# ---------------------------------------------------------------------------
_TEST_KEYWORDS: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\bpytest\b",
        r"\bunittest\b",
        r"\bjest\b",
        r"\bcypress\b",
        r"\bplaywright\b",
        r"\bmocha\b",
        r"\bjasmine\b",
        r"\bvitest\b",
        r"\bnpm\s+test\b",
        r"\bnpm\s+run\s+test\b",
        r"tests/",
        r"\btest_",
        r"_test\.py\b",
        r"\.spec\.(ts|js)\b",
        r"\bcoverage\b",
        r"\btox\b",
        r"\bci\.yml\b",
        r"github/workflows",
        r"\bsetup\.cfg\b.*\[tool:pytest\]",
        r"\bpyproject\.toml\b.*pytest",
    ]
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_tech_stack(text: str) -> list[str]:
    """
    Return a deduplicated list of detected technology names.

    Uses pre-compiled regex patterns against the full text.
    Order is stable (matches catalogue order).

    Args:
        text: Any block of text (README prose + code blocks).

    Returns:
        e.g. ["Python", "FastAPI", "PostgreSQL", "Docker"]
    """
    found: list[str] = []
    for name, patterns in _COMPILED_TECH:
        if any(p.search(text) for p in patterns):
            found.append(name)

    logger.debug("heuristics: detected tech stack: %s", found)
    return found


def detect_tests(text: str) -> bool:
    """
    Return True if the text contains evidence of a test infrastructure.

    Args:
        text: Any block of text.

    Returns:
        True if at least one test-related keyword is found.
    """
    result = any(p.search(text) for p in _TEST_KEYWORDS)
    logger.debug("heuristics: has_tests = %s", result)
    return result


def detect_license(text: str) -> Optional[str]:
    """
    Return the canonical license name if detected, else None.

    Checks from most-specific to most-generic so 'GPL-3.0' beats 'GPL'.

    Args:
        text: Any block of text.

    Returns:
        e.g. "MIT", "Apache 2.0", or None.
    """
    for name, patterns in _COMPILED_LICENSE:
        if any(p.search(text) for p in patterns):
            logger.debug("heuristics: detected license '%s'", name)
            return name
    logger.debug("heuristics: no license detected")
    return None