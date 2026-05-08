#!/usr/bin/env python3
# src/utils/demo.py
"""
Standalone demo for the Test Scenario Generator pipeline.
"""

from __future__ import annotations

import json
import logging
import os
import sys

# ── Logging setup ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG,
    format="%(levelname)-8s %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("demo")

# ---------------------------------------------------------------------------
# Sample fixtures (simulate what cleaner.clean_readme() + user form produce)
# ---------------------------------------------------------------------------

SAMPLE_CLEANED_README = {
    "prose": """
# FastAPI Todo API

A production-ready REST API for managing todos, built with FastAPI and PostgreSQL.

## Features
- Full CRUD for todos (create, read, update, delete)
- JWT authentication with refresh tokens
- Role-based access control (admin / regular user)
- Pagination and search on all list endpoints
- OpenAPI docs auto-generated

## Tech Stack
Python 3.11, FastAPI, PostgreSQL, Redis, Docker, pytest

## License
MIT
""",
    "code_blocks": [
        "pip install -r requirements.txt",
        "docker-compose up -d",
        "uvicorn app.main:app --reload",
        "pytest tests/ --cov=app",
    ],
    "sections": {
        "installation": "pip install -r requirements.txt\ndocker-compose up -d",
        "usage":        "uvicorn app.main:app --reload\ncurl http://localhost:8000/todos",
        "testing":      "pytest tests/ --cov=app --cov-report=html",
        "license":      "MIT",
        "tech":         "Python 3.11, FastAPI, PostgreSQL, Redis, Docker",
    },
}

SAMPLE_USER_INPUT = {
    "description":  "A REST API for managing personal todos with user authentication.",
    "problems":     "Users can delete other people's todos because the ownership check is missing.",
    "expected":     "Each user should only be able to read, update, and delete their own todos.",
    "edge_cases":   "Empty todo title, very long description (>10 000 chars), concurrent deletes of the same todo.",
}

# ---------------------------------------------------------------------------
# Optional: load the real model if MODEL_PATH is set
# ---------------------------------------------------------------------------

def _load_model_if_available():
    model_path = os.getenv("MODEL_PATH", "")
    if not model_path or not os.path.exists(model_path):
        logger.info("demo: no MODEL_PATH set or file not found — running in heuristics-only mode")
        return None

    try:
        from llama_cpp import Llama  # type: ignore
        logger.info("demo: loading model from %s …", model_path)
        model = Llama(
            model_path=model_path,
            n_ctx=4096,
            n_threads=8,
            n_gpu_layers=int(os.getenv("GPU_LAYERS", "0")),
            verbose=False,
        )
        logger.info("demo: model loaded ✓")
        return model
    except Exception as exc:
        logger.warning("demo: model load failed (%s) — falling back to heuristics-only", exc)
        return None

# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------

def main() -> None:
    # التعديل هنا عشان يشوف الفولدرات الجديدة
    from src.services.pipeline import run_pipeline 
    from src.models.models import FinalOutput

    logger.info("=" * 60)
    logger.info("Test Scenario Generator — Demo")
    logger.info("=" * 60)

    model = _load_model_if_available()

    logger.info("demo: running pipeline …")
    result: FinalOutput = run_pipeline(
        cleaned_readme=SAMPLE_CLEANED_README,
        user_input=SAMPLE_USER_INPUT,
        model=model,
    )

    # ── Pretty-print the result ──────────────────────────────────────────────
    output_dict = result.to_api_dict()
    print("\n" + "=" * 60)
    print("PIPELINE OUTPUT")
    print("=" * 60)
    print(json.dumps(output_dict, indent=2, ensure_ascii=False))

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print(f"Project      : {result.features.project_name}")
    print(f"Tech Stack   : {', '.join(result.features.tech_stack)}")
    print(f"Has Tests    : {result.features.has_tests}")
    print(f"License      : {result.features.license_type}")
    print(f"Scenarios    : {len(result.test_scenarios)}")
    print(f"Low-conf     : {result.low_confidence_fields or 'none'}")
    print("─" * 60)

    by_type: dict[str, int] = {}
    for s in result.test_scenarios:
        by_type[s.type] = by_type.get(s.type, 0) + 1

    print("Scenario mix :")
    for t, count in sorted(by_type.items()):
        print(f"  {t:<12} {count}")
    print("─" * 60)

if __name__ == "__main__":
    
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    main()