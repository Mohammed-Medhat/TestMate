# src/scenario_generator/llm_engine.py
from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional, Type, TypeVar

from pydantic import BaseModel

logger = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)

try:
    import instructor
    _INSTRUCTOR_AVAILABLE = True
    logger.debug("llm_engine: instructor available")
except ImportError:
    _INSTRUCTOR_AVAILABLE = False
    logger.warning("llm_engine: instructor not installed — falling back to raw JSON parsing")


def _strip_json_fences(text: str) -> str:
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```\s*", "", text)
    return text.strip()


def _parse_raw_json(raw: str, response_model: Optional[Type[T]]) -> Any:
    clean = _strip_json_fences(raw)
    # Pull out first {...} or [...] block
    match = re.search(r"(\{.*\}|\[.*\])", clean, re.DOTALL)
    if match:
        clean = match.group(0)

    try:
        data = json.loads(clean)
    except json.JSONDecodeError as exc:
        logger.warning("llm_engine: JSON parse failed: %s | raw: %.200s", exc, raw)
        return {} if response_model is None else response_model()

    if response_model is not None:
        try:
            return response_model(**data) if isinstance(data, dict) else response_model()
        except Exception as ve:
            logger.warning("llm_engine: Pydantic validation failed: %s", ve)
            return response_model()

    return data


def _call_with_instructor(prompt, model, response_model, system_prompt, max_tokens, temperature):
    patched_create = instructor.patch(
        create=model.create_chat_completion_openai_v1,
        mode=instructor.Mode.MD_JSON,
    )
    return patched_create(
        response_model=response_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        max_retries=2,
    )


def _call_raw(prompt, model, response_model, system_prompt, max_tokens, temperature):
    output = model.create_chat_completion(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    raw: str = output["choices"][0]["message"]["content"]
    logger.debug("llm_engine: raw output (%.300s)", raw)
    return raw  # FIX #8: return the RAW STRING so caller can parse it


def call_llm(
    prompt: str,
    model: Any,
    response_model: Optional[Type[T]] = None,
    system_prompt: str = "You are a helpful assistant. Return valid JSON only.",
    max_tokens: int = 1_000,
    temperature: float = 0.2,
) -> Any:
    """
    Returns:
      - response_model instance  when response_model is provided and instructor is available
      - raw string               when response_model is None  (FIX #8 — scenario_generator uses this)
      - {} / response_model()    on failure
    """
    if model is None:
        logger.error("llm_engine: model is None")
        return response_model() if response_model else {}

    try:
        if _INSTRUCTOR_AVAILABLE and response_model is not None:
            return _call_with_instructor(
                prompt, model, response_model, system_prompt, max_tokens, temperature
            )
        else:
            raw = _call_raw(
                prompt, model, response_model, system_prompt, max_tokens, temperature
            )
            # FIX #8: if caller wants a model, parse here; if not, return raw string
            if response_model is not None:
                return _parse_raw_json(raw, response_model)
            return raw   # scenario_generator expects the raw string

    except Exception as exc:
        logger.error("llm_engine: call failed: %s", exc)
        return response_model() if response_model else {}