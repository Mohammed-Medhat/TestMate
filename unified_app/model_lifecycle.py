"""
Model lifecycle management for the unified TestMate app.

Part A and Part B share the same GPU but use different models.
They must NEVER be resident simultaneously — this module enforces
sequential load → work → flush cycles.

Usage:
    from model_lifecycle import with_part_a_model, with_part_b_model

    # In combined mode:
    with_part_a_model(lambda m: do_part_a(m))
    # GPU flushed here automatically
    with_part_b_model(lambda m, t: do_part_b(m, t))
"""
from __future__ import annotations

import gc
import logging
import sys
from contextlib import contextmanager
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).parent.parent
_PART_A_README = _ROOT / "PartA" / "readme_extractor"
_PART_B_TESTGEN = _ROOT / "PartB" / "testgen"


def flush_gpu() -> None:
    """Hard-flush all GPU memory. Call between Part A and Part B loads."""
    try:
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.synchronize()
            logger.info(
                "GPU flushed — %.1fGB free",
                torch.cuda.mem_get_info()[0] / 1e9,
            )
    except ImportError:
        gc.collect()


@contextmanager
def part_a_model():
    """
    Context manager that loads Part A's HF model, yields it,
    then deletes it and flushes GPU on exit.

    with part_a_model() as model:
        result = execute_part_a_readme(content, model=model)
    """
    # Add Part A to path so its internal imports resolve
    sys.path.insert(0, str(_PART_A_README))
    sys.path.insert(0, str(_PART_A_README / "src"))

    from src.utils.model_loader import load_model
    logger.info("Loading Part A model...")
    model = load_model()
    try:
        yield model
    finally:
        logger.info("Unloading Part A model → flushing GPU...")
        del model
        flush_gpu()


@contextmanager
def part_b_model(use_lora: bool = True):
    """
    Context manager that loads Part B's 4-bit Qwen model (with or without LoRA).

    Args:
        use_lora: True (default) = load base + value_reasoning LoRA.
                  False = load base only (raw 4-bit Qwen, no adapter).
    """
    sys.path.insert(0, str(_PART_B_TESTGEN))

    from main import load_model  # type: ignore[import]
    mode = "with LoRA" if use_lora else "BASE ONLY"
    logger.info("Loading Part B model [%s]...", mode)
    model, tokenizer = load_model(use_lora=use_lora)
    try:
        yield model, tokenizer
    finally:
        logger.info("Unloading Part B model → flushing GPU...")
        del model, tokenizer
        flush_gpu()
