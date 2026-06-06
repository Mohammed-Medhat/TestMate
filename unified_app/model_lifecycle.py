"""
Model lifecycle management for the unified TestMate app.

Part A and Part B/C share the same GPU but use different models.
They must NEVER be resident simultaneously — this module enforces
sequential load → work → flush cycles.

Multi-adapter sharing (Part B + Part C):
    Both PartB and PartC use Qwen2.5-Coder-7B-Instruct with rank-16 LoRA adapters.
    We load the base model ONCE, attach both adapters under named keys,
    then swap cheaply between them via set_adapter().

Usage:
    from model_lifecycle import part_a_model, part_b_model, part_c_model, qwen_session

    # Standalone PartB:
    with part_b_model() as (model, tokenizer):
        execute_part_b(model=model, tokenizer=tokenizer, ...)

    # Standalone PartC:
    with part_c_model() as (model, tokenizer):
        execute_part_c(model=model, tokenizer=tokenizer, ...)

    # PartB then PartC (cheapest — one base load, adapter swap):
    with qwen_session() as sess:
        sess.use("partb")
        execute_part_b(model=sess.model, tokenizer=sess.tokenizer, ...)
        sess.use("partc")
        execute_part_c(model=sess.model, tokenizer=sess.tokenizer, ...)
"""
from __future__ import annotations

import gc
import logging
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────
_ROOT        = Path(__file__).parent.parent
_PART_A_README  = _ROOT / "PartA" / "readme_extractor"
_PART_B_TESTGEN = _ROOT / "PartB" / "testgen"
_PART_C_ROOT    = _ROOT / "PartC"

# LoRA adapter paths (relative to repo root — same absolute paths as main.py)
_PARTB_LORA_PATH = _ROOT / "PartB" / "value_reasoning_model"
_PARTC_LORA_PATH = _ROOT / "PartC" / "models" / "adapter"


def flush_gpu() -> None:
    """Hard-flush all GPU memory. Call between Part A and Part B/C loads."""
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


# ── Part A ─────────────────────────────────────────────────────────────────

@contextmanager
def part_a_model():
    """
    Load Part A's HF model, yield it, then delete and flush GPU on exit.

    with part_a_model() as model:
        result = execute_part_a_readme(content, model=model)
    """
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


# ── Qwen multi-adapter session (Part B + Part C) ───────────────────────────

@dataclass
class QwenSession:
    """
    Holds a single Qwen2.5-Coder-7B base loaded with both PartB and PartC adapters.
    Use `.use("partb")` or `.use("partc")` to hot-swap the active adapter.
    """
    model: Any = field(default=None, repr=False)
    tokenizer: Any = field(default=None, repr=False)
    _active: Optional[str] = field(default=None, repr=False)
    _adapters_loaded: set = field(default_factory=set, repr=False)

    def use(self, adapter: str) -> None:
        """Activate a named adapter ("partb" or "partc"). No-op if already active."""
        if adapter not in self._adapters_loaded:
            raise ValueError(f"Adapter '{adapter}' was not loaded into this session. "
                             f"Loaded adapters: {self._adapters_loaded}")
        if self._active == adapter:
            return
        try:
            self.model.set_adapter(adapter)
            logger.info("Swapped to adapter: %s", adapter)
            self._active = adapter
        except Exception as exc:
            logger.warning("set_adapter('%s') failed: %s — adapter may already be active", adapter, exc)


def _load_qwen_with_adapters(
    partb_lora: Path,
    partc_lora: Path,
    use_lora: bool = True,
) -> tuple:
    """
    Load Qwen2.5-Coder-7B-Instruct 4-bit and attach both LoRA adapters.
    Returns (model, tokenizer, adapters_loaded: set).
    """
    import os
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    # Re-use PartB's main.py BASE_MODEL path (absolute, matches the installed model)
    sys.path.insert(0, str(_PART_B_TESTGEN))
    import main as _partb_main
    BASE_MODEL = _partb_main.BASE_MODEL

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
        "expandable_segments:True,garbage_collection_threshold:0.6"
    )

    logger.info("Loading Qwen2.5-Coder-7B base (4-bit)...")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True, use_fast=True)
    tokenizer.truncation_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    total_vram = torch.cuda.mem_get_info()[1] / 1e9 if torch.cuda.is_available() else 0
    compute_dtype = torch.float16 if total_vram <= 10 else torch.bfloat16
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
    )

    # Accuracy-preserving decode speedups:
    #   attn_implementation="sdpa": PyTorch fused SDPA (numerically equivalent to
    #     eager attention, decode-equivalent under fp16/bf16).
    #   use_cache=True: enable KV cache so generate() reuses past keys/values.
    # Both are no-op on accuracy; cut wall-clock per generate() call by ~10–25%.
    _load_kwargs = dict(
        quantization_config=bnb,
        device_map={"": 0},
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        torch_dtype=compute_dtype,
    )
    try:
        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            attn_implementation="sdpa",
            **_load_kwargs,
        )
    except (TypeError, ValueError) as _e:
        logger.info("attn_implementation='sdpa' not supported by this transformers/model build (%s); falling back to default.", _e)
        base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, **_load_kwargs)
    base.config.use_cache = True
    base.eval()
    logger.info("Base model loaded.")

    adapters_loaded: set = set()

    if not use_lora:
        logger.info("LoRA skipped — using raw base model.")
        return base, tokenizer, adapters_loaded

    # Attach PartB adapter first (makes it the default)
    # Use as_posix() to avoid PEFT treating Windows backslash paths as HF repo IDs.
    model = base
    if partb_lora.exists():
        partb_path_str = partb_lora.resolve().as_posix()
        model = PeftModel.from_pretrained(base, partb_path_str, adapter_name="partb")
        adapters_loaded.add("partb")
        logger.info("PartB LoRA attached as 'partb'.")
    else:
        logger.warning("PartB LoRA not found at %s — skipping", partb_lora)

    # Attach PartC adapter under a separate name
    # Guard: check the actual weights file exists, not just the directory.
    # adapter_config.json alone is not enough — the weights (safetensors/bin) must be present.
    _partc_weights = (
        (partc_lora / "adapter_model.safetensors").exists()
        or (partc_lora / "adapter_model.bin").exists()
    )
    if not partc_lora.exists():
        logger.warning("PartC LoRA directory not found at %s — skipping", partc_lora)
    elif not _partc_weights:
        logger.warning(
            "PartC LoRA directory found (%s) but adapter_model.safetensors/.bin is missing. "
            "Run PartC training (python PartC/training/train.py) to generate the weights. "
            "Auto-repair will be disabled until weights are present.",
            partc_lora,
        )
    else:
        try:
            # Use as_posix() to avoid PEFT treating Windows backslash paths as HF repo IDs
            partc_path_str = partc_lora.resolve().as_posix()
            if "partb" in adapters_loaded:
                model.load_adapter(partc_path_str, adapter_name="partc")
            else:
                model = PeftModel.from_pretrained(base, partc_path_str, adapter_name="partc")
            adapters_loaded.add("partc")
            logger.info("PartC LoRA attached as 'partc'.")
        except Exception as e:
            logger.warning(
                "PartC LoRA failed to load (%s). Auto-repair disabled. "
                "PartB will still run normally. Error: %s",
                partc_lora, e,
            )

    model.eval()
    torch.cuda.synchronize()
    return model, tokenizer, adapters_loaded


@contextmanager
def qwen_session(use_lora: bool = True):
    """
    Context manager that loads Qwen once with both PartB and PartC adapters,
    yields a QwenSession for cheap adapter swapping, then flushes GPU on exit.

    with qwen_session() as sess:
        sess.use("partb")
        execute_part_b(model=sess.model, tokenizer=sess.tokenizer, ...)
        sess.use("partc")
        execute_part_c(model=sess.model, tokenizer=sess.tokenizer, ...)
    """
    logger.info("Starting Qwen multi-adapter session...")
    model, tokenizer, adapters_loaded = _load_qwen_with_adapters(
        partb_lora=_PARTB_LORA_PATH,
        partc_lora=_PARTC_LORA_PATH,
        use_lora=use_lora,
    )
    sess = QwenSession(model=model, tokenizer=tokenizer, _adapters_loaded=adapters_loaded)

    # Default to partb if available
    if "partb" in adapters_loaded:
        sess._active = "partb"
    elif "partc" in adapters_loaded:
        sess._active = "partc"

    try:
        yield sess
    finally:
        logger.info("Closing Qwen session → flushing GPU...")
        del model, tokenizer
        flush_gpu()


@contextmanager
def part_b_model(use_lora: bool = True):
    """
    Load PartB model. Uses the shared qwen_session so the base model is loaded
    once; if PartC adapter is also present it will be co-resident but inactive.

    with part_b_model() as (model, tokenizer):
        execute_part_b(model=model, tokenizer=tokenizer, ...)
    """
    with qwen_session(use_lora=use_lora) as sess:
        if use_lora and "partb" in sess._adapters_loaded:
            sess.use("partb")
        yield sess.model, sess.tokenizer


@contextmanager
def part_c_model(use_lora: bool = True):
    """
    Load PartC APR model (Qwen base + PartC LoRA).

    with part_c_model() as (model, tokenizer):
        execute_part_c(model=model, tokenizer=tokenizer, ...)
    """
    with qwen_session(use_lora=use_lora) as sess:
        if use_lora and "partc" in sess._adapters_loaded:
            sess.use("partc")
        elif use_lora:
            logger.warning("PartC adapter not loaded — using active adapter as fallback")
        yield sess.model, sess.tokenizer
