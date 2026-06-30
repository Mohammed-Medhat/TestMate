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
import os
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────
# Dev layout:      model_lifecycle.py lives in unified_app/ → parent.parent is repo root
# Packaged layout: model_lifecycle.py lives in resources/   → parent is resources (PartB/ is there)
_HERE        = Path(__file__).parent
_ROOT        = _HERE if (_HERE / "PartB").exists() else _HERE.parent
_PART_A_README  = _ROOT / "PartA" / "readme_extractor"
_PART_B_TESTGEN = _ROOT / "PartB" / "testgen"
_PART_C_ROOT    = _ROOT / "PartC"

# LoRA adapter paths
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
            # Second pass: finalizers run during the first collect/empty_cache
            # (e.g. accelerate hook objects, bnb Params4bit buffers) can leave
            # behind newly-unreferenced blocks that only become collectible —
            # and only get returned to the driver — on a follow-up pass.
            gc.collect()
            torch.cuda.empty_cache()
            logger.info(
                "GPU flushed — %.1fGB free",
                torch.cuda.mem_get_info()[0] / 1e9,
            )
    except ImportError:
        gc.collect()


def _free_vram_gb() -> float:
    """Free GPU memory in GB, or +inf if CUDA isn't available (nothing to wait for)."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.mem_get_info()[0] / 1e9
    except Exception:
        pass
    return float("inf")


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

    # Extract BASE_MODEL from main.py by text search — avoids executing main.py's
    # module-level code (torch.cuda.set_device, rag_store init_db, etc.) which
    # segfaults when run from a background worker thread on Windows.
    import re as _re
    _main_src = (_PART_B_TESTGEN / "main.py").read_text(encoding="utf-8", errors="ignore")
    _bm = _re.search(r'^BASE_MODEL\s*=\s*[rR]?"([^"]+)"', _main_src, _re.MULTILINE)
    BASE_MODEL = _bm.group(1) if _bm else r"D:\donwloader\Qwen2.5-Coder-7B-Instruct"
    logger.info("BASE_MODEL resolved to: %s", BASE_MODEL)

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
        "expandable_segments:True,garbage_collection_threshold:0.6"
    )

    logger.info("Loading Qwen2.5-Coder-7B base (4-bit) from %s ...", BASE_MODEL)
    logger.info("PartB LoRA: %s (exists=%s)", partb_lora, partb_lora.exists())
    gc.collect()
    if torch.cuda.is_available():
        # Explicitly initialize the CUDA context from this thread before BitsAndBytes
        # tries to load its native CUDA kernels — avoids a Windows SEH crash when the
        # context is first created from a non-main thread inside a native DLL.
        logger.info("Initializing CUDA context...")
        torch.cuda.init()
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.cuda.set_device(0)
        free_gb = torch.cuda.mem_get_info()[0] / 1e9
        total_gb = torch.cuda.mem_get_info()[1] / 1e9
        logger.info("VRAM before load: %.1f GB free / %.1f GB total", free_gb, total_gb)
        torch.cuda.empty_cache()

    logger.info("Loading tokenizer from %s ...", BASE_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True, use_fast=True)
    tokenizer.truncation_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    logger.info("Tokenizer loaded.")

    total_vram = torch.cuda.mem_get_info()[1] / 1e9 if torch.cuda.is_available() else 0
    compute_dtype = torch.float16 if total_vram <= 10 else torch.bfloat16
    logger.info("compute_dtype=%s  total_vram=%.1f GB", compute_dtype, total_vram)
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
    logger.info("Calling AutoModelForCausalLM.from_pretrained (4-bit BnB)...")
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
    if torch.cuda.is_available():
        used_gb = (torch.cuda.mem_get_info()[1] - torch.cuda.mem_get_info()[0]) / 1e9
        logger.info("Base model loaded — VRAM used: %.1f GB", used_gb)
    else:
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


# ── Keep-alive resident PartB model (for interactive chat refinement) ───────
#
# The standard context managers above load → work → flush. That's correct for
# batch runs, but it means the model is gone the moment a run ends — so a chat
# turn afterwards would pay a ~30–45 s reload every message.
#
# PartBResident holds the model resident for a short window after the last use
# (default 10 min, env TESTMATE_KEEPALIVE_SEC) so chat turns are instant, then
# auto-unloads when idle to free VRAM. PartA / PartC paths MUST call
# force_release() before they load, since the Qwen model and Part A's model
# cannot coexist in GPU memory.

class _PartBResident:
    """Singleton that keeps the PartB Qwen model warm between chat turns."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.model: Any = None
        self.tokenizer: Any = None
        self._adapters: set = set()
        self._use_lora: bool = True
        self._last_used: float = 0.0
        self._reaper: Optional[threading.Thread] = None
        try:
            self._keepalive = float(os.environ.get("TESTMATE_KEEPALIVE_SEC", "600"))
        except ValueError:
            self._keepalive = 600.0

    def acquire(self, use_lora: bool = True, status_cb: Optional[Callable[[str], None]] = None):
        """Return (model, tokenizer), loading (or reloading) if cold or the
        LoRA flag changed. Cheap when already warm."""
        with self._lock:
            if self.model is None or self._use_lora != use_lora:
                if self.model is not None:
                    self._flush_locked()
                if status_cb:
                    try:
                        status_cb("warming up the model (first message may take ~30s)…")
                    except Exception:
                        pass
                logger.info("PartBResident: loading model (use_lora=%s)…", use_lora)
                self.model, self.tokenizer, self._adapters = _load_qwen_with_adapters(
                    partb_lora=_PARTB_LORA_PATH,
                    partc_lora=_PARTC_LORA_PATH,
                    use_lora=use_lora,
                )
                self._use_lora = use_lora
                if use_lora and "partb" in self._adapters:
                    try:
                        self.model.set_adapter("partb")
                    except Exception:
                        pass
                self._start_reaper_locked()
            self._last_used = time.monotonic()
            return self.model, self.tokenizer

    def touch(self) -> None:
        """Reset the idle timer (call after each run / chat turn)."""
        with self._lock:
            if self.model is not None:
                self._last_used = time.monotonic()

    def force_release(self, min_free_gb: Optional[float] = None, timeout_s: float = 20.0) -> bool:
        """Immediately unload + flush. Call from any PartA/PartC path before it
        loads its own model (VRAM-coexistence rule).

        `del` + `empty_cache()` doesn't always hand memory back to the driver
        on the first try (see `_flush_locked`/`flush_gpu`), and a PartA load
        that starts while VRAM is still pinned can crash the whole process
        (a 4-bit BnB load failing OOM has been observed to take down the
        parent server, not just raise). So after flushing, poll
        `mem_get_info()` and retry the flush for up to `timeout_s` until at
        least `min_free_gb` is free. Returns True if it's safe to proceed,
        False if the caller should back off (e.g. respond 503) instead of
        starting a load that's likely to crash.

        If CUDA isn't available there's nothing to wait for — always True.
        """
        if min_free_gb is None:
            try:
                min_free_gb = float(os.environ.get("TESTMATE_PARTA_MIN_FREE_GB", "4.0"))
            except ValueError:
                min_free_gb = 4.0

        with self._lock:
            self._flush_locked()
            free = _free_vram_gb()
            if free == float("inf"):
                return True
            deadline = time.monotonic() + timeout_s
            while free < min_free_gb and time.monotonic() < deadline:
                time.sleep(1.0)
                flush_gpu()
                free = _free_vram_gb()
            if free < min_free_gb:
                logger.warning(
                    "force_release: only %.1fGB free after flush (wanted %.1fGB) — "
                    "a PartA/PartC load now may fail or crash",
                    free, min_free_gb,
                )
                return False
            return True

    @property
    def is_resident(self) -> bool:
        with self._lock:
            return self.model is not None

    def _flush_locked(self) -> None:
        if self.model is None:
            return
        before = _free_vram_gb()
        logger.info(
            "PartBResident: releasing model → flushing GPU (%.1fGB free before)…",
            before,
        )
        try:
            del self.model
        except Exception:
            pass
        try:
            del self.tokenizer
        except Exception:
            pass
        self.model = None
        self.tokenizer = None
        self._adapters = set()
        flush_gpu()
        after = _free_vram_gb()
        logger.info(
            "PartBResident: released — %.1fGB free after (was %.1fGB)",
            after, before,
        )

    def _start_reaper_locked(self) -> None:
        if self._reaper is not None and self._reaper.is_alive():
            return

        def _loop() -> None:
            while True:
                time.sleep(30)
                with self._lock:
                    if self.model is None:
                        return  # nothing resident — let the thread die
                    idle = time.monotonic() - self._last_used
                    if idle > self._keepalive:
                        logger.info(
                            "PartBResident: idle %.0fs > %.0fs — auto-unloading",
                            idle, self._keepalive,
                        )
                        self._flush_locked()
                        return

        self._reaper = threading.Thread(target=_loop, daemon=True, name="partb-keepalive-reaper")
        self._reaper.start()


# Process-wide singleton.
PARTB_RESIDENT = _PartBResident()
