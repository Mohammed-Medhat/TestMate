import sys
import os
import torch
from typing import Any, Optional

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
for path in [current_dir, parent_dir]:
    if path not in sys.path:
        sys.path.insert(0, path)

try:
    from config import REPAIR_TEMPERATURES, TOP_P, MAX_NEW_TOKENS
except ImportError:
    REPAIR_TEMPERATURES = [0.2, 0.5, 0.8]
    TOP_P = 0.95
    MAX_NEW_TOKENS = 1024

_model = None
_tokenizer = None


def reset_model():
    """Unload the current model and tokenizer, freeing GPU memory."""
    global _model, _tokenizer
    if _model is not None:
        import gc
        _model = None
        _tokenizer = None
        gc.collect()
        torch.cuda.empty_cache()
        print("RELOAD Model unloaded and GPU memory freed.")


def load_testmate_model():
    global _model, _tokenizer
    if _model is not None:
        return _model, _tokenizer
    print("LOADING Loading TestMate (PartC APR model)...")
    from inference import load_model
    _model, _tokenizer = load_model()
    print("OK Model loaded successfully")
    return _model, _tokenizer


def run_testmate(
    prompt: str,
    attempt: int = 1,
    model: Optional[Any] = None,
    tokenizer: Optional[Any] = None,
) -> str:
    """
    Run the repair prompt through the model.

    Args:
        prompt:    The repair prompt text.
        attempt:   Attempt number (controls temperature schedule).
        model:     Pre-loaded model (loaded from disk if None).
        tokenizer: Pre-loaded tokenizer (loaded from disk if None).
    """
    _m = model
    _t = tokenizer
    if _m is None or _t is None:
        _m, _t = load_testmate_model()

    try:
        messages = [
            {"role": "system", "content": "Expert APR agent. Output ONLY the complete fixed python function with no line numbers , no explanation no markdown."},
            {"role": "user", "content": prompt}
        ]
        chat_prompt = _t.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = _t(chat_prompt, return_tensors="pt").to("cuda")

        current_temp = REPAIR_TEMPERATURES[min(attempt - 1, len(REPAIR_TEMPERATURES) - 1)]

        with torch.no_grad():
            outputs = _m.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=current_temp,
                top_p=TOP_P,
                eos_token_id=_t.eos_token_id,
                pad_token_id=_t.pad_token_id
            )

        fixed_code = _t.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        return fixed_code.strip()
    except Exception as e:
        print(f"Error in model inference: {e}")
        return ""
