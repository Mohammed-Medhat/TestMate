"""
LangChain LLM Wrapper for TestMate's Local Qwen2.5-Coder-7B + LoRA Model
==========================================================================
Thin wrapper that lets LangChain/LangGraph invoke the existing local model
without changing any model loading or inference logic.
"""

from typing import Any, List, Optional
from langchain_core.language_models.llms import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
import torch


class QwenLoRALLM(LLM):
    """LangChain-compatible wrapper around the locally loaded Qwen2.5-Coder-7B + LoRA model.

    This does NOT change how the model is loaded or run — it simply adapts
    the existing `generate_test()` function to the LangChain LLM interface.
    """

    model: Any = None
    tokenizer: Any = None
    max_new_tokens: int = 1536
    temperature: float = 0.5
    top_p: float = 0.9
    repetition_penalty: float = 1.05

    class Config:
        arbitrary_types_allowed = True

    @property
    def _llm_type(self) -> str:
        return "qwen-lora-local"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Run the local Qwen model on the given prompt.

        Accepts optional kwargs:
            - max_new_tokens (int)
            - temperature (float)
            - messages (list[dict])  — if provided, uses chat template instead of raw prompt
        """
        _max_tokens = kwargs.get("max_new_tokens", self.max_new_tokens)
        _temp = kwargs.get("temperature", self.temperature)
        _messages = kwargs.get("messages", None)

        if _messages:
            # Chat template mode (used by the autonomous loop)
            chat = self.tokenizer.apply_chat_template(
                _messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.tokenizer(
                chat, return_tensors="pt", truncation=True, max_length=2048
            ).to(self.model.device)
        else:
            # Raw prompt mode
            inputs = self.tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=2048
            ).to(self.model.device)

        _do_sample = _temp > 0.01
        with torch.inference_mode():
            out = self.model.generate(
                **inputs,
                max_new_tokens=_max_tokens,
                do_sample=_do_sample,
                temperature=_temp if _do_sample else None,
                top_p=self.top_p if _do_sample else None,
                repetition_penalty=self.repetition_penalty,
                pad_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
            )

        raw = self.tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        return raw

    @property
    def _identifying_params(self) -> dict:
        return {
            "model_type": "qwen2.5-coder-7b-lora",
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
        }
