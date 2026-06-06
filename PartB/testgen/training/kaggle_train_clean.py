# ============================================================
# KAGGLE: Clean QLoRA training for Test Generation
# ============================================================
# Contamination-free retrain for a defensible evaluation:
#   - TRAIN on MBPP only (decontaminated vs HumanEval)
#   - so HumanEval AND realistic sets (testgenevallite, real repos)
#     stay clean held-out test sets.
#   - Base model = Qwen2.5-Coder-7B-INSTRUCT (matches inference;
#     the old adapter was trained on the non-instruct base — a bug).
#
# Drops SWE-bench (so realistic test sets stay clean) and Magicoder
# (its "tests" were fixed templates with `pass` bodies — no signal).
#
# Run as a Kaggle notebook (T4/P100). Internet must be ON to pull MBPP +
# HumanEval (both tiny). Output adapter -> /kaggle/working/testgen_clean/final
# Download it and place at PartB/models/graphrag_lora_clean/final/.
#
# All stdout is ASCII-only (cp1252-safe).
# ============================================================

# CELL 1: deps
"""
!pip install -q transformers datasets peft bitsandbytes accelerate
"""

import json
import re
from datetime import datetime
import os

import torch
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer,
    DataCollatorForLanguageModeling, BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset, Dataset

# Base must match inference (PartB loads the Instruct model).
MODEL_NAME = "Qwen/Qwen2.5-Coder-7B-Instruct"
MAX_LENGTH = 2048
OUTPUT_DIR = "/kaggle/working/testgen_clean"

LORA_CONFIG = {
    "r": 16, "lora_alpha": 32,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "lora_dropout": 0.05, "bias": "none", "task_type": "CAUSAL_LM",
}

_TOKEN_RE = re.compile(r"[a-zA-Z_][a-zA-Z_0-9]*")
_STOP = {
    "def", "return", "for", "in", "if", "else", "elif", "while", "the", "a",
    "an", "of", "to", "and", "or", "is", "are", "be", "function", "write",
    "python", "given", "list", "number", "numbers", "string", "value", "true",
    "false", "none", "self", "import", "from", "assert", "print", "range",
    "len", "int", "str", "float", "bool",
}


def _tokens(text):
    return {t.lower() for t in _TOKEN_RE.findall(text or "")} - _STOP


def _jaccard(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def clean_mbpp_task_ids(threshold=0.35):
    """Return MBPP task_ids with no HumanEval near-duplicate."""
    he = load_dataset("openai_humaneval")["test"]
    he_tok = [_tokens(f"{r['prompt']}\n{r['canonical_solution']}") for r in he]
    mb = load_dataset("mbpp", "full")
    rows = []
    for split in ("train", "test", "validation", "prompt"):
        if split in mb:
            rows.extend(list(mb[split]))
    kept, dropped = [], 0
    for r in rows:
        tok = _tokens(f"{r['text']}\n{r['code']}")
        best = max((_jaccard(tok, ht) for ht in he_tok), default=0.0)
        if best >= threshold:
            dropped += 1
        else:
            kept.append(r["task_id"])
    print(f"MBPP decontam: kept {len(kept)} / {len(rows)} (dropped {dropped})")
    return set(kept), {r["task_id"]: r for r in rows}


def _asserts_to_pytest(entry_point, test_list, setup_code):
    """Turn MBPP's bare `assert ...` list into one pytest function so the model
    learns pytest structure (not just bare asserts)."""
    body = []
    if setup_code:
        for line in setup_code.splitlines():
            body.append(f"    {line}")
    for a in test_list:
        body.append(f"    {a.strip()}")
    fn = entry_point or "function"
    return "import pytest\n\n\ndef test_" + fn + "():\n" + "\n".join(body) + "\n"


def _entry_point(code, test_list):
    """The function actually under test = the symbol called in the asserts.
    MBPP solutions often include helper classes/functions, so deriving the
    entry point from `code`'s first def is wrong; parse the assert instead."""
    for a in test_list or []:
        # e.g. "assert max_chain_length(arr, 4) == 3" -> max_chain_length
        m = re.search(r"assert\s+([a-zA-Z_]\w*)\s*\(", a)
        if m:
            return m.group(1)
    m = re.search(r"def\s+([a-zA-Z_]\w*)\s*\(", code or "")
    return m.group(1) if m else "function"


def build_training_samples():
    clean_ids, by_id = clean_mbpp_task_ids()
    samples = []
    for tid, r in by_id.items():
        if tid not in clean_ids:
            continue
        code = r.get("code", "")
        test_list = r.get("test_list", [])
        if not code or not test_list:
            continue
        entry = _entry_point(code, test_list)
        pytest_block = _asserts_to_pytest(entry, test_list,
                                          r.get("test_setup_code", ""))
        prompt = (
            "=== COMPREHENSIVE TEST GENERATION ===\n\n"
            "## Source Code\n```python\n" + code + "\n```\n\n"
            "## Context\nGenerate comprehensive pytest unit tests for `"
            + entry + "`.\nInclude basic, edge, and error cases.\n\n"
            "## Generated Tests\n```python\n" + pytest_block + "```"
        )
        samples.append({"type": "mbpp_clean", "text": prompt})
    print(f"Built {len(samples)} clean MBPP training samples")
    return samples


def train():
    print("=" * 60)
    print("CLEAN TEST-GEN TRAINING  (MBPP-only, Instruct base)")
    print("=" * 60)
    print(f"GPU: {torch.cuda.get_device_name(0)}  "
          f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    samples = build_training_samples()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize(s):
        t = tokenizer(s["text"], truncation=True, max_length=MAX_LENGTH,
                      padding="max_length", return_tensors=None)
        t["labels"] = t["input_ids"].copy()
        return t

    dataset = Dataset.from_list([tokenize(s) for s in samples])
    print(f"Dataset: {len(dataset)} samples")

    quant = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, quantization_config=quant, device_map="auto",
        trust_remote_code=True, torch_dtype=torch.float16,
    )
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, LoraConfig(**LORA_CONFIG))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    run_name = f"clean_{datetime.now().strftime('%Y%m%d_%H%M')}"
    args = TrainingArguments(
        output_dir=f"{OUTPUT_DIR}/{run_name}",
        num_train_epochs=2,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4, warmup_ratio=0.1,
        logging_steps=10, save_steps=200, save_total_limit=2,
        fp16=True, optim="paged_adamw_8bit",
        gradient_checkpointing=True, remove_unused_columns=False,
        report_to="none",
    )
    trainer = Trainer(
        model=model, args=args, train_dataset=dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    print("\nStarting clean training...")
    trainer.train()

    final = f"{OUTPUT_DIR}/final"
    model.save_pretrained(final)
    tokenizer.save_pretrained(final)
    with open(f"{final}/testgen_meta.json", "w", encoding="utf-8") as f:
        json.dump({
            "task": "clean_test_generation",
            "base_model": MODEL_NAME,
            "train_data": "MBPP (decontaminated vs HumanEval)",
            "samples": len(samples),
            "epochs": 2,
            "note": "HumanEval + testgenevallite are clean held-out test sets",
        }, f, indent=2)
    print(f"\nTRAINING COMPLETE -> {final}")
    print("Download and place at PartB/models/graphrag_lora_clean/final/")


if __name__ == "__main__":
    train()
