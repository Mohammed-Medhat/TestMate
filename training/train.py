"""
train.py — QLoRA fine-tuning of Qwen on the TestMate APR dataset.
Saves the final LoRA adapter to models/adapter/ (= ADAPTER_PATH in config.py).
"""
import sys
import os

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch
import numpy as np
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, TrainingArguments,
    Trainer, DataCollatorForLanguageModeling, BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from config import (
    MODEL_NAME, CACHE_DIR, OUTPUT_DIR,
    MAX_LENGTH, BATCH_SIZE, GRAD_ACCUMULATION, LEARNING_RATE, NUM_EPOCHS,
    TRAIN_FILE, TEST_FILE
)
# Fixed import path based on the updated factory
from dataset_factory import get_tokenized_dataset, build_and_save_datasets

# Allow loading numpy random state from checkpoints
torch.serialization.add_safe_globals([
    np.ndarray, np._core.multiarray._reconstruct, np.dtype,
    np.dtypes.UInt32DType,
    np.random._pickle.__generator_ctor,
    np.random._pickle.__bit_generator_ctor,
])

def run_training():
    # ── Auto-build dataset if not present ────────────────────────────
    if not os.path.exists(TRAIN_FILE) or not os.path.exists(TEST_FILE):
        print("📊 Dataset not found — building now...")
        build_and_save_datasets()

    print("🚀 Loading Tokeniser...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
    tokenizer.pad_token = tokenizer.eos_token

    print("🚀 Loading Tokenised Dataset...")
    tokenized_ds = get_tokenized_dataset(tokenizer)

    print("🚀 Loading Base Model (4-bit QLoRA)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=False,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        cache_dir=CACHE_DIR,
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    print("🚀 Applying LoRA Adapter...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    steps_per_epoch = len(tokenized_ds["train"]) // (BATCH_SIZE * GRAD_ACCUMULATION)
    warmup_steps = max(50, int(steps_per_epoch * NUM_EPOCHS * 0.03))
    print(f"📈 warmup_steps = {warmup_steps}")

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_EPOCHS,
        fp16=True,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        weight_decay=0.01,
        warmup_steps=warmup_steps,
        save_steps=100,
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        load_best_model_at_end=True,
        report_to="none",
        # ── Uncomment below to push directly to HuggingFace ──
        # push_to_hub=True,
        # hub_model_id="YOUR_USERNAME/testmate-adapter",
        # hub_strategy="checkpoint",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["test"],
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    trainer.train()
    print("✅ Training complete. Saving final adapter...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

if __name__ == "__main__":
    run_training()