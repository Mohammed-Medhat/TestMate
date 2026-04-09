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
from training.dataset_factory import get_tokenized_dataset

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
        from training.dataset_factory import build_and_save_datasets
        build_and_save_datasets()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, cache_dir=CACHE_DIR)
    tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        cache_dir=CACHE_DIR,
    )
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    tokenized_ds = get_tokenized_dataset(tokenizer)

    # Estimate warmup steps as ~3% of total training steps
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
        weight_decay=0.01,          # prevents overfitting on repeated patterns
        warmup_steps=warmup_steps,  # avoids large gradient updates early in training
        save_steps=100,
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        load_best_model_at_end=True,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["test"],
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    trainer.train()
    print("✅ Training complete!")

    # Saves adapter_config.json + weights into OUTPUT_DIR (= models/adapter/)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"✅ Adapter saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    run_training()