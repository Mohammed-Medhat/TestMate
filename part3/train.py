import torch
import numpy as np
import os
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, TrainingArguments, 
    Trainer, DataCollatorForLanguageModeling, BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from config import *
from dataset_factory import get_tokenized_dataset

torch.serialization.add_safe_globals([
    np.ndarray, np._core.multiarray._reconstruct, np.dtype,
    np.dtypes.UInt32DType, np.random._pickle.__generator_ctor,
    np.random._pickle.__bit_generator_ctor
])

def run_training():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, cache_dir=CACHE_DIR)
    tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4"
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, quantization_config=bnb_config, device_map="auto", 
        trust_remote_code=True, cache_dir=CACHE_DIR
    )
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=8, lora_alpha=32, target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05, task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)

    tokenized_ds = get_tokenized_dataset(tokenizer)

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_EPOCHS,
        fp16=True,
        save_steps=100,
        logging_steps=10,
        report_to="none"
    )

    trainer = Trainer(
        model=model, args=args,
        train_dataset=tokenized_ds["train"],
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    )

    if os.path.exists(RESUME_PATH):
        trainer.train(resume_from_checkpoint=RESUME_PATH)
    else:
        trainer.train()

    model.save_pretrained(f"{OUTPUT_DIR}/final")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/final")

if __name__ == "__main__":
    run_training()