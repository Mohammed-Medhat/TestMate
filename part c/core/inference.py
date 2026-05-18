import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from config import MODEL_NAME, CACHE_DIR, ADAPTER_PATH

def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4"
    )

    print("Loading Base Model...")
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map={"": 0},
        cache_dir=CACHE_DIR,
        trust_remote_code=True
    )

    print(f"Loading Peft Adapter from: {ADAPTER_PATH}")
    model = PeftModel.from_pretrained(base, ADAPTER_PATH)
    return model, tokenizer