import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from config import MODEL_NAME, CACHE_DIR, ADAPTER_PATH


def _check_adapter_weights(adapter_path: str) -> bool:
    """Return True if adapter weights file is present."""
    p = Path(adapter_path)
    return (p / "adapter_model.safetensors").exists() or (p / "adapter_model.bin").exists()


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

    # Use as_posix() so PEFT doesn't mistake a Windows backslash path for a HF repo ID
    adapter_path_posix = Path(ADAPTER_PATH).resolve().as_posix()

    if not _check_adapter_weights(ADAPTER_PATH):
        print(f"WARNING: adapter weights missing at {ADAPTER_PATH}")
        print("         Run: python PartC/training/train.py  to generate the adapter.")
        print("         Returning base model without LoRA.")
        return base, tokenizer

    print(f"Loading Peft Adapter from: {ADAPTER_PATH}")
    model = PeftModel.from_pretrained(base, adapter_path_posix)
    return model, tokenizer