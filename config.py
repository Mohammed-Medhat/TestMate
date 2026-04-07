import torch
from pathlib import Path

# Project root 
BASE_DIR = Path(__file__).parent

# ── Model paths ───────────────────────────────────────────────────────
MODEL_NAME   = str(BASE_DIR / "models" / "Qwen_Model")   # base model weights
CACHE_DIR    = str(BASE_DIR / "models" / "cache")        # HuggingFace cache
ADAPTER_PATH = str(BASE_DIR / "models" / "adapter")      # LoRA adapter 
# ── Training output ───────────────────────────────────────────────────
OUTPUT_DIR   = str(BASE_DIR / "models" / "adapter")

# ── Data paths ────────────────────────────────────────────────────────
TRAIN_FILE  = str(BASE_DIR / "data" / "train_prepared.jsonl")
TEST_FILE   = str(BASE_DIR / "data" / "test_prepared.jsonl")

# ── Training hyperparameters ──────────────────────────────────────────
MAX_LENGTH        = 2048
BATCH_SIZE        = 2  
GRAD_ACCUMULATION = 16
LEARNING_RATE     = 2e-4
NUM_EPOCHS        = 3