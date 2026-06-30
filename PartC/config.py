import torch
from pathlib import Path

# Project root 
BASE_DIR = Path(__file__).parent

# ── Model paths ───────────────────────────────────────────────────────
MODEL_NAME   = r"D:\donwloader\Qwen2.5-Coder-7B-Instruct"  # shared with PartB
CACHE_DIR    = str(BASE_DIR / "models" / "cache")        # HuggingFace cache
ADAPTER_PATH = str(BASE_DIR / "models" / "adapter")      # LoRA adapter 
# ── Training output ───────────────────────────────────────────────────
OUTPUT_DIR   = str(BASE_DIR / "models" / "adapter")

# ── Data paths ────────────────────────────────────────────────────────
TRAIN_FILE  = str(BASE_DIR / "data" / "train_prepared.jsonl")
TEST_FILE   = str(BASE_DIR / "data" / "test_prepared.jsonl")

# ── Training hyperparameters ──────────────────────────────────────────
MAX_LENGTH        = 2048
BATCH_SIZE        = 1  
GRAD_ACCUMULATION = 16
LEARNING_RATE     = 2e-4
NUM_EPOCHS        = 3

# ── Inference / repair hyperparameters ───────────────────────────────
# Temperature schedule per attempt (index 0 = attempt 1, etc.)
# Low temp first for high-confidence fix; higher temp on retries for diversity.
REPAIR_TEMPERATURES = [0.2, 0.5, 0.8]
MAX_REPAIR_ATTEMPTS = 3
TOP_P               = 0.95
MAX_NEW_TOKENS      = 1024