import torch
from pathlib import Path

BASE_DIR = Path(__file__).parent

MODEL_NAME   = str(BASE_DIR / "Qwen_Model")
CACHE_DIR    = str(BASE_DIR / "model_cache")
OUTPUT_DIR   = str(BASE_DIR / "qwen-testmate-adapter")
ADAPTER_PATH = str(BASE_DIR / "Final")   

TRAIN_FILE  = str(BASE_DIR / "data" / "train_prepared.jsonl")
TEST_FILE   = str(BASE_DIR / "data" / "test_prepared.jsonl")

MAX_LENGTH        = 2048
BATCH_SIZE        = 1
GRAD_ACCUMULATION = 16
LEARNING_RATE     = 2e-4
NUM_EPOCHS        = 3