import torch

MODEL_NAME = "/home/c/my_apr_project/Qwen_Model"
CACHE_DIR = "./model_cache"
OUTPUT_DIR = "./qwen-testmate-adapter"
ADAPTER_PATH = "/home/c/my_apr_project/final"

TRAIN_FILE = "./data/train_prepared.jsonl"
TEST_FILE  = "./data/test_prepared.jsonl"

RESUME_PATH = "./checkpoints/checkpoint-300"


MAX_LENGTH = 2048
BATCH_SIZE = 1
GRAD_ACCUMULATION = 16
LEARNING_RATE = 2e-4
NUM_EPOCHS = 3