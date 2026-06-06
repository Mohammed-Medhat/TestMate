"""
Local ablation — HumanEval track (clean standard benchmark).
================================================================================
Runs the full TestMate stack on HumanEval (164 self-contained problems). This
is the clean, contamination-free headline number: the model was NOT trained on
HumanEval (training = MBPP only, decontaminated).

HumanEval functions are self-contained, so:
  - no venv clusters / identity restoration (single host env)
  - the RAG layers retrieve little (no surrounding codebase) — expected; the
    RAG contribution is shown on the realistic testgenevallite track instead.

Zero model download (local snapshot). HumanEval is tiny and already cached.

Run:
    cd PartB/testgen/training
    python local_ablation_humaneval.py
"""
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")        # model + cached datasets offline
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from _ablation_common import AblationConfig, run_ablation, run_mutation_pass

# Local model snapshot → zero download
_MODEL_CANDIDATES = [
    r"D:\donwloader\Qwen2.5-Coder-7B-Instruct",
    r"D:\TestMate\huggingface_cache\hub\models--Qwen--Qwen2.5-Coder-7B",
]
MODEL_DIR = next((p for p in _MODEL_CANDIDATES if os.path.isdir(p)), _MODEL_CANDIDATES[0])

# Prefer the CLEAN retrained adapter; fall back to the current one so the
# plumbing is testable before the clean retrain lands.
_models = Path(__file__).parent.parent.parent / "models"
_LORA_CANDIDATES = [
    str(_models / "graphrag_lora_clean" / "final"),   # clean retrain (preferred)
    str(_models / "graphrag_lora" / "final"),         # current (fallback)
]
LORA_PATH = next((p for p in _LORA_CANDIDATES if os.path.isdir(p)), _LORA_CANDIDATES[-1])
_IS_CLEAN = "graphrag_lora_clean" in LORA_PATH

print(f"[humaneval] MODEL_DIR -> {MODEL_DIR}  ({'exists' if os.path.isdir(MODEL_DIR) else 'MISSING'})")
print(f"[humaneval] LoRA      -> {LORA_PATH}  ({'CLEAN retrain' if _IS_CLEAN else 'OLD adapter (fallback)'})")

config = AblationConfig(
    variant                     = "testmate",
    enable_layer1_docs          = True,
    enable_layer2_graph         = True,
    enable_layer2_vector        = True,
    enable_rag_memory           = True,
    enable_self_correction_loop = True,
    enable_plan_mode            = False,
    enable_lora                 = True,
    lora_path                   = LORA_PATH,
    model_id                    = MODEL_DIR,
    dataset                     = "humaneval",
    sample_size                 = 5,    # smoke; bump to None for all 164
    max_file_seconds            = 300.0,  # HumanEval funcs are small/fast
)

if __name__ == "__main__":
    run_ablation(config, output_path="results/local_humaneval_testmate.json")
    run_mutation_pass(
        input_json  = "results/local_humaneval_testmate.json",
        output_json = "results/local_humaneval_testmate_with_mut.json",
    )
