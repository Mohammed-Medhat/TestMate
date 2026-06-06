"""
Local ablation — TestEval track (recognized headline benchmark).
================================================================================
Runs the full TestMate stack on TestEval (Wang et al. 2024), the standard LLM
test-generation benchmark: 210 LeetCode Python programs. The headline metric is
the COVERAGE a generated test suite achieves on each canonical solution
(line/branch), directly comparable to the TestEval leaderboard.

Clean: the model is trained on MBPP only (decontaminated), never on TestEval.
TestEval programs are self-contained → single host env, no venv clusters /
identity restoration. The dataset is fetched + cached on first use.

Zero model download (local snapshot).

Run:
    cd PartB/testgen/training
    python local_ablation_testeval.py            # smoke (sample_size=5)
    # for the full headline: edit sample_size=None OR use
    #   python run_comparisons.py --dataset testeval --sample 0
"""
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")        # model offline; TestEval fetched via https
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

print(f"[testeval] MODEL_DIR -> {MODEL_DIR}  ({'exists' if os.path.isdir(MODEL_DIR) else 'MISSING'})")
print(f"[testeval] LoRA      -> {LORA_PATH}  ({'CLEAN retrain' if _IS_CLEAN else 'OLD adapter (fallback)'})")

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
    dataset                     = "testeval",
    sample_size                 = 5,    # smoke; bump to None for all 210
    max_file_seconds            = 300.0,
)

if __name__ == "__main__":
    run_ablation(config, output_path="results/local_testeval_testmate.json")
    run_mutation_pass(
        input_json  = "results/local_testeval_testmate.json",
        output_json = "results/local_testeval_testmate_with_mut.json",
    )
