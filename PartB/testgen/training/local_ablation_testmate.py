"""
Local ablation — TestMate variant, zero model download, version-matched venvs.
================================================================================
Runs the full TestMate stack (Layer 1/2/3 RAG + self-correction loop + LoRA) on
the local machine instead of Kaggle.

Two things make this cheap and accurate:
  1. **Zero model download.** Points at the complete on-disk Qwen snapshot at
     MODEL_DIR (the default HF cache copy is incomplete). HF_HUB_OFFLINE +
     TRANSFORMERS_OFFLINE guarantee no network access for the model.
  2. **Version-matched test execution.** Each eval file's import gate + pytest
     run inside a cluster venv pinned to the file's framework version (created
     by setup_version_venvs.py). Files with no matching venv fall back to host
     python automatically. This is what lifts pass rate above the ~7% ceiling
     caused by host packages being newer than every eval file's target.

Setup (one-time):
    cd PartB/testgen/training
    python setup_version_venvs.py --only django42     # validate one cluster first
    # later, build the rest:  python setup_version_venvs.py

Run:
    python local_ablation_testmate.py
"""
import os

# ── Offline guards MUST be set before transformers/HF is imported anywhere ──
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import sys
from pathlib import Path

# Make sibling modules importable when run from any CWD.
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))  # PartB/testgen for main.py

from _ablation_common import AblationConfig, run_ablation, run_mutation_pass

# ── Complete local model snapshot (all 4 shards present) → zero download ──
_MODEL_CANDIDATES = [
    r"D:\donwloader\Qwen2.5-Coder-7B-Instruct",
    r"D:\TestMate\huggingface_cache\hub\models--Qwen--Qwen2.5-Coder-7B",
]
MODEL_DIR = next((p for p in _MODEL_CANDIDATES if os.path.isdir(p)), _MODEL_CANDIDATES[0])

# ── Local LoRA adapter (already in the repo) ──
_LORA_CANDIDATES = [
    str(Path(__file__).parent.parent.parent / "models" / "graphrag_lora" / "final"),
    r"D:\TestMate\TestMate\PartB\models\graphrag_lora\final",
]
LORA_PATH = next((p for p in _LORA_CANDIDATES if os.path.isdir(p)), _LORA_CANDIDATES[0])

print(f"[local] MODEL_DIR -> {MODEL_DIR}  ({'exists' if os.path.isdir(MODEL_DIR) else 'MISSING'})")
print(f"[local] LoRA      -> {LORA_PATH}  ({'exists' if os.path.isdir(LORA_PATH) else 'MISSING'})")
print(f"[local] offline   -> HF_HUB_OFFLINE={os.environ.get('HF_HUB_OFFLINE')} "
      f"TRANSFORMERS_OFFLINE={os.environ.get('TRANSFORMERS_OFFLINE')}")

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
    model_id                    = MODEL_DIR,   # ← local path, loads offline
    # Start with 5 for a smoke test (confirm no downloads + venv routing),
    # then bump to 30, then None for all files.
    sample_size                 = 5,
    max_file_seconds            = 900.0,
)

if __name__ == "__main__":
    run_ablation(config, output_path="results/local_ablation_testmate.json")
    run_mutation_pass(
        input_json  = "results/local_ablation_testmate.json",
        output_json = "results/local_ablation_testmate_with_mut.json",
    )
