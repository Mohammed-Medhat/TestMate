"""
Kaggle Ablation — TESTMATE (production system)
==============================================
The full TestMate stack:
  ✓ Qwen2.5-Coder-7B-Instruct (4-bit NF4)
  ✓ LoRA fine-tuned adapter
  ✓ Layer 1 (docs FAISS)
  ✓ Layer 2 (knowledge graph, built on the fly via KGCompassGraphRAG)
  ✓ Layer 2 (vector / semantic search)
  ✓ RAG memory (SQLite store)
  ✓ Self-correction loop

This matches the production pipeline used by:
  - PartB/testgen/main.py  (autonomous_loop)
  - unified_app/orchestrator.py

Dataset: PartB/eval_lite/testgen_eval_files/ (ALL 102 files).

Usage on Kaggle (one notebook session):

  # CELL 1: deps
  !pip install -q transformers bitsandbytes accelerate peft \\
                  sentence-transformers faiss-cpu networkx \\
                  pytest pytest-cov coverage mutmut

  # CELL 2: clone and cd
  !git clone --depth 1 https://github.com/Mohammed-Medhat/TestMate.git \\
      /kaggle/working/TestMate
  %cd /kaggle/working/TestMate/PartB/testgen/training

  # CELL 3: run
  !python kaggle_ablation_testmate.py
"""
import os
from _ablation_common import AblationConfig, run_ablation, run_mutation_pass

# LoRA adapter uploaded as the Kaggle dataset `testgen-model`.
# Multiple paths are tried in order so the same script works across both
# of your Kaggle accounts. Add more candidates here if you upload the
# adapter under another account.
_LORA_PATH_CANDIDATES = [
    "/kaggle/input/datasets/mohammedmedhat08/testgen-model/final",
    "/kaggle/input/datasets/mohammed8medhat/testgen-model/final",
]
LORA_PATH = next((p for p in _LORA_PATH_CANDIDATES if os.path.isdir(p)),
                 _LORA_PATH_CANDIDATES[0])
print(f"[testmate] LoRA path resolved -> {LORA_PATH}"
      f"  ({'exists' if os.path.isdir(LORA_PATH) else 'MISSING — check uploads'})")

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
    # 102 files × 900s cap = 25.5h worst-case → does not fit in Kaggle's 9h T4.
    # Sample 30 files for the validation run; bump up after fixes are verified.
    sample_size                 = 30,
    max_file_seconds            = 900.0,
)

if __name__ == "__main__":
    run_ablation(config, output_path="results/ablation_testmate.json")
    run_mutation_pass(
        input_json  = "results/ablation_testmate.json",
        output_json = "results/ablation_testmate_with_mut.json",
    )
