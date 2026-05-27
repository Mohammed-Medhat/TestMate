"""
Kaggle Ablation — TESTMATE without BES gate
===========================================
Identical to `kaggle_ablation_testmate.py` except `skip_bes_gate=True`.

Purpose: isolate the BES quality gate's contribution. The delta between
`testmate` and `testmate_no_bes` measures exactly how much BES improves
final test quality (the harness compares the same generation pipeline,
LoRA, RAG layers — only the gate differs).

Dataset: PartB/eval_lite/testgen_eval_files/ (ALL files).

Usage on Kaggle:

  # CELL 1: deps
  !pip install -q transformers bitsandbytes accelerate peft \\
                  sentence-transformers faiss-cpu networkx \\
                  pytest pytest-cov coverage

  # CELL 2: clone and cd
  !git clone --depth 1 https://github.com/Mohammed-Medhat/TestMate.git \\
      /kaggle/working/TestMate
  %cd /kaggle/working/TestMate/PartB/testgen/training

  # CELL 3: run
  !python kaggle_ablation_testmate_no_bes.py
"""
import os
from _ablation_common import AblationConfig, run_ablation, run_mutation_pass

_LORA_PATH_CANDIDATES = [
    "/kaggle/input/datasets/mohammedmedhat08/testgen-model/final",
    "/kaggle/input/datasets/mohammed8medhat/testgen-model/final",
]
LORA_PATH = next((p for p in _LORA_PATH_CANDIDATES if os.path.isdir(p)),
                 _LORA_PATH_CANDIDATES[0])
print(f"[testmate_no_bes] LoRA path resolved -> {LORA_PATH}"
      f"  ({'exists' if os.path.isdir(LORA_PATH) else 'MISSING — check uploads'})")

config = AblationConfig(
    variant                     = "testmate_no_bes",
    enable_layer1_docs          = True,
    enable_layer2_graph         = True,
    enable_layer2_vector        = True,
    enable_rag_memory           = True,
    enable_self_correction_loop = True,
    enable_plan_mode            = False,
    enable_lora                 = True,
    lora_path                   = LORA_PATH,
    sample_size                 = None,
    skip_bes_gate               = True,   # the ONLY difference vs testmate
)

if __name__ == "__main__":
    run_ablation(config, output_path="results/ablation_testmate_no_bes.json")
    run_mutation_pass(
        input_json  = "results/ablation_testmate_no_bes.json",
        output_json = "results/ablation_testmate_no_bes_with_mut.json",
    )
