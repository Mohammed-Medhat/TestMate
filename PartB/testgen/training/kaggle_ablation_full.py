"""
Kaggle Ablation #4 — FULL RAG (no plan, no loop)
=================================================
Tests Qwen2.5-Coder-7B-Instruct (4-bit NF4, loaded from HuggingFace, no LoRA)
with ALL RAG layers enabled but NO plan mode and NO self-correction loop.

Components ON:
  ✓ Layer 1 (docs FAISS)
  ✓ Layer 2 (knowledge graph)
  ✓ Layer 2 (vector / semantic search)
  ✓ RAG memory (SQLite store)
Components OFF:
  ✗ Plan mode
  ✗ Self-correction loop

Dataset: PartB/eval_lite/testgen_eval_files/ (ALL 102 files).

Usage on Kaggle (one notebook session, two cells):

  # CELL 1: deps + clone
  !pip install -q transformers bitsandbytes accelerate \\
                  sentence-transformers faiss-cpu networkx \\
                  pytest pytest-cov coverage mutmut
  !git clone https://github.com/<user>/TestMate /kaggle/working/TestMate

  # CELL 2: run both phases
  %cd /kaggle/working/TestMate/PartB/testgen/training
  !python kaggle_ablation_full.py
"""
from _ablation_common import AblationConfig, run_ablation, run_mutation_pass

config = AblationConfig(
    variant                     = "full",
    enable_layer1_docs          = True,
    enable_layer2_graph         = True,
    enable_layer2_vector        = True,
    enable_rag_memory           = True,
    enable_self_correction_loop = False,
    enable_plan_mode            = False,
    sample_size                 = None,    # ALL files
)

if __name__ == "__main__":
    # Phase 1: generation
    run_ablation(config, output_path="results/ablation_full.json")
    # Phase 2: mutation
    run_mutation_pass(
        input_json  = "results/ablation_full.json",
        output_json = "results/ablation_full_with_mut.json",
    )
