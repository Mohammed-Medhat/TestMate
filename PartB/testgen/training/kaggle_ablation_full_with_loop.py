"""
Kaggle Ablation #5 — FULL RAG + AUTONOMOUS LOOP (no plan)
==========================================================
Tests Qwen2.5-Coder-7B-Instruct (4-bit NF4, loaded from HuggingFace, no LoRA)
with EVERYTHING enabled including the self-correction loop.
This represents how Part B normally runs.

Components ON:
  ✓ Layer 1 (docs FAISS)
  ✓ Layer 2 (knowledge graph)
  ✓ Layer 2 (vector / semantic search)
  ✓ RAG memory
  ✓ Self-correction loop (max_retries=3)
Components OFF:
  ✗ Plan mode

Dataset: PartB/eval_lite/testgen_eval_files/ (ALL 102 files).
Note: longest variant. Plan for a dedicated Kaggle session.
"""
from _ablation_common import AblationConfig, run_ablation, run_mutation_pass

config = AblationConfig(
    variant                     = "full_with_loop",
    enable_layer1_docs          = True,
    enable_layer2_graph         = True,
    enable_layer2_vector        = True,
    enable_rag_memory           = True,
    enable_self_correction_loop = True,    # ← ON
    enable_plan_mode            = False,
    max_retries                 = 3,
    sample_size                 = None,
)

if __name__ == "__main__":
    run_ablation(config, output_path="results/ablation_full_with_loop.json")
    run_mutation_pass(
        input_json  = "results/ablation_full_with_loop.json",
        output_json = "results/ablation_full_with_loop_with_mut.json",
    )
