"""
Kaggle Ablation — PLAN MODE, no self-correction loop
=====================================================
Tests Qwen2.5-Coder-7B-Instruct (4-bit NF4, loaded from HuggingFace, no LoRA)
with plan-then-generate but WITHOUT the self-correction retry loop.

Isolates the contribution of plan mode independently from the loop.

Components ON:
  ✓ Layer 1 (docs FAISS)
  ✓ Layer 2 (knowledge graph)
  ✓ Layer 2 (vector / semantic search)
  ✓ RAG memory
  ✓ Plan mode
Components OFF:
  ✗ Self-correction loop
  ✗ LoRA

Dataset: PartB/eval_lite/testgen_eval_files/ (ALL files).
Compare against ablation_full_with_loop to isolate plan mode contribution.
"""
from _ablation_common import AblationConfig, run_ablation, run_mutation_pass

config = AblationConfig(
    variant                     = "plan_only",
    enable_layer1_docs          = True,
    enable_layer2_graph         = True,
    enable_layer2_vector        = True,
    enable_rag_memory           = True,
    enable_self_correction_loop = False,   # ← OFF: single-shot after plan
    enable_plan_mode            = True,    # ← ON
    enable_lora                 = False,
    max_retries                 = 2,
    sample_size                 = None,
    mutation_workers            = 2,
)

if __name__ == "__main__":
    run_ablation(config, output_path="results/ablation_plan_only.json")
    run_mutation_pass(
        input_json  = "results/ablation_plan_only.json",
        output_json = "results/ablation_plan_only_with_mut.json",
        workers     = 2,
    )
