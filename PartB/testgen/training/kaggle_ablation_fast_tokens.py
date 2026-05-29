"""
Kaggle Ablation — FAST TOKENS (max_new_tokens=512)
====================================================
Full RAG + self-correction loop but with half the token budget (512 vs 1024).
Measures the speed/quality tradeoff: does halving generation tokens significantly
hurt pass rate and mutation score, or do most tests fit in 512 tokens anyway?

Note: calculate_max_tokens() in main.py already scales by method complexity,
so the effective cap is min(calculate_max_tokens(...), 512).

Components ON:
  ✓ Layer 1 (docs FAISS)
  ✓ Layer 2 (knowledge graph)
  ✓ Layer 2 (vector / semantic search)
  ✓ RAG memory
  ✓ Self-correction loop (max_retries=2)
Components OFF:
  ✗ Plan mode
  ✗ LoRA

Dataset: PartB/eval_lite/testgen_eval_files/ (ALL files).
Compare against ablation_full_with_loop to measure token-budget impact.
"""
from _ablation_common import AblationConfig, run_ablation, run_mutation_pass

config = AblationConfig(
    variant                     = "fast_tokens",
    enable_layer1_docs          = True,
    enable_layer2_graph         = True,
    enable_layer2_vector        = True,
    enable_rag_memory           = True,
    enable_self_correction_loop = True,
    enable_plan_mode            = False,
    enable_lora                 = False,
    max_new_tokens              = 512,     # ← halved token budget
    max_retries                 = 2,
    sample_size                 = None,
    mutation_workers            = 2,
)

if __name__ == "__main__":
    run_ablation(config, output_path="results/ablation_fast_tokens.json")
    run_mutation_pass(
        input_json  = "results/ablation_fast_tokens.json",
        output_json = "results/ablation_fast_tokens_with_mut.json",
        workers     = 2,
    )
