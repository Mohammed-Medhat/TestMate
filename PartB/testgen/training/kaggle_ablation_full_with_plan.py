"""
Kaggle Ablation #6 — FULL RAG + PLAN MODE (no loop)
====================================================
Tests Qwen2.5-Coder-7B-Instruct (4-bit NF4, loaded from HuggingFace, no LoRA)
with all RAG layers AND explicit plan-first generation.

Flow per file:
  1. LLM call #1 — generate a brief test plan (5-7 bullets)
  2. LLM call #2 — generate the test, with the plan injected into the prompt
  3. Measure plan_adherence (how well the test follows the plan)

Components ON:
  ✓ Layer 1 (docs FAISS)
  ✓ Layer 2 (knowledge graph)
  ✓ Layer 2 (vector / semantic search)
  ✓ RAG memory
  ✓ Plan mode (explicit two-call generation)
Components OFF:
  ✗ Self-correction loop

Dataset: PartB/eval_lite/testgen_eval_files/ (ALL 102 files).

Purpose: directly comparable to `kaggle_ablation_full.py` (which is identical
but with plan mode OFF). The delta in metrics tells you what plan mode adds.
"""
from _ablation_common import AblationConfig, run_ablation, run_mutation_pass

config = AblationConfig(
    variant                     = "full_with_plan",
    enable_layer1_docs          = True,
    enable_layer2_graph         = True,
    enable_layer2_vector        = True,
    enable_rag_memory           = True,
    enable_self_correction_loop = False,
    enable_plan_mode            = True,    # ← ON
    sample_size                 = None,
)

if __name__ == "__main__":
    run_ablation(config, output_path="results/ablation_full_with_plan.json")
    run_mutation_pass(
        input_json  = "results/ablation_full_with_plan.json",
        output_json = "results/ablation_full_with_plan_with_mut.json",
    )
