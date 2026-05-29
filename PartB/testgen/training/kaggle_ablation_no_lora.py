"""
Kaggle Ablation — NO LoRA (base model only)
============================================
Tests Qwen2.5-Coder-7B-Instruct (4-bit NF4) with NO LoRA adapter.
Measures the lift that the LoRA fine-tuning provides.

Components ON:
  ✓ Layer 1 (docs FAISS)
  ✓ Layer 2 (knowledge graph)
  ✓ Layer 2 (vector / semantic search)
  ✓ RAG memory
  ✓ Self-correction loop (max_retries=2)
Components OFF:
  ✗ LoRA adapter
  ✗ Plan mode

Dataset: PartB/eval_lite/testgen_eval_files/ (ALL files).
Compare against ablation_full_with_loop to isolate LoRA contribution.
"""
from _ablation_common import AblationConfig, run_ablation, run_mutation_pass

config = AblationConfig(
    variant                     = "no_lora",
    enable_layer1_docs          = True,
    enable_layer2_graph         = True,
    enable_layer2_vector        = True,
    enable_rag_memory           = True,
    enable_self_correction_loop = True,
    enable_plan_mode            = False,
    enable_lora                 = False,   # ← OFF: base model only
    max_retries                 = 2,
    sample_size                 = None,
    mutation_workers            = 2,
)

if __name__ == "__main__":
    run_ablation(config, output_path="results/ablation_no_lora.json")
    run_mutation_pass(
        input_json  = "results/ablation_no_lora.json",
        output_json = "results/ablation_no_lora_with_mut.json",
        workers     = 2,
    )
