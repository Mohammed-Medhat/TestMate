"""
Kaggle Ablation — BES GATE DISABLED
=====================================
Full RAG + self-correction loop, but with the BES quality gate skipped.
Measures the value of the BES (Bug Exposure Score) gate: does enforcing
minimum BES quality on generated tests improve final pass rate and mutation
score, or does it waste token budget on unnecessary retries?

Components ON:
  ✓ Layer 1 (docs FAISS)
  ✓ Layer 2 (knowledge graph)
  ✓ Layer 2 (vector / semantic search)
  ✓ RAG memory
  ✓ Self-correction loop (max_retries=2)
Components OFF:
  ✗ BES acceptance gate (skip_bes_gate=True)
  ✗ Plan mode
  ✗ LoRA

Dataset: PartB/eval_lite/testgen_eval_files/ (ALL files).
Compare against ablation_full_with_loop to isolate BES gate contribution.
"""
from _ablation_common import AblationConfig, run_ablation, run_mutation_pass

config = AblationConfig(
    variant                     = "bes_disabled",
    enable_layer1_docs          = True,
    enable_layer2_graph         = True,
    enable_layer2_vector        = True,
    enable_rag_memory           = True,
    enable_self_correction_loop = True,
    enable_plan_mode            = False,
    enable_lora                 = False,
    skip_bes_gate               = True,    # ← gate OFF
    max_retries                 = 2,
    sample_size                 = None,
    mutation_workers            = 2,
)

if __name__ == "__main__":
    run_ablation(config, output_path="results/ablation_bes_disabled.json")
    run_mutation_pass(
        input_json  = "results/ablation_bes_disabled.json",
        output_json = "results/ablation_bes_disabled_with_mut.json",
        workers     = 2,
    )
