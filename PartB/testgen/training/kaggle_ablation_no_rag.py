"""
Kaggle Ablation #1 — NO RAG (raw LLM baseline, no plan, no loop)
=================================================================
Tests Qwen2.5-Coder-7B-Instruct (4-bit NF4, loaded from HuggingFace, no LoRA)
with NOTHING enabled — only the source code + AST structure (parsing, not RAG).

Components ON: (none)
Components OFF: Layer 1 · Layer 2 graph · Layer 2 vector · RAG memory · Plan · Loop

Dataset: PartB/eval_lite/testgen_eval_files/ (ALL 102 files).
Purpose: pure baseline. All other ablations should beat this.
"""
from _ablation_common import AblationConfig, run_ablation, run_mutation_pass

config = AblationConfig(
    variant                     = "no_rag",
    enable_layer1_docs          = False,
    enable_layer2_graph         = False,
    enable_layer2_vector        = False,
    enable_rag_memory           = False,
    enable_self_correction_loop = False,
    enable_plan_mode            = False,
    sample_size                 = None,
)

if __name__ == "__main__":
    run_ablation(config, output_path="results/ablation_no_rag.json")
    run_mutation_pass(
        input_json  = "results/ablation_no_rag.json",
        output_json = "results/ablation_no_rag_with_mut.json",
    )
