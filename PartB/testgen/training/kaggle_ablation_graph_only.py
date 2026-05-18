"""
Kaggle Ablation #2 — GRAPH RAG ONLY (no vector RAG, no plan, no loop)
======================================================================
Tests Qwen2.5-Coder-7B-Instruct (4-bit NF4, loaded from HuggingFace, no LoRA)
with knowledge graph traversal ONLY.

Components ON:
  ✓ Layer 2 (knowledge graph)
Components OFF: Layer 1 · Layer 2 vector · RAG memory · Plan · Loop

Dataset: PartB/eval_lite/testgen_eval_files/ (ALL 102 files).
Purpose: isolate the contribution of graph traversal on its own.
"""
from _ablation_common import AblationConfig, run_ablation, run_mutation_pass

config = AblationConfig(
    variant                     = "graph_only",
    enable_layer1_docs          = False,
    enable_layer2_graph         = True,
    enable_layer2_vector        = False,
    enable_rag_memory           = False,
    enable_self_correction_loop = False,
    enable_plan_mode            = False,
    sample_size                 = None,
)

if __name__ == "__main__":
    run_ablation(config, output_path="results/ablation_graph_only.json")
    run_mutation_pass(
        input_json  = "results/ablation_graph_only.json",
        output_json = "results/ablation_graph_only_with_mut.json",
    )
