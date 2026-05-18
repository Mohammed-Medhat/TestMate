"""
Kaggle Ablation #3 — NO GRAPH RAG (vector RAG only, no plan, no loop)
======================================================================
Tests Qwen2.5-Coder-7B-Instruct (4-bit NF4, loaded from HuggingFace, no LoRA)
with VECTOR RAG only (Layer 1 docs + Layer 2 vector + RAG memory).
No knowledge graph traversal, no plan mode, no self-correction loop.

Components ON:
  ✓ Layer 1 (docs FAISS)
  ✓ Layer 2 (vector / semantic search)
  ✓ RAG memory
Components OFF:
  ✗ Layer 2 (knowledge graph)
  ✗ Plan mode
  ✗ Self-correction loop

Dataset: PartB/eval_lite/testgen_eval_files/ (ALL 102 files).
Purpose: isolate the contribution of graph traversal vs vector-only retrieval.
"""
from _ablation_common import AblationConfig, run_ablation, run_mutation_pass

config = AblationConfig(
    variant                     = "no_graph",
    enable_layer1_docs          = True,
    enable_layer2_graph         = False,   # ← off
    enable_layer2_vector        = True,
    enable_rag_memory           = True,
    enable_self_correction_loop = False,
    enable_plan_mode            = False,
    sample_size                 = None,
)

if __name__ == "__main__":
    run_ablation(config, output_path="results/ablation_no_graph.json")
    run_mutation_pass(
        input_json  = "results/ablation_no_graph.json",
        output_json = "results/ablation_no_graph_with_mut.json",
    )
