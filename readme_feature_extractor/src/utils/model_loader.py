import os
from llama_cpp import Llama

def load_model(model_path="models/qwen2.5-coder-7b-instruct-q4_k_m.gguf"):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")

    # Set GPU layers: -1 for NVIDIA GPU, 0 for Intel/CPU
    # This makes the code work on both Colab and your Local PC
    gpu_layers = -1 if os.environ.get("USE_GPU", "0") == "1" else 0
    
    print(f"Loading model on {'GPU' if gpu_layers == -1 else 'CPU'} Mode...")
    
    llm = Llama(
        model_path=model_path,
        n_ctx=8192,      # Optimized context for 16GB RAM
        n_threads=8,     # Utilizing i7 cores
        n_gpu_layers=gpu_layers,
        verbose=False
    )
    return llm