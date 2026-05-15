import os
from huggingface_hub import hf_hub_download

def setup_models():
    # 1. Create the local directory for models if it doesn't exist
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    
    # 2. Define the models to be used in the TestMate Demo
    models_to_download = [
        {
            "repo_id": "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF",
            "filename": "qwen2.5-coder-7b-instruct-q4_k_m.gguf"
        },
        {
            "repo_id": "MaziyarPanahi/Meta-Llama-3-8B-Instruct-GGUF",
            "filename": "Meta-Llama-3-8B-Instruct.Q4_K_M.gguf"
        }
    ]
    
    print("--- TestMate Offline Setup: Checking Models ---")
    
    for model in models_to_download:
        dest_path = os.path.join(model_dir, model["filename"])
        
        if not os.path.exists(dest_path):
            print(f"Downloading {model['filename']} for offline use...")
            hf_hub_download(
                repo_id=model["repo_id"],
                filename=model["filename"],
                local_dir=model_dir
            )
            print(f"Successfully saved to {dest_path}")
        else:
            print(f"Model {model['filename']} already exists locally.")

if __name__ == "__main__":
    setup_models()
    print("--- Environment Ready for Offline Demo ---")