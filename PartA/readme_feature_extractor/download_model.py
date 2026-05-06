from huggingface_hub import hf_hub_download
import os

def download_qwen_coder():
    repo_id = "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF"
    filename = "qwen2.5-coder-7b-instruct-q4_k_m.gguf"
    
    print(f"Starting download of {filename}...")
    os.makedirs("models", exist_ok=True)
    
    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir="models"
    )
    print(f"Model downloaded to: {path}")

if __name__ == "__main__":
    download_qwen_coder()