import os
import torch
from dataset_factory import build_and_save_datasets
from train import run_training

def main():
    # 1. Generate the augmented QuixBugs + Synthetic dataset
    print("🛠️ Generating production datasets...")
    build_and_save_datasets(total_synthetic=6000)

    # 2. Check if dataset files were created successfully
    if os.path.exists("./data/train_prepared.jsonl"):
        print("✅ Dataset generation successful.")
    else:
        print("❌ Dataset generation failed. Check your paths.")
        return

    # 3. Start the fine-tuning process
    print("🎯 Starting TestMate Training Pipeline...")
    try:
        run_training()
    except Exception as e:
        print(f"⚠️ Training interrupted: {e}")

if __name__ == "__main__":
    main()