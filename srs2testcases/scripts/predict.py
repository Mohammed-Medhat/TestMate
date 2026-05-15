#!/usr/bin/env python
# scripts/predict.py
"""
Use the trained Requirement Classifier to predict labels for new text.
Usage:
    py scripts/predict.py "The system shall display the login screen."
    py scripts/predict.py --file path/to/text.txt
"""

import sys
import torch
import argparse
from pathlib import Path
from transformers import BertTokenizer, BertForSequenceClassification

def predict(text, model, tokenizer, device):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        pred_label = torch.argmax(probs, dim=-1).item()
        confidence = probs[0][pred_label].item()
        
    return pred_label, confidence

def main():
    parser = argparse.ArgumentParser(description="Predict if text is a Requirement (1) or Not (0)")
    parser.add_argument("text", nargs="?", help="Text to classify (if not using --file)")
    parser.add_argument("--file", help="Path to text file to classify line by line")
    args = parser.parse_args()

    if not args.text and not args.file:
        print("Error: Please provide text or a file path.")
        parser.print_help()
        return

    # Load Model
    base_dir = Path(__file__).parent.parent
    # Try loading from main folder first, then checkpoint
    model_path = base_dir / "models" / "bert-classifier"
    
    if not model_path.exists():
        print(f"Error: Model folder not found at {model_path}")
        return

    # Check for config.json in main folder
    if not (model_path / "config.json").exists():
        # Check subdirectories (checkpoints)
        subdirs = [d for d in model_path.iterdir() if d.is_dir() and "checkpoint" in d.name]
        if subdirs:
            # Sort by name (usually timestamp or step count) -> take latest
            # step count is usually at end.
            # Assuming format: checkpoint-123
            try:
                subdirs.sort(key=lambda x: int(x.name.split("-")[-1]))
                model_path = subdirs[-1]
                print(f"Found checkpoint: {model_path.name}")
            except:
                pass # fallback to main path or whatever

    print(f"Loading model from {model_path}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        tokenizer = BertTokenizer.from_pretrained(model_path)
        model = BertForSequenceClassification.from_pretrained(model_path).to(device)
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        print("Tip: Make sure you have 'config.json', 'pytorch_model.bin', etc.")
        return

    model.eval()
    print("Model loaded successfully.\n")

    if args.text:
        label, conf = predict(args.text, model, tokenizer, device)
        status = "REQUIREMENT" if label == 1 else "NON-REQUIREMENT"
        print(f"Text: \"{args.text}\"")
        print(f"Prediction: {status} ({conf:.2%})")

    if args.file:
        file_path = Path(args.file)
        if not file_path.exists():
            print(f"File not found: {file_path}")
            return
            
        print(f"Processing file: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                label, conf = predict(line, model, tokenizer, device)
                status = "REQ" if label == 1 else "NOT"
                print(f"[{status} {conf:.0%}] {line[:60]}..." if len(line) > 60 else f"[{status} {conf:.0%}] {line}")

if __name__ == "__main__":
    main()
