#!/usr/bin/env python
# scripts/train_classifier.py
"""
Train Requirement Classifier (BERT)

This script trains a BERT-based binary classifier to identify requirements in SRS documents.
It uses the aligned PURE dataset (`pure_train_aligned.json`) for training.
"""

import os
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from datasets import Dataset, DatasetDict

def main():
    print("="*60)
    print("Training Requirement Classifier (BERT)")
    print("="*60)

    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Data
    print("\nLoading data...")
    # Define paths
    base_dir = Path(__file__).parent.parent
    data_path = base_dir / "data" / "aligned" / "pure_train_aligned_full.json"
    
    if not data_path.exists():
        print(f"Error: Data file not found at {data_path}")
        return

    # Load data
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Convert to DataFrame for easier handling
    df = pd.DataFrame(data)
    print(f"Total samples: {len(df)}")
    print("Label distribution:")
    print(df["label"].value_counts())

    # Prepare for Hugging Face Dataset
    # We strictly need 'text' and 'label' columns
    dataset_df = df[["text", "label"]].copy()

    # Split into Train (80%) and Test (20%)
    train_df, test_df = train_test_split(dataset_df, test_size=0.2, random_state=42, stratify=dataset_df["label"])

    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)

    dataset = DatasetDict({
        "train": train_dataset,
        "test": test_dataset
    })
    
    print("\nDataset split:")
    print(dataset)

    # 2. Preprocessing (Tokenization)
    print("\nInitializing Tokenizer...")
    model_name = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_name)

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

    print("Tokenizing dataset...")
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # 3. Model Initialization
    print("\nInitializing Model...")
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)

    # 4. Training Configuration
    def compute_metrics(p):
        pred, labels = p
        pred = np.argmax(pred, axis=1)

        accuracy = accuracy_score(labels, pred)
        recall = recall_score(labels, pred)
        precision = precision_score(labels, pred)
        f1 = f1_score(labels, pred)

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    output_dir = base_dir / "models" / "bert-classifier"
    
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        load_best_model_at_end=True,
        report_to="none",  # Disable wandb/mlflow logging
        dataloader_pin_memory=False,
        logging_steps=10,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],

        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # 5. Training Loop
    print("\nStarting Training...")
    trainer.train()

    # 6. Evaluation & Saving
    print("\nEvaluating Model...")
    metrics = trainer.evaluate()
    print("Final Metrics:", metrics)

    # Save Model
    print(f"\nSaving model to {output_dir}...")
    trainer.save_model(str(output_dir))
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)

if __name__ == "__main__":
    main()
