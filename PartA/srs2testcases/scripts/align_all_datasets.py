#!/usr/bin/env python
# scripts/align_all_datasets.py
"""
Batch script to align all PURE_Annotated CSV files (train, valid, test).

This script processes all three PURE dataset splits and generates aligned JSON files.
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from preprocessing.align_pure_datasets import (
    load_pure_annotated,
    align_datasets,
    save_aligned_dataset
)

import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("align_all")


def main():
    # Define paths
    base_dir = Path(__file__).parent.parent.parent
    pure_annotated_dir = base_dir / "PURE_annotated"
    output_dir = base_dir / "srs2testcases" / "data" / "aligned"
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define dataset splits
    datasets = {
        'train': 'PURE_train.csv',
        'valid': 'PURE_valid.csv',
        'test': 'PURE_test.csv'
    }
    
    results = {}
    
    for split_name, csv_filename in datasets.items():
        csv_path = pure_annotated_dir / csv_filename
        output_path = output_dir / f"pure_{split_name}_aligned.json"
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {split_name} split")
        logger.info(f"{'='*60}")
        
        if not csv_path.exists():
            logger.warning(f"CSV file not found: {csv_path}")
            continue
        
        try:
            # Load annotated data
            annotated_data = load_pure_annotated(str(csv_path))
            
            # Align datasets
            aligned_data = align_datasets(annotated_data)
            
            # Save results
            save_aligned_dataset(str(output_path), aligned_data)
            
            results[split_name] = {
                'total': len(aligned_data),
                'requirements': sum(1 for x in aligned_data if x['label'] == 1),
                'non_requirements': sum(1 for x in aligned_data if x['label'] == 0),
                'output_path': str(output_path)
            }
            
        except Exception as e:
            logger.error(f"Error processing {split_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Print final summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    
    for split_name, stats in results.items():
        print(f"\n{split_name.upper()} Split:")
        print(f"  Total items: {stats['total']}")
        print(f"  Requirements: {stats['requirements']}")
        print(f"  Non-requirements: {stats['non_requirements']}")
        print(f"  Output: {stats['output_path']}")
    
    print("\n" + "="*60)
    print("All datasets aligned successfully!")
    print("="*60)


if __name__ == "__main__":
    main()
