#!/usr/bin/env python
# scripts/verify_alignment.py
"""
Verification script to check the quality and format of aligned datasets.

This script validates the aligned JSON files and generates a detailed report.
"""

import json
import sys
from pathlib import Path
from collections import Counter


def verify_aligned_dataset(json_path: str) -> dict:
    """
    Verify an aligned dataset and return statistics.
    
    Args:
        json_path: Path to aligned JSON file
        
    Returns:
        Dictionary with verification statistics
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    stats = {
        'total_items': len(data),
        'label_distribution': Counter(),
        'match_type_distribution': Counter(),
        'score_stats': {'min': 1.0, 'max': 0.0, 'avg': 0.0},
        'doc_count': len(set(item['doc_id'] for item in data)),
        'missing_fields': [],
        'sample_items': []
    }
    
    required_fields = ['text', 'label', 'doc_id', 'block_id', 'sentence_id', 
                      'annotated_text', 'match_type', 'score']
    
    scores = []
    
    for idx, item in enumerate(data):
        # Check required fields
        missing = [f for f in required_fields if f not in item]
        if missing:
            stats['missing_fields'].append((idx, missing))
        
        # Collect statistics
        stats['label_distribution'][item.get('label')] += 1
        stats['match_type_distribution'][item.get('match_type')] += 1
        
        score = item.get('score', 0.0)
        scores.append(score)
        stats['score_stats']['min'] = min(stats['score_stats']['min'], score)
        stats['score_stats']['max'] = max(stats['score_stats']['max'], score)
        
        # Collect samples
        if idx < 3:
            stats['sample_items'].append(item)
    
    if scores:
        stats['score_stats']['avg'] = sum(scores) / len(scores)
    
    return stats


def print_verification_report(split_name: str, stats: dict):
    """Print a formatted verification report."""
    print(f"\n{'='*60}")
    print(f"{split_name.upper()} SPLIT VERIFICATION")
    print(f"{'='*60}")
    
    print(f"\nTotal Items: {stats['total_items']}")
    print(f"Unique Documents: {stats['doc_count']}")
    
    print(f"\nLabel Distribution:")
    for label, count in sorted(stats['label_distribution'].items()):
        label_name = "Requirements" if label == 1 else "Non-requirements"
        percentage = (count / stats['total_items'] * 100) if stats['total_items'] > 0 else 0
        print(f"  {label_name} ({label}): {count} ({percentage:.1f}%)")
    
    print(f"\nMatch Type Distribution:")
    for match_type, count in sorted(stats['match_type_distribution'].items()):
        percentage = (count / stats['total_items'] * 100) if stats['total_items'] > 0 else 0
        print(f"  {match_type}: {count} ({percentage:.1f}%)")
    
    print(f"\nScore Statistics:")
    print(f"  Min: {stats['score_stats']['min']:.3f}")
    print(f"  Max: {stats['score_stats']['max']:.3f}")
    print(f"  Avg: {stats['score_stats']['avg']:.3f}")
    
    if stats['missing_fields']:
        print(f"\n⚠ WARNING: Found {len(stats['missing_fields'])} items with missing fields!")
        for idx, missing in stats['missing_fields'][:5]:
            print(f"  Item {idx}: missing {missing}")
    else:
        print(f"\n✓ All items have required fields")
    
    print(f"\nSample Items:")
    for idx, item in enumerate(stats['sample_items']):
        print(f"\n  Sample {idx + 1}:")
        print(f"    Text: {item['text'][:80]}...")
        print(f"    Label: {item['label']}")
        print(f"    Doc ID: {item['doc_id']}")
        print(f"    Block ID: {item['block_id']}")
        print(f"    Sentence ID: {item['sentence_id']}")
        print(f"    Match Type: {item['match_type']}")
        print(f"    Score: {item['score']:.3f}")


def main():
    base_dir = Path(__file__).parent.parent
    aligned_dir = base_dir / "data" / "aligned"
    
    if not aligned_dir.exists():
        print(f"Error: Aligned data directory not found: {aligned_dir}")
        sys.exit(1)
    
    datasets = {
        'train': 'pure_train_aligned.json',
        'valid': 'pure_valid_aligned.json',
        'test': 'pure_test_aligned.json'
    }
    
    print("="*60)
    print("DATASET ALIGNMENT VERIFICATION REPORT")
    print("="*60)
    
    all_stats = {}
    
    for split_name, filename in datasets.items():
        json_path = aligned_dir / filename
        
        if not json_path.exists():
            print(f"\n⚠ Warning: {split_name} split not found at {json_path}")
            continue
        
        try:
            stats = verify_aligned_dataset(str(json_path))
            all_stats[split_name] = stats
            print_verification_report(split_name, stats)
        except Exception as e:
            print(f"\n✗ Error verifying {split_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Overall summary
    print(f"\n{'='*60}")
    print("OVERALL SUMMARY")
    print(f"{'='*60}")
    
    total_items = sum(s['total_items'] for s in all_stats.values())
    total_requirements = sum(s['label_distribution'].get(1, 0) for s in all_stats.values())
    total_non_requirements = sum(s['label_distribution'].get(0, 0) for s in all_stats.values())
    
    print(f"\nTotal Items Across All Splits: {total_items}")
    print(f"Total Requirements: {total_requirements} ({total_requirements/total_items*100:.1f}%)")
    print(f"Total Non-requirements: {total_non_requirements} ({total_non_requirements/total_items*100:.1f}%)")
    
    print(f"\nSplit Sizes:")
    for split_name, stats in all_stats.items():
        print(f"  {split_name}: {stats['total_items']} items")
    
    print(f"\n{'='*60}")
    print("✓ Verification Complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
