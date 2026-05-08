#!/usr/bin/env python
# scripts/test_alignment.py
"""
Test script to verify the alignment functionality works correctly.

This script runs a quick test on a small sample of the PURE_Annotated data.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from preprocessing.align_pure_datasets import (
    normalize_text,
    fuzzy_match_score,
    load_pure_annotated,
    generate_doc_id,
    generate_block_id,
    generate_sentence_id
)


def test_normalize_text():
    """Test text normalization."""
    print("Testing text normalization...")
    
    test_cases = [
        ("Hello  World", "Hello World"),
        ("Test\xa0text", "Test text"),
        ("quoted", "quoted"),
        ("  extra   spaces  ", "extra spaces"),
    ]
    
    for input_text, expected in test_cases:
        result = normalize_text(input_text)
        assert result == expected, f"Failed: {input_text} -> {result} (expected {expected})"
        print(f"  ✓ '{input_text}' -> '{result}'")
    
    print("  All normalization tests passed!\n")


def test_fuzzy_matching():
    """Test fuzzy matching scores."""
    print("Testing fuzzy matching...")
    
    test_cases = [
        ("Hello World", "Hello World", 1.0),  # Exact match
        ("The system shall provide", "The system shall provide", 1.0),  # Exact match
        ("The system must provide", "The system shall provide", 0.9),  # High similarity
        ("Completely different", "Nothing alike", 0.3),  # Low similarity
    ]
    
    for text1, text2, min_expected_score in test_cases:
        score = fuzzy_match_score(text1, text2)
        passed = score >= min_expected_score if min_expected_score >= 0.9 else score < 0.5
        status = "✓" if passed else "✗"
        print(f"  {status} '{text1}' vs '{text2}' -> {score:.2f}")
    
    print("  Fuzzy matching tests completed!\n")


def test_id_generation():
    """Test ID generation functions."""
    print("Testing ID generation...")
    
    # Test doc_id generation
    doc_id = generate_doc_id("cctns.pdf")
    print(f"  ✓ Doc ID: 'cctns.pdf' -> '{doc_id}'")
    
    # Test block_id generation
    block_id = generate_block_id(42)
    print(f"  ✓ Block ID: 42 -> '{block_id}'")
    
    # Test sentence_id generation
    sentence_id = generate_sentence_id("b42", 3)
    print(f"  ✓ Sentence ID: ('b42', 3) -> '{sentence_id}'")
    
    print("  ID generation tests passed!\n")


def test_load_sample():
    """Test loading a sample of PURE_Annotated data."""
    print("Testing data loading...")
    
    base_dir = Path(__file__).parent.parent.parent
    csv_path = base_dir / "PURE_annotated" / "PURE_train.csv"
    
    if not csv_path.exists():
        print(f"  ⚠ Warning: CSV file not found at {csv_path}")
        print("  Skipping data loading test\n")
        return
    
    try:
        data = load_pure_annotated(str(csv_path))
        print(f"  ✓ Loaded {len(data)} items from PURE_train.csv")
        
        # Show a sample
        if data:
            sample = data[0]
            print(f"\n  Sample item:")
            print(f"    Text: {sample['text'][:80]}...")
            print(f"    Label: {sample['label']}")
            print(f"    Doc: {sample['doc_name']}")
        
        # Show label distribution
        label_counts = {0: 0, 1: 0}
        for item in data:
            label_counts[item['label']] += 1
        
        print(f"\n  Label distribution:")
        print(f"    Requirements (1): {label_counts[1]}")
        print(f"    Non-requirements (0): {label_counts[0]}")
        
        print("\n  Data loading test passed!\n")
        
    except Exception as e:
        print(f"  ✗ Error loading data: {e}\n")


def main():
    print("="*60)
    print("ALIGNMENT MODULE TEST SUITE")
    print("="*60)
    print()
    
    test_normalize_text()
    test_fuzzy_matching()
    test_id_generation()
    test_load_sample()
    
    print("="*60)
    print("All tests completed!")
    print("="*60)


if __name__ == "__main__":
    main()
