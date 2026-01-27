#!/usr/bin/env python
# scripts/test_pipeline.py
"""
Quick test script for the run_pipeline.py functionality.
Tests the pipeline on a sample document.
"""

import sys
import json
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from run_pipeline import RequirementExtractionPipeline


def test_pipeline():
    """Test the pipeline with a sample document."""
    
    print("="*60)
    print("PIPELINE TEST")
    print("="*60)
    
    # Setup paths
    base_dir = Path(__file__).parent.parent.parent
    sample_doc = base_dir / "req" / "2011 - opensg 0.1.docx"
    aligned_dataset = base_dir / "srs2testcases" / "data" / "aligned" / "pure_train_aligned.json"
    output_file = base_dir / "srs2testcases" / "results" / "test_pipeline_output.json"
    
    # Check if sample document exists
    if not sample_doc.exists():
        print(f"\n⚠ Sample document not found: {sample_doc}")
        print("Please ensure the document exists or update the path.")
        return
    
    # Check if aligned dataset exists
    aligned_path = str(aligned_dataset) if aligned_dataset.exists() else None
    if not aligned_path:
        print(f"\n⚠ Aligned dataset not found: {aligned_dataset}")
        print("Running without aligned dataset (unlabeled mode)")
    
    try:
        # Initialize pipeline
        print(f"\nInitializing pipeline...")
        pipeline = RequirementExtractionPipeline(aligned_path)
        
        # Process document
        print(f"\nProcessing: {sample_doc.name}")
        stats = pipeline.process_document(
            str(sample_doc),
            str(output_file),
            fuzzy_threshold=0.85
        )
        
        # Print results
        print("\n" + "="*60)
        print("TEST RESULTS")
        print("="*60)
        print(f"Document: {stats['document']}")
        print(f"Total sentences: {stats['total_sentences']}")
        print(f"Requirements: {stats['requirements']}")
        print(f"Non-requirements: {stats['non_requirements']}")
        print(f"Unlabeled: {stats['unlabeled']}")
        print(f"\nMatch types:")
        print(f"  Exact: {stats['match_types']['exact']}")
        print(f"  Fuzzy: {stats['match_types']['fuzzy']}")
        print(f"  Unmatched: {stats['match_types']['unmatched']}")
        print(f"\nOutput: {stats['output_path']}")
        
        # Load and show sample results
        with open(output_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        if results:
            print(f"\nSample output (first item):")
            sample = results[0]
            print(f"  Text: {sample['text'][:80]}...")
            print(f"  Label: {sample['label']}")
            print(f"  Match type: {sample['match_type']}")
            print(f"  Score: {sample['score']:.3f}")
        
        print("\n" + "="*60)
        print("✓ Pipeline test completed successfully!")
        print("="*60)
        
    except Exception as e:
        print(f"\n✗ Error during pipeline test: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_pipeline()
