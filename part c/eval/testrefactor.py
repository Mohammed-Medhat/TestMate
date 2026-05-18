#!/usr/bin/env python3
"""
test_dataset_structure.py — Quick validation script
Tests that the generated dataset has proper structure for pytest.
"""

import json
import tempfile
import subprocess
import sys
from pathlib import Path


def test_single_sample(sample: dict, verbose: bool = True) -> bool:
    """
    Test a single sample to ensure pytest can find and run tests.
    Returns True if the test structure is valid.
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"Testing sample: {sample.get('id', 'Unknown')}")
        print(f"{'='*70}")
    
    # Check required fields
    required = ['buggy_code', 'tests']
    for field in required:
        if field not in sample:
            print(f"❌ Missing field: {field}")
            return False
    
    # Create temp directory
    tmpdir = tempfile.mkdtemp(prefix="test_validation_")
    
    try:
        # Write files
        solution_file = Path(tmpdir) / "solution.py"
        test_file = Path(tmpdir) / "test_solution.py"
        
        solution_file.write_text(sample['buggy_code'], encoding='utf-8')
        test_file.write_text(sample['tests'], encoding='utf-8')
        
        if verbose:
            print(f"\n📁 Created test directory: {tmpdir}")
            print(f"📝 solution.py: {len(sample['buggy_code'])} chars")
            print(f"📝 test_solution.py: {len(sample['tests'])} chars")
            
            print(f"\n🔍 First 10 lines of test file:")
            for i, line in enumerate(sample['tests'].splitlines()[:10], 1):
                print(f"   {i:2}: {line}")
        
        # Try to collect tests (don't run them, just check if pytest finds them)
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "test_solution.py", "--collect-only", "-q"],
            capture_output=True,
            text=True,
            cwd=tmpdir,
            timeout=10
        )
        
        output = result.stdout + result.stderr
        
        if verbose:
            print(f"\n🔬 Pytest collection output:")
            print(f"   {output}")
        
        import re
        # Works for both old ("1 selected") and new ("1 collected") pytest formats
        match = re.search(r"(\d+) test[s]? collected", output)
        if match:
            print(f"✅ Successfully collected {match.group(1)} test(s)")
            return True

        # Explicit failures
        if "ERROR" in output or "ImportError" in output or "SyntaxError" in output:
            print(f"❌ Pytest collection failed:\n   {output}")
            return False

        if "no tests ran" in output.lower() or "0 test" in output:
            print(f"❌ No tests collected — check for 'def test_*' functions\n   {output}")
            return False

        # Fallback: test node path visible and no error = likely fine
        if "::" in output and "error" not in output.lower():
            match2 = re.search(r"(\d+) test", output)
            count = match2.group(1) if match2 else "?"
            print(f"✅ Successfully collected {count} test(s) (fallback detection)")
            return True

        print(f"⚠️  Unclear pytest output:\n   {output}")
        return False
        
    except Exception as e:
        print(f"❌ Exception during test: {e}")
        return False
    
    finally:
        # Cleanup
        import shutil
        try:
            shutil.rmtree(tmpdir)
        except:
            pass


def main():
    """Main validation routine."""
    dataset_path = "humaneval_eval_dataset.json"
    
    print("═" * 70)
    print("🧪 DATASET STRUCTURE VALIDATION")
    print("═" * 70)
    
    # Check if dataset exists
    if not Path(dataset_path).exists():
        print(f"\n❌ Dataset not found: {dataset_path}")
        print(f"   Please run: python fetch_humaneval_fixed.py")
        sys.exit(1)
    
    # Load dataset
    print(f"\n📊 Loading dataset: {dataset_path}")
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    print(f"✅ Loaded {len(dataset)} samples")
    
    # Test first sample in detail
    print(f"\n{'─'*70}")
    print(f"DETAILED TEST: First Sample")
    print(f"{'─'*70}")
    
    if dataset:
        success = test_single_sample(dataset[0], verbose=True)
        
        if success:
            print(f"\n✅ First sample structure is VALID")
        else:
            print(f"\n❌ First sample structure has ISSUES")
            print(f"\n💡 Recommendation:")
            print(f"   1. Re-run: python fetch_humaneval_fixed.py")
            print(f"   2. Check that test file has 'def test_*' functions")
            print(f"   3. Verify imports: 'from solution import ...'")
            sys.exit(1)
    
    # Quick validation of remaining samples
    print(f"\n{'─'*70}")
    print(f"QUICK VALIDATION: All Samples")
    print(f"{'─'*70}")
    
    valid_count = 0
    invalid_samples = []
    
    for i, sample in enumerate(dataset):
        # Quick checks without running pytest
        has_test_func = 'def test_' in sample.get('tests', '')
        has_import = 'from solution import' in sample.get('tests', '')
        has_buggy = len(sample.get('buggy_code', '')) > 0
        
        is_valid = has_test_func and has_import and has_buggy
        
        if is_valid:
            valid_count += 1
            status = "✅"
        else:
            invalid_samples.append((i, sample.get('id', f'Sample_{i}')))
            status = "❌"
        
        if i < 5 or not is_valid:  # Show first 5 and all invalid
            print(f"  [{i:3}] {status} {sample.get('id', 'Unknown'):<20} "
                  f"test_func={has_test_func} import={has_import} code={has_buggy}")
    
    print(f"\n{'─'*70}")
    print(f"📊 SUMMARY")
    print(f"{'─'*70}")
    print(f"Total samples: {len(dataset)}")
    print(f"Valid samples: {valid_count}")
    print(f"Invalid samples: {len(invalid_samples)}")
    
    if invalid_samples:
        print(f"\n⚠️  Invalid samples:")
        for idx, sample_id in invalid_samples[:10]:  # Show first 10
            print(f"   [{idx}] {sample_id}")
        if len(invalid_samples) > 10:
            print(f"   ... and {len(invalid_samples) - 10} more")
        print(f"\n💡 These samples will be skipped during evaluation")
    
    if valid_count == len(dataset):
        print(f"\n🎉 ALL SAMPLES ARE VALID!")
        print(f"\n✅ Ready to run evaluation:")
        print(f"   python apr_evaluator_fixed.py --mode finetuned --limit 5")
    else:
        print(f"\n⚠️  Some samples have issues, but evaluation can proceed")
        print(f"   Valid samples will be automatically selected")
    
    print(f"\n{'═'*70}\n")


if __name__ == "__main__":
    main()