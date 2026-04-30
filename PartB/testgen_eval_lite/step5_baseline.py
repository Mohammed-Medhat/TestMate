import sys
import os
import json
import time

sys.path.append(r"D:\TestMate\TestMate\PartB\testgen")
from main import load_model, generate_test, run_pytest_with_coverage, extract_test_code, run_mutation_testing

FILES_DIR = "testgen_eval_files"
RESULTS_FILE = "testgeneval_baseline_results.json"

BASELINE_PROMPT = """Write pytest unit tests for this Python code.
Import everything needed. Use concrete assertions.

```python
{source_code}
```

Return only the test code."""

def main():
    model, tokenizer = load_model()
    
    files = [f for f in os.listdir(FILES_DIR) if f.endswith(".py")]
    print(f"Running Baseline on {len(files)} files...\n")
    
    results = []
    os.makedirs("baseline_tests", exist_ok=True)
    
    for filename in sorted(files):
        filepath = os.path.join(os.path.abspath(FILES_DIR), filename)
        test_filepath = os.path.join(os.path.abspath("baseline_tests"), "test_" + filename)
        
        with open(filepath, "r", encoding="utf-8") as f:
            source_code = f.read()
            
        prompt = BASELINE_PROMPT.format(source_code=source_code)
        messages = [{"role": "user", "content": prompt}]
        
        start = time.time()
        print(f"\n{'='*50}\nTarget: {filename} (Baseline)\n{'='*50}")
        
        raw_output = generate_test(model, tokenizer, messages, max_tokens=1536)
        test_code = extract_test_code(raw_output)
        
        # Write test code
        with open(test_filepath, "w", encoding="utf-8") as f:
            f.write(test_code)
            
        # Run pytest
        try:
            passed, output, line_cov, branch_cov, exc_lines = run_pytest_with_coverage(
                test_filepath, filepath, need_lines=False
            )
        except Exception as e:
            print(f"Pytest failed: {e}")
            passed, line_cov, branch_cov = False, 0.0, 0.0
        
        # Run mutation
        try:
            mut_killed, mut_feedback = run_mutation_testing(filepath, test_filepath)
            import re
            m = re.search(r'(\d+)%', mut_feedback)
            mut_score = float(m.group(1)) if m else (100.0 if mut_killed else 0.0)
        except Exception as e:
            print(f"Mutation failed: {e}")
            mut_score = 0.0
            
        elapsed = round(time.time() - start, 1)
        
        metrics = {
            "file": filename,
            "runtime_minutes": round(elapsed / 60, 1),
            "success": passed,
            "line_coverage": line_cov,
            "branch_coverage": branch_cov,
            "mutation_score": mut_score,
        }
        results.append(metrics)
        
        print(f"  Line cov:  {line_cov}%")
        print(f"  Branch:    {branch_cov}%")
        print(f"  Mutation:  {mut_score}%")
        print(f"  Runtime:   {metrics['runtime_minutes']} min")
        
        with open(RESULTS_FILE, "w") as f:
            json.dump(results, f, indent=2)
            
    # Print summary table
    print(f"\n{'='*60}")
    print("FINAL BASELINE RESULTS TABLE")
    print(f"{'='*60}")
    print(f"{'File':<30} {'Line':<8} {'Branch':<8} {'Mut':<8}")
    print("-" * 60)

    for r in results:
        print(f"{r['file'][:28]:<30} "
              f"{r['line_coverage']:<8} {r['branch_coverage']:<8} "
              f"{r['mutation_score']:<8}")

    avg_line     = sum(r["line_coverage"]    for r in results) / len(results) if results else 0
    avg_branch   = sum(r["branch_coverage"]  for r in results) / len(results) if results else 0
    avg_mutation = sum(r["mutation_score"]   for r in results) / len(results) if results else 0

    print("-" * 60)
    print(f"{'AVERAGE':<30} {avg_line:<8.1f} "
          f"{avg_branch:<8.1f} {avg_mutation:<8.1f}")

if __name__ == "__main__":
    main()
