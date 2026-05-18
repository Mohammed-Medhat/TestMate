import subprocess
import json
import os
import time
import re

FILES_DIR = "testgen_eval_files"
RESULTS_FILE = "testgeneval_results.json"
MAIN_PY = r"D:\TestMate\TestMate\PartB\testgen\main.py"

results = []

files = [f for f in os.listdir(FILES_DIR) if f.endswith(".py")]
print(f"Running TestMate on {len(files)} files...\n")

for filename in sorted(files):
    filepath = os.path.join(os.path.abspath(FILES_DIR), filename)
    print(f"\n{'='*50}")
    print(f"Target: {filename}")
    print(f"{'='*50}")
    
    start = time.time()
    
    result = subprocess.run(
        ["python", "-X", "utf8", MAIN_PY, "--target", filepath],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=1800  # 30 min max per file
    )
    
    elapsed = round(time.time() - start, 1)
    output = result.stdout + result.stderr
    
    print("OUTPUT SNIPPET:")
    print(output[-1000:])
    
    metrics = {
        "file": filename,
        "runtime_minutes": round(elapsed / 60, 1),
        "success": result.returncode == 0,
        "targets_covered": 0,
        "targets_total": 0,
        "line_coverage": 0.0,
        "branch_coverage": 0.0,
        "mutation_score": 0.0,
        "composite_score": 0.0,
        "tests_generated": 0,
    }
    
    m = re.search(r"Chunking complete: (\d+)/(\d+)", output)
    if m:
        metrics["targets_covered"] = int(m.group(1))
        metrics["targets_total"]   = int(m.group(2))
    
    m = re.search(r"composite quality:\s*(\d+)/100", output)
    if m:
        metrics["composite_score"] = float(m.group(1))

    m = re.search(r"Branch cov:\s*(\d+\.?\d*)/100", output)
    if m:
        metrics["branch_coverage"] = float(m.group(1))

    m = re.search(r"Mutation:\s*(\d+\.?\d*)/100", output)
    if m:
        metrics["mutation_score"] = float(m.group(1))

    m = re.search(r"Assertion:\s*(\d+\.?\d*)/100", output)
    if m:
        metrics["assertion_score"] = float(m.group(1))

    # Line coverage comes from the coverage raw line
    m = re.search(r"line_coverage.*?(\d+\.?\d*)", output)
    if not m:
        # Try the final composite breakdown
        m = re.search(r"Line cov:\s*(\d+\.?\d*)/100", output)
    if m:
        metrics["line_coverage"] = float(m.group(1))
    
    m = re.search(r"Tests: (\d+)", output)
    if m:
        metrics["tests_generated"] = int(m.group(1))
    
    if "ABSOLUTE SUCCESS" in output:
        metrics["absolute_success"] = True
    
    results.append(metrics)
    
    print(f"\nResult for {filename}:")
    print(f"  Targets:   {metrics['targets_covered']}/{metrics['targets_total']}")
    print(f"  Line cov:  {metrics['line_coverage']}%")
    print(f"  Branch:    {metrics['branch_coverage']}%")
    print(f"  Mutation:  {metrics['mutation_score']}%")
    print(f"  Composite: {metrics['composite_score']}/100")
    print(f"  Runtime:   {metrics['runtime_minutes']} min")
    
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)

print(f"\n{'='*60}")
print("FINAL RESULTS TABLE")
print(f"{'='*60}")
print(f"{'File':<30} {'Pass%':<8} {'Line':<8} {'Branch':<8} {'Mut':<8} {'Comp':<8}")
print("-" * 60)

for r in results:
    if r["targets_total"] > 0:
        pass_rate = round(r["targets_covered"] / r["targets_total"] * 100, 1)
    else:
        pass_rate = 0
    print(f"{r['file'][:28]:<30} {pass_rate:<8} "
          f"{r['line_coverage']:<8} {r['branch_coverage']:<8} "
          f"{r['mutation_score']:<8} {r['composite_score']:<8}")

if len(results) > 0:
    avg_pass     = sum(r["targets_covered"]/max(r["targets_total"],1)*100 for r in results) / len(results)
    avg_line     = sum(r["line_coverage"]    for r in results) / len(results)
    avg_branch   = sum(r["branch_coverage"]  for r in results) / len(results)
    avg_mutation = sum(r["mutation_score"]   for r in results) / len(results)
    avg_comp     = sum(r["composite_score"]  for r in results) / len(results)

    print("-" * 60)
    print(f"{'AVERAGE':<30} {avg_pass:<8.1f} {avg_line:<8.1f} "
          f"{avg_branch:<8.1f} {avg_mutation:<8.1f} {avg_comp:<8.1f}")
    print(f"\nCompare against TestGenEval GPT-4o baseline:")
    print(f"  GPT-4o coverage:  35.2%  (yours: {avg_line:.1f}%)")
    print(f"  GPT-4o mutation:  18.8%  (yours: {avg_mutation:.1f}%)")
