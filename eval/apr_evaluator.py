"""
apr_evaluator.py — Enhanced APR Model Evaluation Framework
Fixed version with better debugging and error handling.

Key improvements:
  - Better test file structure detection
  - Verbose error reporting
  - Test execution debugging
  - Sample validation before evaluation
"""

import sys
import os
import ast
import re
import json
import subprocess
import tempfile
import shutil
import argparse
import time
import gc
from pathlib import Path
from collections import defaultdict
from datetime import datetime

# ── Project imports ────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ── Dataset Loading with Validation ───────────────────────────────────

def validate_sample(sample: dict, idx: int) -> bool:
    """Validate that a sample has all required fields."""
    required = ['id', 'buggy_code', 'tests']
    missing = [f for f in required if f not in sample]
    
    if missing:
        print(f"⚠️  Sample {idx} missing fields: {missing}")
        return False
    
    # Check if test file has proper structure
    test_code = sample['tests']
    if 'def test_' not in test_code:
        print(f"⚠️  Sample {idx} ({sample.get('id', '?')}) has no 'def test_' function!")
        print(f"     First 200 chars of test:\n     {test_code[:200]}...")
        return False
    
    if 'from solution import' not in test_code:
        print(f"⚠️  Sample {idx} ({sample.get('id', '?')}) missing 'from solution import' statement!")
        return False
    
    return True


def load_evaluation_dataset(json_path: str, limit: int = None) -> list:
    """Load and validate the HumanEvalFix dataset."""
    if not os.path.exists(json_path):
        print(f"❌ Dataset file not found: {json_path}")
        print(f"⚠️  Please run 'python fetch_humaneval.py' first to generate it.")
        sys.exit(1)
    
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    print(f"📊 Loaded {len(data)} samples from {json_path}")
    
    # Validate all samples
    print("🔍 Validating samples...")
    valid_samples = []
    for i, sample in enumerate(data):
        if validate_sample(sample, i):
            valid_samples.append(sample)
    
    print(f"✅ {len(valid_samples)}/{len(data)} samples are valid")
    
    if len(valid_samples) != len(data):
        print(f"⚠️  {len(data) - len(valid_samples)} samples were skipped due to validation errors")
    
    if limit:
        valid_samples = valid_samples[:limit]
        print(f"✂️  Limiting evaluation to first {limit} samples")
    
    return valid_samples


# ── Model Loading ──────────────────────────────────────────────────────

def load_finetuned_model():
    """Load LoRA fine-tuned TestMate model."""
    try:
        from core.inference import load_model
        model, tokenizer = load_model()
        print("✅ Fine-tuned model loaded")
        return model, tokenizer, "finetuned"
    except Exception as e:
        print(f"❌ Could not load fine-tuned model: {e}")
        raise


def load_base_model():
    """Load raw base Qwen model (no adapter)."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        try:
            from config import MODEL_NAME, CACHE_DIR
        except ImportError:
            MODEL_NAME = "Qwen/Qwen2.5-Coder-7B-Instruct"
            CACHE_DIR  = None

        print(f"📦 Loading base model: {MODEL_NAME}")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_config,
            device_map={"": 0},
            cache_dir=CACHE_DIR,
            trust_remote_code=True,
        )
        print("✅ Base model loaded (no adapter)")
        return model, tokenizer, "base"
    except Exception as e:
        print(f"❌ Could not load base model: {e}")
        raise


def run_model(model, tokenizer, prompt: str, attempt: int = 1, max_new_tokens: int = 1024) -> str:
    """Run model with repair prompt, adjusting temperature by attempt."""
    import torch

    system_msg = (
        "Expert APR agent. Output ONLY the complete fixed Python function(s). "
        "No line numbers, no explanation, no markdown."
    )
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user",   "content": prompt},
    ]

    chat_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(chat_prompt, return_tensors="pt").to("cuda")

    # Dynamic temperature: 0.2 → 0.5 → 0.8
    current_temp = 0.2 if attempt == 1 else (0.5 if attempt == 2 else 0.8)
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": True,
        "temperature": current_temp,
        "top_p": 0.95,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
    }

    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )
    return response.strip()


# ── Code Cleaning & Patching ───────────────────────────────────────────

def extract_clean_code(raw: str) -> str:
    """Strip markdown, prompt echoes, and line numbers from model output."""
    # Remove prompt echo markers
    for marker in ("BUGGY CODE", "FIXED CODE", "ISSUE:", "TRACE:", "SUSPICIOUS"):
        if marker in raw:
            raw = raw[:raw.index(marker)]

    # Extract from markdown code blocks
    match = re.search(r"```(?:python)?(.*?)```", raw, re.DOTALL | re.IGNORECASE)
    if match:
        clean = match.group(1).strip()
    else:
        # Find first function/class definition
        lines = raw.split("\n")
        start = 0
        for i, line in enumerate(lines):
            if line.strip().startswith(("def ", "class ", "@")):
                start = i
                break
        clean = "\n".join(lines[start:]).strip()
    
    # Remove line number prefixes
    clean = re.sub(r"^\s*\d+:\s?", "", clean, flags=re.MULTILINE)
    
    return clean


def apply_ast_patch(original_source: str, model_output: str) -> str | None:
    """
    Replace functions in original_source with versions from model_output.
    Returns patched source, or None if patching fails.
    """
    clean = extract_clean_code(model_output)

    try:
        new_ast = ast.parse(clean)
    except SyntaxError as e:
        print(f"      ⚠️  Model output has syntax error: {e}")
        return None

    new_fns = {n.name: n for n in new_ast.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    if not new_fns:
        print(f"      ⚠️  No functions found in model output")
        return None

    try:
        orig_ast = ast.parse(original_source)
    except SyntaxError:
        # Original is broken — use full model output
        print(f"      ℹ️  Original has syntax error, using full model output")
        return clean

    modified = False
    for i, node in enumerate(orig_ast.body):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in new_fns:
            orig_ast.body[i] = new_fns[node.name]
            modified = True
            print(f"      🔧 Replaced: {node.name}")

    if not modified:
        print(f"      ⚠️  No matching functions to replace")
        return None

    return ast.unparse(orig_ast)


# ── Test Execution with Enhanced Debugging ────────────────────────────

def run_tests_in_sandbox(source_code: str, test_code: str, debug: bool = False) -> dict:
    """
    Write source + tests to temp dir and run pytest.
    Returns { passed: bool, pass_rate: float, output: str, num_passed: int, num_total: int }
    """
    tmpdir = tempfile.mkdtemp(prefix="apr_eval_")
    
    if debug:
        print(f"      📁 Test directory: {tmpdir}")
    
    try:
        # Write files
        solution_file = Path(tmpdir) / "solution.py"
        test_file = Path(tmpdir) / "test_solution.py"
        
        solution_file.write_text(source_code, encoding='utf-8')
        test_file.write_text(test_code, encoding='utf-8')
        
        if debug:
            print(f"      📝 Written solution.py ({len(source_code)} chars)")
            print(f"      📝 Written test_solution.py ({len(test_code)} chars)")
            print(f"      🔍 First 5 lines of test file:")
            for i, line in enumerate(test_code.splitlines()[:5], 1):
                print(f"         {i}: {line}")

        # Run pytest with verbose output
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "test_solution.py", "-v", "--tb=short", "--no-header"],
            capture_output=True,
            text=True,
            cwd=tmpdir,
            timeout=30,
        )

        output = result.stdout + result.stderr
        
        # Parse pytest output for pass/fail counts
        passed_match = re.search(r"(\d+) passed", output)
        failed_match = re.search(r"(\d+) failed", output)
        
        num_passed = int(passed_match.group(1)) if passed_match else 0
        num_failed = int(failed_match.group(1)) if failed_match else 0
        num_total  = num_passed + num_failed
        
        # If no tests were found
        if num_total == 0:
            print(f"      ❌ No tests found! pytest output:")
            print(f"         {output[:500]}")
            return {
                "passed": False,
                "pass_rate": 0.0,
                "num_passed": 0,
                "num_total": 0,
                "output": output,
                "error": "No tests collected"
            }
        
        all_passed = (result.returncode == 0)
        pass_rate = num_passed / num_total if num_total > 0 else 0.0

        return {
            "passed": all_passed,
            "pass_rate": pass_rate,
            "num_passed": num_passed,
            "num_total": num_total,
            "output": output
        }

    except subprocess.TimeoutExpired:
        return {
            "passed": False,
            "pass_rate": 0.0,
            "num_passed": 0,
            "num_total": 0,
            "output": "Test execution timeout (30s)",
            "error": "Timeout"
        }
    except Exception as e:
        return {
            "passed": False,
            "pass_rate": 0.0,
            "num_passed": 0,
            "num_total": 0,
            "output": str(e),
            "error": str(e)
        }
    finally:
        # Clean up temp directory
        try:
            shutil.rmtree(tmpdir)
        except:
            pass


# ── Sample Evaluation ──────────────────────────────────────────────────

def build_repair_prompt(sample: dict, fail_output: str = "", attempt: int = 1) -> str:
    """Build a structured repair prompt for the model."""
    buggy_code = sample['buggy_code']
    description = sample.get('description', 'Fix the buggy code')
    
    prompt = f"ISSUE: {description}\n\n"
    
    if fail_output and attempt > 1:
        # Add failure context for subsequent attempts
        error_lines = [
            line for line in fail_output.splitlines() 
            if any(kw in line for kw in ['FAILED', 'AssertionError', 'Error', 'assert'])
        ][:5]
        
        prompt += f"[ATTEMPT {attempt} - Previous fix FAILED]\n"
        prompt += "Previous error:\n" + "\n".join(error_lines) + "\n\n"
        prompt += "Try a COMPLETELY DIFFERENT approach.\n\n"
    
    prompt += f"BUGGY CODE:\n{buggy_code}\n\n"
    prompt += "Output ONLY the fixed code, no explanations."
    
    return prompt


def evaluate_sample(sample: dict, model, tokenizer, max_attempts: int = 3) -> dict:
    """Evaluate a single sample with multiple repair attempts."""
    result = {
        "id": sample['id'],
        "bug_type": sample.get('bug_type', 'Unknown'),
        "repaired": False,
        "attempts_used": 0,
        "final_pass_rate": 0.0,
        "final_passed": 0,
        "final_total": 0,
        "time_seconds": 0.0
    }

    buggy_code = sample['buggy_code']
    test_code = sample['tests']
    
    # First, check if tests run at all with buggy code
    print(f"   🧪 Running initial test (should fail)...", end=" ")
    initial_test = run_tests_in_sandbox(buggy_code, test_code, debug=False)
    
    if initial_test.get("error") == "No tests collected":
        print(f"❌ TEST STRUCTURE ERROR")
        print(f"      ⚠️  Test file is malformed - no tests found by pytest")
        result["error"] = "Test structure invalid"
        return result
    
    if initial_test["passed"]:
        print(f"⚠️  Already passing! ({initial_test['num_passed']}/{initial_test['num_total']})")
        result["repaired"] = True
        result["final_pass_rate"] = 1.0
        result["final_passed"] = initial_test["num_passed"]
        result["final_total"] = initial_test["num_total"]
        return result
    
    print(f"❌ Failed as expected ({initial_test['num_passed']}/{initial_test['num_total']})")
    
    # Now attempt repairs
    t_start = time.time()
    fail_output = initial_test["output"]
    
    for attempt in range(1, max_attempts + 1):
        print(f"   🔄 Attempt {attempt}/{max_attempts}...", end=" ")
        
        prompt = build_repair_prompt(sample, fail_output, attempt)
        raw_out = run_model(model, tokenizer, prompt, attempt=attempt)
        
        patched = apply_ast_patch(buggy_code, raw_out)
        
        if patched is None:
            print(f"❌ Patch failed")
            continue
        
        test_res = run_tests_in_sandbox(patched, test_code, debug=False)
        result["attempts_used"] = attempt
        
        if test_res.get("error") == "No tests collected":
            print(f"❌ Test error after patch")
            break
        
        if test_res["passed"]:
            print(f"✅ FIXED ({test_res['num_passed']}/{test_res['num_total']})")
            result["repaired"] = True
            result["final_pass_rate"] = test_res["pass_rate"]
            result["final_passed"] = test_res["num_passed"]
            result["final_total"] = test_res["num_total"]
            break
        else:
            print(f"❌ Still failing ({test_res['num_passed']}/{test_res['num_total']})")
            fail_output = test_res["output"]
            result["final_pass_rate"] = test_res["pass_rate"]
            result["final_passed"] = test_res["num_passed"]
            result["final_total"] = test_res["num_total"]

    result["time_seconds"] = round(time.time() - t_start, 2)
    return result


# ── Metrics Computation ────────────────────────────────────────────────

def compute_metrics(results: list) -> dict:
    """Compute overall and per-category metrics."""
    total = len(results)
    repaired = sum(1 for r in results if r["repaired"])
    pass_rates = [r["final_pass_rate"] for r in results]
    attempts = [r["attempts_used"] for r in results if r["attempts_used"] > 0]

    per_category = defaultdict(lambda: {"total": 0, "repaired": 0, "pass_rates": []})
    for r in results:
        cat = r["bug_type"]
        per_category[cat]["total"] += 1
        per_category[cat]["repaired"] += int(r["repaired"])
        per_category[cat]["pass_rates"].append(r["final_pass_rate"])

    cat_summary = {}
    for cat, data in per_category.items():
        cat_summary[cat] = {
            "repair_success_rate": round(data["repaired"] / data["total"], 4) if data["total"] else 0,
            "avg_test_pass_rate": round(sum(data["pass_rates"]) / len(data["pass_rates"]), 4) if data["pass_rates"] else 0,
            "total": data["total"],
            "repaired": data["repaired"],
        }

    return {
        "overall": {
            "total_samples": total,
            "repaired": repaired,
            "repair_success_rate": round(repaired / total, 4) if total else 0,
            "avg_test_pass_rate": round(sum(pass_rates) / len(pass_rates), 4) if pass_rates else 0,
            "avg_attempts": round(sum(attempts) / len(attempts), 2) if attempts else 0,
        },
        "per_category": cat_summary,
    }


# ── Report Generation ──────────────────────────────────────────────────

def print_report(model_label: str, metrics: dict, results: list):
    sep = "═" * 70
    print(f"\n{sep}")
    print(f"  📊 EVALUATION REPORT — {model_label.upper()}")
    print(sep)

    ov = metrics["overall"]
    print(f"\n  Overall Results ({ov['total_samples']} samples):")
    print(f"    Repair Success Rate : {ov['repair_success_rate']*100:.1f}%  ({ov['repaired']}/{ov['total_samples']})")
    print(f"    Avg Test Pass Rate  : {ov['avg_test_pass_rate']*100:.1f}%")
    print(f"    Avg Attempts Used   : {ov['avg_attempts']:.1f}")

    print(f"\n  Per-Category Breakdown:")
    print(f"  {'Category':<25} {'Success':>10} {'Pass%':>8}")
    print(f"  {'-'*45}")
    for cat, data in metrics["per_category"].items():
        print(f"  {cat:<25} {data['repaired']}/{data['total']:>3}      {data['avg_test_pass_rate']*100:>5.1f}%")
    
    print(f"\n{sep}\n")


def print_comparison(ft_metrics: dict, base_metrics: dict):
    sep = "═" * 70
    print(f"\n{sep}")
    print(f"  🆚  COMPARISON: Fine-Tuned vs Base Model")
    print(sep)

    ft_ov = ft_metrics["overall"]
    b_ov = base_metrics["overall"]

    rsr_delta = (ft_ov["repair_success_rate"] - b_ov["repair_success_rate"]) * 100
    tpr_delta = (ft_ov["avg_test_pass_rate"] - b_ov["avg_test_pass_rate"]) * 100

    print(f"\n  {'Metric':<30} {'Fine-Tuned':>12} {'Base':>12} {'Δ':>10}")
    print(f"  {'-'*65}")
    print(f"  {'Repair Success Rate':<30} {ft_ov['repair_success_rate']*100:>11.1f}% {b_ov['repair_success_rate']*100:>11.1f}% {rsr_delta:>+9.1f}%")
    print(f"  {'Avg Test Pass Rate':<30} {ft_ov['avg_test_pass_rate']*100:>11.1f}% {b_ov['avg_test_pass_rate']*100:>11.1f}% {tpr_delta:>+9.1f}%")
    print(f"  {'Avg Attempts':<30} {ft_ov['avg_attempts']:>12.1f} {b_ov['avg_attempts']:>12.1f}")

    winner = "Fine-Tuned" if rsr_delta > 0 else ("Base" if rsr_delta < 0 else "TIE")
    print(f"\n  🏆  Winner (by Repair Success Rate): {winner}")
    print(f"{sep}\n")


def save_results(results_dict: dict, output_path: str):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results_dict, f, indent=2)
    print(f"📁 Results saved → {output_path}")


# ── Main Evaluation Loop ───────────────────────────────────────────────

def run_evaluation(model, tokenizer, model_label: str, max_attempts: int, dataset: list) -> tuple:
    print(f"\n{'─'*70}")
    print(f"  Running evaluation: {model_label}  ({len(dataset)} samples, max {max_attempts} attempts)")
    print(f"{'─'*70}")

    all_results = []
    for i, sample in enumerate(dataset, 1):
        print(f"\n[{i:2}/{len(dataset)}] {sample['id']}  ({sample.get('bug_type', 'Unknown')})")
        res = evaluate_sample(sample, model, tokenizer, max_attempts=max_attempts)
        all_results.append(res)

    metrics = compute_metrics(all_results)
    print_report(model_label, metrics, all_results)
    return all_results, metrics


def main():
    parser = argparse.ArgumentParser(description="APR Evaluation — Fine-tuned vs Base")
    parser.add_argument("--mode", choices=["both", "finetuned", "base"], default="both")
    parser.add_argument("--max_attempts", type=int, default=3, help="Repair attempts per sample")
    parser.add_argument("--output_dir", default="eval_results", help="Output directory")
    parser.add_argument("--dataset", default="humaneval_eval_dataset.json", help="Dataset path")
    parser.add_argument("--limit", type=int, default=None, help="Limit to N samples")
    args = parser.parse_args()

    # Load and validate dataset
    dataset = load_evaluation_dataset(args.dataset, limit=args.limit)
    
    if not dataset:
        print("❌ No valid samples to evaluate!")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_all = {}

    ft_results = ft_metrics = None
    base_results = base_metrics = None

    # Fine-tuned model
    if args.mode in ("both", "finetuned"):
        print("\n🔬 Loading FINE-TUNED model...")
        ft_model, ft_tok, _ = load_finetuned_model()
        ft_results, ft_metrics = run_evaluation(
            ft_model, ft_tok, "Fine-Tuned (TestMate)", args.max_attempts, dataset
        )
        output_all["finetuned"] = {"results": ft_results, "metrics": ft_metrics}

        if args.mode == "both":
            import torch
            del ft_model
            gc.collect()
            torch.cuda.empty_cache()
            print("🧹 GPU memory freed")

    # Base model
    if args.mode in ("both", "base"):
        print("\n🔬 Loading BASE model...")
        b_model, b_tok, _ = load_base_model()
        base_results, base_metrics = run_evaluation(
            b_model, b_tok, "Base Qwen (1 Attempt)", 1, dataset
        )
        output_all["base"] = {"results": base_results, "metrics": base_metrics}

    # Comparison
    if ft_metrics and base_metrics:
        print_comparison(ft_metrics, base_metrics)
        output_all["comparison"] = {
            "finetuned_repair_rate": ft_metrics["overall"]["repair_success_rate"],
            "base_repair_rate": base_metrics["overall"]["repair_success_rate"],
            "delta_repair_rate": round(
                ft_metrics["overall"]["repair_success_rate"] - 
                base_metrics["overall"]["repair_success_rate"], 4
            ),
        }

    # Save results
    out_path = os.path.join(args.output_dir, f"eval_{timestamp}.json")
    save_results(output_all, out_path)

    print("✅ Evaluation complete!")


if __name__ == "__main__":
    main()