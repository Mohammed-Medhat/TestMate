"""
apr_evaluator.py — Comprehensive 4-Way APR Evaluation Framework

Modes (run independently or all together):
  base_simple    — Base Qwen-Instruct, minimal prompt (matches paper setup)
  base_pipeline  — Base Qwen-Instruct, full rich prompt (trace + description)
  finetuned_zero — Fine-tuned + LoRA, pass@1 only (zero-shot)
  finetuned_full — Fine-tuned + LoRA, iterative self-reflection k≤3

Speed optimisations vs previous version:
  - initial_test sandbox run is SKIPPED for base_simple (no fail output needed)
  - initial_test is optional for other modes via --skip_initial_test flag
  - GPU memory freed between model loads
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

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from config import REPAIR_TEMPERATURES, MAX_REPAIR_ATTEMPTS, TOP_P, MAX_NEW_TOKENS
except ImportError:
    REPAIR_TEMPERATURES = [0.2, 0.5, 0.8]
    MAX_REPAIR_ATTEMPTS = 3
    TOP_P = 0.95
    MAX_NEW_TOKENS = 1024


# ── Dataset Loading ────────────────────────────────────────────────────

def validate_sample(sample: dict, idx: int) -> bool:
    required = ['id', 'buggy_code', 'tests']
    missing = [f for f in required if f not in sample]
    if missing:
        print(f"⚠️  Sample {idx} missing fields: {missing}")
        return False
    if 'def test_' not in sample['tests']:
        print(f"⚠️  Sample {idx} ({sample.get('id','?')}) has no 'def test_' function!")
        return False
    if 'from solution import' not in sample['tests']:
        print(f"⚠️  Sample {idx} ({sample.get('id','?')}) missing 'from solution import'!")
        return False
    return True


def load_evaluation_dataset(json_path: str, limit: int = None) -> list:
    if not os.path.exists(json_path):
        print(f"❌ Dataset not found: {json_path}")
        sys.exit(1)
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"📊 Loaded {len(data)} samples from {json_path}")
    valid = [s for i, s in enumerate(data) if validate_sample(s, i)]
    print(f"✅ {len(valid)}/{len(data)} samples valid")
    if limit:
        valid = valid[:limit]
        print(f"✂️  Limited to first {limit} samples")
    return valid


# ── Model Loading ──────────────────────────────────────────────────────

def load_finetuned_model():
    from core.inference import load_model
    model, tokenizer = load_model()
    print("✅ Fine-tuned model loaded")
    return model, tokenizer


def load_base_model():
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
    return model, tokenizer


def free_gpu(model):
    import torch
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print("🧹 GPU memory freed")


# ── Inference ──────────────────────────────────────────────────────────

def run_model(model, tokenizer, prompt: str, attempt: int = 1) -> str:
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
    current_temp = REPAIR_TEMPERATURES[min(attempt - 1, len(REPAIR_TEMPERATURES) - 1)]
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=current_temp,
            top_p=TOP_P,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )
    return response.strip()


# ── Prompt Builders ────────────────────────────────────────────────────

def build_simple_prompt(sample: dict) -> str:
    """
    Minimal prompt matching the paper's zero-shot setup for base model comparison.
    No trace, no description context — just the task instruction + buggy code.
    """
    return (
        f"Fix the bug in the following Python code.\n\n"
        f"```python\n{sample['buggy_code']}\n```\n\n"
        f"Output ONLY the fixed Python function(s), no explanation."
    )


def build_paper_prompt(sample: dict) -> str:
    """
    Exact HumanEvalFix prompt format used by OctoPack and QiMeng-PRepair papers.
    The model receives the buggy code and must output the fixed version.
    No system instruction, no trace, no description — raw code only.
    This matches the bigcode-evaluation-harness format used in both papers.
    """
    return sample["buggy_code"]


def strip_markdown_only(raw: str) -> str:
    """
    Paper-style minimal post-processing: only strip markdown code fences.
    No AST surgery, no function extraction, no prompt-echo removal.
    Whatever the model outputs goes directly to solution.py — exactly
    as done in OctoPack and QiMeng-PRepair evaluation harnesses.
    """
    match = re.search(r"```(?:python)?(.*?)```", raw, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return raw.strip()


def build_rich_prompt(sample: dict, fail_output: str = "",
                      attempt: int = 1) -> str:
    """
    Full pipeline prompt: description + pytest trace + buggy code.
    Used for base_pipeline, finetuned_zero, and finetuned_full modes.
    """
    description = sample.get("description", "Fix the buggy code")
    failure_lines = [
        line.strip() for line in fail_output.splitlines()
        if any(kw in line for kw in ("E   ", "FAILED", "Error", "assert"))
    ]
    trace_text = "\n".join(failure_lines[:4]) or "AssertionError: test failed."

    prompt = f"ISSUE: {description}\n\n"
    prompt += f"TRACE:\n{trace_text}\n\n"
    prompt += f"BUGGY CODE:\n{sample['buggy_code']}\n\n"
    prompt += "Output ONLY the fixed code, no explanations."

    if attempt > 1 and fail_output:
        error_lines = [
            l for l in fail_output.splitlines()
            if any(kw in l for kw in ("FAILED", "AssertionError", "Error", "assert"))
        ][:5]
        prompt += (
            f"\n\n[ATTEMPT {attempt} — previous fix FAILED]\n"
            f"Previous error:\n" + "\n".join(error_lines) +
            "\n\nTry a COMPLETELY DIFFERENT approach."
        )
    return prompt


# ── Code Patching ──────────────────────────────────────────────────────

def extract_clean_code(raw: str) -> str:
    for marker in ("BUGGY CODE", "FIXED CODE", "ISSUE:", "TRACE:", "SUSPICIOUS"):
        if marker in raw:
            raw = raw[:raw.index(marker)]
    match = re.search(r"```(?:python)?(.*?)```", raw, re.DOTALL | re.IGNORECASE)
    if match:
        clean = match.group(1).strip()
    else:
        lines = raw.split("\n")
        start = 0
        for i, line in enumerate(lines):
            if line.strip().startswith(("def ", "class ", "@")):
                start = i
                break
        clean = "\n".join(lines[start:]).strip()
    clean = re.sub(r"^\s*\d+:\s?", "", clean, flags=re.MULTILINE)
    return clean


def apply_ast_patch(original_source: str, model_output: str) -> str | None:
    clean = extract_clean_code(model_output)
    try:
        new_ast = ast.parse(clean)
    except SyntaxError as e:
        print(f"      ⚠️  Model output has syntax error: {e}")
        return None
    new_fns = {n.name: n for n in new_ast.body
               if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    if not new_fns:
        print(f"      ⚠️  No functions found in model output")
        return None
    try:
        orig_ast = ast.parse(original_source)
    except SyntaxError:
        print(f"     ℹ  Original has syntax error — using full model output")
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


# ── Sandbox Test Runner ────────────────────────────────────────────────

def run_tests_in_sandbox(source_code: str, test_code: str) -> dict:
    tmpdir = tempfile.mkdtemp(prefix="apr_eval_")
    try:
        (Path(tmpdir) / "solution.py").write_text(source_code, encoding="utf-8")
        (Path(tmpdir) / "test_solution.py").write_text(test_code, encoding="utf-8")
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "test_solution.py",
             "-v", "--tb=short", "--no-header"],
            capture_output=True, text=True, cwd=tmpdir, timeout=30,
        )
        output = result.stdout + result.stderr
        passed_m = re.search(r"(\d+) passed", output)
        failed_m = re.search(r"(\d+) failed", output)
        num_passed = int(passed_m.group(1)) if passed_m else 0
        num_failed = int(failed_m.group(1)) if failed_m else 0
        num_total  = num_passed + num_failed
        if num_total == 0:
            return {"passed": False, "pass_rate": 0.0,
                    "num_passed": 0, "num_total": 0,
                    "output": output, "error": "No tests collected"}
        return {"passed": result.returncode == 0,
                "pass_rate": num_passed / num_total,
                "num_passed": num_passed, "num_total": num_total,
                "output": output}
    except subprocess.TimeoutExpired:
        return {"passed": False, "pass_rate": 0.0,
                "num_passed": 0, "num_total": 0,
                "output": "Timeout (30s)", "error": "Timeout"}
    except Exception as e:
        return {"passed": False, "pass_rate": 0.0,
                "num_passed": 0, "num_total": 0,
                "output": str(e), "error": str(e)}
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ── Core Sample Evaluator ──────────────────────────────────────────────

def evaluate_sample(sample: dict, model, tokenizer,
                    mode: str = "finetuned_full") -> dict:
    """
    Evaluate one sample under the given mode.

    mode:
      base_paper     — 1 attempt, raw HumanEvalFix prompt, NO AST patch,
                       NO initial sandbox run. Matches OctoPack / QiMeng eval.
      base_simple    — 1 attempt, simple prompt, NO initial sandbox run (fast)
      base_pipeline  — 1 attempt, rich prompt, WITH initial sandbox for trace
      finetuned_zero — 1 attempt, rich prompt, WITH initial sandbox for trace
      finetuned_full — k≤3 attempts, rich prompt, WITH initial sandbox + reflection
    """
    result = {
        "id": sample["id"],
        "bug_type": sample.get("bug_type", "Unknown"),
        "repaired": False,
        "attempts_used": 0,
        "final_pass_rate": 0.0,
        "final_passed": 0,
        "final_total": 0,
        "time_seconds": 0.0,
        "mode": mode,
    }

    buggy_code = sample["buggy_code"]
    test_code  = sample["tests"]
    max_attempts = 1 if mode in ("base_paper", "base_simple",
                                 "base_pipeline", "finetuned_zero") \
                   else MAX_REPAIR_ATTEMPTS

    # ── Initial test (skip for paper/simple — no fail output needed) ───
    fail_output = ""
    if mode not in ("base_paper", "base_simple"):
        print(f"   🧪 Initial test...", end=" ")
        initial = run_tests_in_sandbox(buggy_code, test_code)
        if initial.get("error") == "No tests collected":
            print("❌ TEST STRUCTURE ERROR")
            result["error"] = "Test structure invalid"
            return result
        if initial["passed"]:
            print(f"⚠️  Already passing! ({initial['num_passed']}/{initial['num_total']})")
            result.update(repaired=True, final_pass_rate=1.0,
                          final_passed=initial["num_passed"],
                          final_total=initial["num_total"])
            return result
        print(f"❌ Failed ({initial['num_passed']}/{initial['num_total']})")
        fail_output = initial["output"]

    # ── Repair loop ────────────────────────────────────────────────────
    t_start = time.time()
    for attempt in range(1, max_attempts + 1):
        print(f"   🔄 Attempt {attempt}/{max_attempts}...", end=" ")

        # Build prompt
        if mode == "base_paper":
            prompt = build_paper_prompt(sample)
        elif mode == "base_simple":
            prompt = build_simple_prompt(sample)
        else:
            prompt = build_rich_prompt(sample, fail_output, attempt)

        raw_out = run_model(model, tokenizer, prompt, attempt=attempt)

        # ── Apply patch (paper mode = NO AST surgery) ──────────────────
        if mode == "base_paper":
            # Exact paper evaluation: strip markdown only, write directly
            patched = strip_markdown_only(raw_out)
            if not patched:
                print("❌ Empty output")
                result["attempts_used"] = attempt
                continue
        else:
            patched = apply_ast_patch(buggy_code, raw_out)
            if patched is None:
                print("❌ Patch failed")
                result["attempts_used"] = attempt
                continue

        test_res = run_tests_in_sandbox(patched, test_code)
        result["attempts_used"] = attempt

        if test_res.get("error") == "No tests collected":
            print("❌ Test error after patch")
            break

        if test_res["passed"]:
            print(f"✅ FIXED ({test_res['num_passed']}/{test_res['num_total']})")
            result.update(repaired=True,
                          final_pass_rate=test_res["pass_rate"],
                          final_passed=test_res["num_passed"],
                          final_total=test_res["num_total"])
            break
        else:
            print(f"❌ Still failing ({test_res['num_passed']}/{test_res['num_total']})")
            fail_output = test_res["output"]
            result.update(final_pass_rate=test_res["pass_rate"],
                          final_passed=test_res["num_passed"],
                          final_total=test_res["num_total"])

    result["time_seconds"] = round(time.time() - t_start, 2)
    return result


# ── Metrics & Reporting ────────────────────────────────────────────────

def compute_metrics(results: list) -> dict:
    total    = len(results)
    repaired = sum(1 for r in results if r["repaired"])
    pass_rates = [r["final_pass_rate"] for r in results]
    attempts   = [r["attempts_used"] for r in results if r["attempts_used"] > 0]
    att1 = sum(1 for r in results if r["repaired"] and r["attempts_used"] == 1)
    att2 = sum(1 for r in results if r["repaired"] and r["attempts_used"] == 2)
    att3 = sum(1 for r in results if r["repaired"] and r["attempts_used"] >= 3)
    return {
        "overall": {
            "total_samples":       total,
            "repaired":            repaired,
            "repair_success_rate": round(repaired / total, 4) if total else 0,
            "avg_test_pass_rate":  round(sum(pass_rates) / len(pass_rates), 4) if pass_rates else 0,
            "avg_attempts":        round(sum(attempts) / len(attempts), 2) if attempts else 0,
            "fixed_attempt_1":     att1,
            "fixed_attempt_2":     att2,
            "fixed_attempt_3plus": att3,
        }
    }


def run_evaluation(model, tokenizer, label: str,
                   mode: str, dataset: list) -> tuple:
    print(f"\n{'─'*70}")
    print(f"  {label}  |  mode={mode}  |  {len(dataset)} samples")
    print(f"{'─'*70}")
    all_results = []
    for i, sample in enumerate(dataset, 1):
        print(f"\n[{i:3}/{len(dataset)}] {sample['id']}")
        res = evaluate_sample(sample, model, tokenizer, mode=mode)
        all_results.append(res)
    metrics = compute_metrics(all_results)
    ov = metrics["overall"]
    print(f"\n  ✅ {label}: {ov['repair_success_rate']*100:.2f}%"
          f"  ({ov['repaired']}/{ov['total_samples']})")
    return all_results, metrics


def print_full_comparison(all_metrics: dict):
    sep = "═" * 72
    print(f"\n{sep}")
    print(f"  📊  FULL 4-WAY COMPARISON")
    print(sep)

    PAPER = {
        "base_paper":     55.49,   # OctoPack / QiMeng reported baseline
        "base_simple":    None,
        "base_pipeline":  None,
        "finetuned_zero": 84.70,
        "finetuned_full": 90.80,
    }
    LABELS = {
        "base_paper":     "Base model — paper prompt, no AST patch (paper-style ✅)",
        "base_simple":    "Base model — simple prompt, with AST patch",
        "base_pipeline":  "Base model — rich prompt,   with AST patch",
        "finetuned_zero": "Fine-tuned — zero-shot      (pass@1)",
        "finetuned_full": "Fine-tuned — self-reflection (pass@k≤3)",
    }

    print(f"\n  {'Configuration':<52} {'Yours':>8} {'Paper':>8} {'Δ':>9}")
    print(f"  {'-'*78}")
    for mode in ("base_paper", "base_simple", "base_pipeline",
                 "finetuned_zero", "finetuned_full"):
        if mode not in all_metrics:
            continue
        rate = all_metrics[mode]["overall"]["repair_success_rate"] * 100
        paper = PAPER[mode]
        delta = f"{rate - paper:+.2f}pp" if paper else "  —"
        paper_str = f"{paper:.2f}%" if paper else "   —"
        print(f"  {LABELS[mode]:<52} {rate:>7.2f}% {paper_str:>8} {delta:>9}")

    print(f"\n  Incremental gains:")
    modes = [m for m in ("base_paper","base_simple","base_pipeline",
                         "finetuned_zero","finetuned_full")
             if m in all_metrics]
    for i in range(1, len(modes)):
        prev = all_metrics[modes[i-1]]["overall"]["repair_success_rate"] * 100
        curr = all_metrics[modes[i]]["overall"]["repair_success_rate"] * 100
        print(f"    {LABELS[modes[i-1]][:35]} → {LABELS[modes[i]][:25]}: {curr-prev:+.2f}pp")
    print(f"\n{sep}\n")


def save_results(data: dict, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"📁 Results saved → {path}")


# ── Main ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="APR 4-Way Evaluation",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--mode",
        choices=["all", "base_paper", "base_simple", "base_pipeline",
                 "finetuned_zero", "finetuned_full",
                 "base_only", "finetuned_only"],
        default="all",
        help=(
            "all            — run all 5 configurations (recommended)\n"
            "base_paper     — base model, raw HumanEvalFix prompt (matches paper)\n"
            "base_only      — base_simple + base_pipeline\n"
            "finetuned_only — finetuned_zero + finetuned_full\n"
            "base_simple    — base model, minimal prompt (paper-style)\n"
            "base_pipeline  — base model, rich prompt\n"
            "finetuned_zero — fine-tuned, pass@1 only\n"
            "finetuned_full — fine-tuned, iterative k≤3"
        )
    )
    parser.add_argument("--output_dir", default="eval_results")
    parser.add_argument("--dataset",    default="humaneval_eval_dataset.json")
    parser.add_argument("--limit",      type=int, default=None,
                        help="Limit to N samples (useful for quick testing)")
    args = parser.parse_args()

    dataset = load_evaluation_dataset(args.dataset, limit=args.limit)
    if not dataset:
        print("❌ No valid samples to evaluate!")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_all = {}
    all_metrics = {}

    # ── Which configs to run ───────────────────────────────────────────
    base_modes = []
    ft_modes   = []

    if args.mode in ("all", "base_only", "base_paper"):
        base_modes.append("base_paper")
    if args.mode in ("all", "base_only", "base_simple"):
        base_modes.append("base_simple")
    if args.mode in ("all", "base_only", "base_pipeline"):
        base_modes.append("base_pipeline")
    if args.mode in ("all", "finetuned_only", "finetuned_zero"):
        ft_modes.append("finetuned_zero")
    if args.mode in ("all", "finetuned_only", "finetuned_full"):
        ft_modes.append("finetuned_full")
    if args.mode in ("all", "base_only", "base_full"):
        base_modes.append("base_full") 

    # ── Base model evaluations ─────────────────────────────────────────
    if base_modes:
        print("\n🔬 Loading BASE model...")
        b_model, b_tok = load_base_model()
        for mode in base_modes:
            label = f"Base Qwen — {mode}"
            results, metrics = run_evaluation(b_model, b_tok, label, mode, dataset)
            output_all[mode] = {"results": results, "metrics": metrics}
            all_metrics[mode] = metrics
        free_gpu(b_model)

    # ── Fine-tuned model evaluations ───────────────────────────────────
    if ft_modes:
        print("\n🔬 Loading FINE-TUNED model...")
        ft_model, ft_tok = load_finetuned_model()
        for mode in ft_modes:
            label = f"TestMate — {mode}"
            results, metrics = run_evaluation(ft_model, ft_tok, label, mode, dataset)
            output_all[mode] = {"results": results, "metrics": metrics}
            all_metrics[mode] = metrics
        free_gpu(ft_model)

    # ── Print & save ───────────────────────────────────────────────────
    if len(all_metrics) > 1:
        print_full_comparison(all_metrics)

    out_path = os.path.join(args.output_dir, f"eval_{timestamp}.json")
    save_results(output_all, out_path)
    print("✅ Evaluation complete!")


if __name__ == "__main__":
    main()