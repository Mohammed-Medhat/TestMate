# ============================================================
# KAGGLE: Baseline Model Evaluation (No Fine-tuning)
# ============================================================
# Compare: Base Qwen 7B vs Fine-tuned TestGen Model
# ============================================================

# CELL 1: Install
"""
!pip install -q transformers datasets accelerate bitsandbytes
"""

# CELL 2: Imports
import torch
import json
import time
import ast
import re
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, Dict, List
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset

# Config
BASE_MODEL = "Qwen/Qwen2.5-Coder-7B"
OUTPUT_DIR = "/kaggle/working/baseline_evaluation"
MAX_INSTANCES = 300  # Same as fine-tuned eval

print(f"GPU: {torch.cuda.get_device_name(0)}")
print("🔵 BASELINE MODEL (No Fine-tuning)")

# ============================================================
# CELL 3: Metrics (Same as fine-tuned eval)
# ============================================================

@dataclass
class TestMetrics:
    instance_id: str = ""
    repo: str = ""
    generation_time_ms: float = 0.0
    test_length: int = 0
    syntax_valid: bool = False
    syntax_error: str = ""
    has_imports: bool = False
    has_test_function: bool = False
    has_class: bool = False
    has_setup: bool = False
    has_teardown: bool = False
    has_assertions: bool = False
    assertion_count: int = 0
    assertion_types: List[str] = None
    compile_success: bool = False
    compile_error: str = ""
    cyclomatic_complexity: int = 0
    num_functions: int = 0
    num_branches: int = 0
    relevance_score: float = 0.0
    keyword_coverage: float = 0.0
    docstring_present: bool = False
    follows_aaa_pattern: bool = False
    has_edge_cases: bool = False
    has_error_handling: bool = False
    
    def __post_init__(self):
        if self.assertion_types is None:
            self.assertion_types = []
    
    def to_dict(self):
        return {k: v for k, v in self.__dict__.items()}

# Metric functions (same as fine-tuned)
def check_syntax(code: str) -> Tuple[bool, str]:
    try:
        ast.parse(code)
        return True, ""
    except SyntaxError as e:
        return False, f"Line {e.lineno}: {e.msg}"

def check_compile(code: str) -> Tuple[bool, str]:
    try:
        compile(code, '<test>', 'exec')
        return True, ""
    except Exception as e:
        return False, str(e)

def count_assertions(code: str) -> Tuple[int, List[str]]:
    patterns = {
        'assert': r'\bassert\b',
        'assertEqual': r'\.assertEqual\(',
        'assertTrue': r'\.assertTrue\(',
        'assertRaises': r'\.assertRaises\(',
        'pytest.raises': r'pytest\.raises',
    }
    types_found = []
    total = 0
    for name, pattern in patterns.items():
        count = len(re.findall(pattern, code))
        if count > 0:
            types_found.append(name)
            total += count
    return total, types_found

def calculate_complexity(code: str) -> Dict[str, int]:
    try:
        tree = ast.parse(code)
        complexity = 1
        num_functions = 0
        num_branches = 0
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                num_functions += 1
            if isinstance(node, (ast.If, ast.While, ast.For)):
                complexity += 1
                num_branches += 1
            if isinstance(node, ast.ExceptHandler):
                complexity += 1
        return {'cyclomatic_complexity': complexity, 'num_functions': num_functions, 'num_branches': num_branches}
    except:
        return {'cyclomatic_complexity': 0, 'num_functions': 0, 'num_branches': 0}

def calculate_relevance(test_code: str, problem: str) -> Dict[str, float]:
    stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'to', 'in', 'on', 'for', 'of', 'and', 'or'}
    problem_words = set(re.findall(r'\b[a-zA-Z_]\w+\b', problem.lower())) - stop_words
    test_words = set(re.findall(r'\b[a-zA-Z_]\w+\b', test_code.lower())) - stop_words
    if not problem_words:
        return {'relevance_score': 0.0, 'keyword_coverage': 0.0}
    overlap = problem_words & test_words
    return {
        'relevance_score': len(overlap) / len(problem_words),
        'keyword_coverage': len(overlap) / max(len(test_words), 1)
    }

def check_test_patterns(code: str) -> Dict[str, bool]:
    return {
        'follows_aaa_pattern': 'assert' in code.lower(),
        'has_edge_cases': any(p in code for p in ['None', 'empty', '[]', '{}', '""']),
        'has_error_handling': 'raises' in code.lower() or 'exception' in code.lower()
    }

def check_structure(code: str) -> Dict[str, bool]:
    return {
        'has_imports': 'import' in code,
        'has_test_function': bool(re.search(r'def test_\w+', code)),
        'has_class': bool(re.search(r'class \w+Test', code)),
        'has_setup': 'setup' in code.lower(),
        'has_teardown': 'teardown' in code.lower(),
        'docstring_present': '"""' in code or "'''" in code
    }

def evaluate_test(test_code: str, instance_id: str, repo: str, problem: str, gen_time: float) -> TestMetrics:
    metrics = TestMetrics(instance_id=instance_id, repo=repo, generation_time_ms=gen_time, test_length=len(test_code))
    metrics.syntax_valid, metrics.syntax_error = check_syntax(test_code)
    metrics.compile_success, metrics.compile_error = check_compile(test_code)
    
    structure = check_structure(test_code)
    for k, v in structure.items():
        setattr(metrics, k, v)
    
    metrics.assertion_count, metrics.assertion_types = count_assertions(test_code)
    metrics.has_assertions = metrics.assertion_count > 0
    
    complexity = calculate_complexity(test_code)
    for k, v in complexity.items():
        setattr(metrics, k, v)
    
    relevance = calculate_relevance(test_code, problem)
    metrics.relevance_score = relevance['relevance_score']
    metrics.keyword_coverage = relevance['keyword_coverage']
    
    patterns = check_test_patterns(test_code)
    for k, v in patterns.items():
        setattr(metrics, k, v)
    
    return metrics

# ============================================================
# CELL 4: Load BASELINE Model (No LoRA!)
# ============================================================

def load_baseline_model():
    """Load base Qwen model WITHOUT any fine-tuning."""
    print("Loading BASELINE model (no fine-tuning)...")
    
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16
    )
    
    print("✅ BASELINE model loaded (Qwen 7B, no fine-tuning)")
    return model, tokenizer

# ============================================================
# CELL 5: Generate Test (Same prompt as fine-tuned)
# ============================================================

def generate_test(model, tokenizer, code_or_problem: str, repo: str) -> Tuple[str, float]:
    prompt = f"""=== COMPREHENSIVE TEST GENERATION ===

## Source Code / Bug Description
Repository: {repo}

{code_or_problem[:2000]}

## Context
Generate comprehensive unit tests for the above code.
Include: basic tests, edge cases, error handling.

## Generated Tests
```python
"""
    
    start = time.time()
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=500,
            temperature=0.7,
            do_sample=True,
            top_p=0.95,
            pad_token_id=tokenizer.pad_token_id
        )
    
    gen_time = (time.time() - start) * 1000
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    test_code = response[len(prompt):]
    
    if "```" in test_code:
        test_code = test_code[:test_code.find("```")]
    
    return test_code.strip(), gen_time

# ============================================================
# CELL 6: Run Evaluation
# ============================================================

def run_baseline_evaluation(model, tokenizer, max_instances=300):
    print("="*60)
    print("🔵 BASELINE EVALUATION (No Fine-tuning)")
    print("="*60)
    
    dataset = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
    print(f"Instances: {min(len(dataset), max_instances)}")
    
    all_metrics = []
    generated_tests = []
    start_time = time.time()
    
    for i, instance in enumerate(dataset):
        if i >= max_instances:
            break
        
        instance_id = instance.get('instance_id', f'instance_{i}')
        repo = instance.get('repo', 'unknown')
        problem = instance.get('problem_statement', '')
        
        print(f"\n[{i+1}/{max_instances}] {instance_id}")
        
        try:
            test_code, gen_time = generate_test(model, tokenizer, problem, repo)
            metrics = evaluate_test(test_code, instance_id, repo, problem, gen_time)
            
            all_metrics.append(metrics.to_dict())
            generated_tests.append({'instance_id': instance_id, 'test_code': test_code})
            
            status = "✅" if metrics.syntax_valid and metrics.has_assertions else "⚠️"
            print(f"   {status} syntax={metrics.syntax_valid}, asserts={metrics.assertion_count}")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    total_time = time.time() - start_time
    return all_metrics, generated_tests, total_time

# ============================================================
# CELL 7: Summary
# ============================================================

def calculate_summary(metrics: List[dict], total_time: float) -> dict:
    n = len(metrics)
    return {
        'model': 'BASELINE (Qwen 7B, no fine-tuning)',
        'total_instances': n,
        'total_time_minutes': round(total_time / 60, 2),
        'syntax_valid_pct': round(100 * sum(1 for m in metrics if m['syntax_valid']) / n, 2),
        'compile_success_pct': round(100 * sum(1 for m in metrics if m['compile_success']) / n, 2),
        'has_test_function_pct': round(100 * sum(1 for m in metrics if m['has_test_function']) / n, 2),
        'has_imports_pct': round(100 * sum(1 for m in metrics if m['has_imports']) / n, 2),
        'has_docstring_pct': round(100 * sum(1 for m in metrics if m['docstring_present']) / n, 2),
        'has_assertions_pct': round(100 * sum(1 for m in metrics if m['has_assertions']) / n, 2),
        'avg_assertion_count': round(sum(m['assertion_count'] for m in metrics) / n, 2),
        'avg_complexity': round(sum(m['cyclomatic_complexity'] for m in metrics) / n, 2),
        'avg_relevance': round(sum(m['relevance_score'] for m in metrics) / n, 4),
        'follows_aaa_pct': round(100 * sum(1 for m in metrics if m['follows_aaa_pattern']) / n, 2),
        'has_edge_cases_pct': round(100 * sum(1 for m in metrics if m['has_edge_cases']) / n, 2),
        'has_error_handling_pct': round(100 * sum(1 for m in metrics if m['has_error_handling']) / n, 2),
        'valid_useful_pct': round(100 * sum(1 for m in metrics if m['syntax_valid'] and m['has_assertions']) / n, 2),
        'high_quality_pct': round(100 * sum(1 for m in metrics if m['syntax_valid'] and m['has_assertions'] and m['relevance_score'] > 0.1) / n, 2),
    }

def print_summary(s: dict):
    print("\n" + "="*60)
    print("📊 BASELINE EVALUATION SUMMARY")
    print("="*60)
    print(f"\n🔵 Model: {s['model']}")
    print(f"\n⏱️ GENERATION:")
    print(f"   Instances: {s['total_instances']}")
    print(f"   Time: {s['total_time_minutes']} min")
    print(f"\n📝 SYNTAX:")
    print(f"   Valid: {s['syntax_valid_pct']}%")
    print(f"   Compile: {s['compile_success_pct']}%")
    print(f"\n🏗️ STRUCTURE:")
    print(f"   Test function: {s['has_test_function_pct']}%")
    print(f"   Imports: {s['has_imports_pct']}%")
    print(f"\n✓ ASSERTIONS:")
    print(f"   Has assertions: {s['has_assertions_pct']}%")
    print(f"   Avg count: {s['avg_assertion_count']}")
    print(f"\n🎯 QUALITY:")
    print(f"   Relevance: {s['avg_relevance']:.2%}")
    print(f"\n🧪 PATTERNS:")
    print(f"   AAA pattern: {s['follows_aaa_pct']}%")
    print(f"   Edge cases: {s['has_edge_cases_pct']}%")
    print(f"   Error handling: {s['has_error_handling_pct']}%")
    print(f"\n⭐ FINAL SCORES:")
    print(f"   Valid & Useful: {s['valid_useful_pct']}%")
    print(f"   High Quality: {s['high_quality_pct']}%")

def save_results(metrics, tests, summary):
    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    with open(f"{OUTPUT_DIR}/baseline_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    with open(f"{OUTPUT_DIR}/baseline_detailed.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    with open(f"{OUTPUT_DIR}/baseline_tests.jsonl", 'w') as f:
        for test in tests:
            f.write(json.dumps(test) + '\n')
    print(f"\n✅ Saved to {OUTPUT_DIR}/")

# ============================================================
# CELL 8: Run Everything
# ============================================================

print("Loading baseline model...")
model, tokenizer = load_baseline_model()

print("\nRunning baseline evaluation...")
metrics, tests, total_time = run_baseline_evaluation(model, tokenizer, MAX_INSTANCES)

print("\nCalculating summary...")
summary = calculate_summary(metrics, total_time)

print_summary(summary)

print("\nSaving results...")
save_results(metrics, tests, summary)

print("\n🎉 BASELINE EVALUATION COMPLETE!")

# ============================================================
# CELL 9: Download
# ============================================================
"""
!zip -r /kaggle/working/baseline_eval.zip /kaggle/working/baseline_evaluation/
"""
