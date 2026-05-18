# ============================================================
# GOOGLE COLAB: Full Research-Grade Test Evaluation
# ============================================================
# Complete evaluation with ALL metrics including:
# - Execution (actually runs tests)
# - Coverage (line, branch, statement)
# - All Kaggle metrics
# ============================================================
# Colab Pro recommended for long runs
# ============================================================

# CELL 1: Setup Environment
"""
# Install dependencies
!pip install -q transformers datasets peft accelerate coverage pytest bitsandbytes

# For GPU support
import torch
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

# Mount Drive for saving results
from google.colab import drive
drive.mount('/content/drive')
"""

# CELL 2: Upload Model or Load from Drive
"""
# Option 1: Upload from local
from google.colab import files
uploaded = files.upload()  # Upload testgen_model.zip
!unzip testgen_model.zip -d /content/

# Option 2: Load from Drive
# !cp /content/drive/MyDrive/testgen_model.zip /content/
# !unzip /content/testgen_model.zip -d /content/
"""

# CELL 3: Imports and Config
import torch
import json
import time
import ast
import re
import subprocess
import tempfile
import os
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import Tuple, Dict, List, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset

# Config
BASE_MODEL = "Qwen/Qwen2.5-Coder-7B"
LORA_PATH = "/content/testgen_model/final"  # Update based on your upload
OUTPUT_DIR = "/content/testgen_full_evaluation"
SAVE_TO_DRIVE = "/content/drive/MyDrive/testgen_evaluation"  # Optional

# Batch Settings (for running in multiple sessions)
START_FROM = 0         # Starting index (0, 100, 200 for 3 batches)
BATCH_SIZE = 100       # Instances per batch
TOTAL_INSTANCES = 300  # Total SWE-bench Lite instances

# Feature Flags
RUN_EXECUTION = True   # Run tests with pytest
RUN_COVERAGE = True    # Measure coverage

# Batch name for saving
BATCH_NAME = f"batch_{START_FROM}_{START_FROM + BATCH_SIZE}"

print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"📦 Batch: {BATCH_NAME} (instances {START_FROM}-{START_FROM + BATCH_SIZE})")

# ============================================================
# CELL 4: Complete Metrics Class
# ============================================================

@dataclass
class FullTestMetrics:
    """Complete research-grade test evaluation metrics."""
    instance_id: str = ""
    repo: str = ""
    
    # Generation
    generation_time_ms: float = 0.0
    test_length: int = 0
    
    # Syntax Metrics
    syntax_valid: bool = False
    syntax_error: str = ""
    
    # Structure
    has_imports: bool = False
    has_test_function: bool = False
    has_class: bool = False
    has_setup: bool = False
    has_docstring: bool = False
    
    # Assertions
    has_assertions: bool = False
    assertion_count: int = 0
    assertion_types: List[str] = field(default_factory=list)
    
    # Compilation
    compile_success: bool = False
    compile_error: str = ""
    
    # Execution (Colab-only)
    execution_attempted: bool = False
    execution_success: bool = False
    execution_error: str = ""
    execution_time_ms: float = 0.0
    tests_passed: int = 0
    tests_failed: int = 0
    
    # Coverage (Colab-only)
    coverage_attempted: bool = False
    line_coverage: float = 0.0
    branch_coverage: float = 0.0
    statements_covered: int = 0
    statements_total: int = 0
    
    # Complexity
    cyclomatic_complexity: int = 0
    num_functions: int = 0
    
    # Quality
    relevance_score: float = 0.0
    follows_aaa_pattern: bool = False
    has_edge_cases: bool = False
    has_error_handling: bool = False
    
    def to_dict(self):
        return {k: v for k, v in self.__dict__.items()}

# ============================================================
# CELL 5: Metric Calculators (Kaggle-compatible)
# ============================================================

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

def calculate_complexity(code: str) -> int:
    try:
        tree = ast.parse(code)
        complexity = 1
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            if isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1
        return complexity
    except:
        return 0

def calculate_relevance(test_code: str, problem: str) -> float:
    stop_words = {'the', 'a', 'an', 'is', 'are', 'to', 'in', 'on', 'for', 'of', 'and', 'or'}
    problem_words = set(re.findall(r'\b[a-zA-Z_]\w+\b', problem.lower())) - stop_words
    test_words = set(re.findall(r'\b[a-zA-Z_]\w+\b', test_code.lower())) - stop_words
    if not problem_words:
        return 0.0
    return len(problem_words & test_words) / len(problem_words)

def check_patterns(code: str) -> Dict[str, bool]:
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
        'has_docstring': '"""' in code or "'''" in code
    }

# ============================================================
# CELL 6: Execution & Coverage (Colab-exclusive)
# ============================================================

def execute_test(test_code: str, timeout: int = 30) -> Dict:
    """Execute test with pytest and return results."""
    
    # Create temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        # Add minimal imports if missing
        if 'import pytest' not in test_code:
            f.write("import pytest\n")
        f.write(test_code)
        f.flush()
        temp_path = f.name
    
    result = {
        'execution_attempted': True,
        'execution_success': False,
        'execution_error': '',
        'execution_time_ms': 0.0,
        'tests_passed': 0,
        'tests_failed': 0
    }
    
    try:
        start = time.time()
        proc = subprocess.run(
            ['python', '-m', 'pytest', temp_path, '-v', '--tb=short', '-q'],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        result['execution_time_ms'] = (time.time() - start) * 1000
        
        # Parse output
        output = proc.stdout + proc.stderr
        
        # Check for passed/failed
        passed_match = re.search(r'(\d+) passed', output)
        failed_match = re.search(r'(\d+) failed', output)
        
        result['tests_passed'] = int(passed_match.group(1)) if passed_match else 0
        result['tests_failed'] = int(failed_match.group(1)) if failed_match else 0
        result['execution_success'] = proc.returncode == 0
        
        if not result['execution_success']:
            result['execution_error'] = output[-500:]  # Last 500 chars
            
    except subprocess.TimeoutExpired:
        result['execution_error'] = "Timeout"
        result['execution_time_ms'] = timeout * 1000
    except Exception as e:
        result['execution_error'] = str(e)
    finally:
        os.unlink(temp_path)
    
    return result

def run_coverage(test_code: str, timeout: int = 60) -> Dict:
    """Run test with coverage measurement."""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        if 'import pytest' not in test_code:
            f.write("import pytest\n")
        f.write(test_code)
        f.flush()
        temp_path = f.name
    
    result = {
        'coverage_attempted': True,
        'line_coverage': 0.0,
        'branch_coverage': 0.0,
        'statements_covered': 0,
        'statements_total': 0
    }
    
    try:
        # Run with coverage
        subprocess.run(
            ['python', '-m', 'coverage', 'run', '--branch', temp_path],
            capture_output=True,
            timeout=timeout
        )
        
        # Generate JSON report
        cov_json = temp_path + '.coverage.json'
        subprocess.run(
            ['python', '-m', 'coverage', 'json', '-o', cov_json],
            capture_output=True
        )
        
        # Parse coverage
        if os.path.exists(cov_json):
            with open(cov_json) as f:
                cov_data = json.load(f)
            
            totals = cov_data.get('totals', {})
            result['line_coverage'] = totals.get('percent_covered', 0.0)
            result['branch_coverage'] = totals.get('percent_covered_branches', 0.0)
            result['statements_covered'] = totals.get('covered_lines', 0)
            result['statements_total'] = totals.get('num_statements', 0)
            
            os.unlink(cov_json)
            
    except Exception as e:
        pass  # Coverage failed, keep defaults
    finally:
        os.unlink(temp_path)
        # Clean up .coverage file
        if os.path.exists('.coverage'):
            os.unlink('.coverage')
    
    return result

# ============================================================
# CELL 7: Full Evaluation Function
# ============================================================

def evaluate_test_full(test_code: str, instance_id: str, repo: str, 
                       problem: str, gen_time: float) -> FullTestMetrics:
    """Complete evaluation with execution and coverage."""
    
    metrics = FullTestMetrics(
        instance_id=instance_id,
        repo=repo,
        generation_time_ms=gen_time,
        test_length=len(test_code)
    )
    
    # Basic metrics (work everywhere)
    metrics.syntax_valid, metrics.syntax_error = check_syntax(test_code)
    metrics.compile_success, metrics.compile_error = check_compile(test_code)
    
    structure = check_structure(test_code)
    metrics.has_imports = structure['has_imports']
    metrics.has_test_function = structure['has_test_function']
    metrics.has_class = structure['has_class']
    metrics.has_setup = structure['has_setup']
    metrics.has_docstring = structure['has_docstring']
    
    metrics.assertion_count, metrics.assertion_types = count_assertions(test_code)
    metrics.has_assertions = metrics.assertion_count > 0
    
    metrics.cyclomatic_complexity = calculate_complexity(test_code)
    metrics.num_functions = len(re.findall(r'def \w+', test_code))
    
    metrics.relevance_score = calculate_relevance(test_code, problem)
    
    patterns = check_patterns(test_code)
    metrics.follows_aaa_pattern = patterns['follows_aaa_pattern']
    metrics.has_edge_cases = patterns['has_edge_cases']
    metrics.has_error_handling = patterns['has_error_handling']
    
    # Execution (Colab-only)
    if RUN_EXECUTION and metrics.compile_success:
        exec_result = execute_test(test_code)
        metrics.execution_attempted = exec_result['execution_attempted']
        metrics.execution_success = exec_result['execution_success']
        metrics.execution_error = exec_result['execution_error']
        metrics.execution_time_ms = exec_result['execution_time_ms']
        metrics.tests_passed = exec_result['tests_passed']
        metrics.tests_failed = exec_result['tests_failed']
    
    # Coverage (Colab-only)
    if RUN_COVERAGE and metrics.execution_success:
        cov_result = run_coverage(test_code)
        metrics.coverage_attempted = cov_result['coverage_attempted']
        metrics.line_coverage = cov_result['line_coverage']
        metrics.branch_coverage = cov_result['branch_coverage']
        metrics.statements_covered = cov_result['statements_covered']
        metrics.statements_total = cov_result['statements_total']
    
    return metrics

# ============================================================
# CELL 8: Load Model
# ============================================================

def load_model():
    print("Loading model...")
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Use 4-bit quantization to fit in T4 memory
    from transformers import BitsAndBytesConfig
    
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16
    )
    
    print(f"Loading LoRA from {LORA_PATH}...")
    model = PeftModel.from_pretrained(base_model, LORA_PATH)
    
    print("✅ Model loaded!")
    return model, tokenizer

# ============================================================
# CELL 9: Generate Test
# ============================================================

def generate_test(model, tokenizer, problem: str, repo: str) -> Tuple[str, float]:
    prompt = f"""=== TEST CASE GENERATION ===

## Bug Report
Repository: {repo}

{problem[:1500]}

## Task
Generate a Python test case that reproduces this bug.

## Test Case
```python
"""
    
    start = time.time()
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1536)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=400,
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
# CELL 10: Run Full Evaluation
# ============================================================

def run_full_evaluation(model, tokenizer, start_from=0, batch_size=100):
    print("="*60)
    print("FULL RESEARCH-GRADE EVALUATION (Colab)")
    print("="*60)
    print(f"Batch: {start_from} to {start_from + batch_size}")
    print(f"Execution: {RUN_EXECUTION}, Coverage: {RUN_COVERAGE}")
    
    dataset = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
    end_idx = min(start_from + batch_size, len(dataset))
    print(f"Instances: {start_from} to {end_idx} (of {len(dataset)} total)")
    
    all_metrics = []
    generated_tests = []
    
    start_time = time.time()
    
    for i, instance in enumerate(dataset):
        # Skip until start_from
        if i < start_from:
            continue
        # Stop at batch end
        if i >= end_idx:
            break
        
        instance_id = instance.get('instance_id', f'instance_{i}')
        repo = instance.get('repo', 'unknown')
        problem = instance.get('problem_statement', '')
        
        print(f"\n[{i+1}/{end_idx}] {instance_id}")
        
        try:
            test_code, gen_time = generate_test(model, tokenizer, problem, repo)
            metrics = evaluate_test_full(test_code, instance_id, repo, problem, gen_time)
            
            all_metrics.append(metrics.to_dict())
            generated_tests.append({
                'instance_id': instance_id,
                'test_code': test_code
            })
            
            # Status
            exec_status = "✅" if metrics.execution_success else "❌" if metrics.execution_attempted else "⏭️"
            print(f"   syntax={metrics.syntax_valid}, exec={exec_status}, "
                  f"cov={metrics.line_coverage:.1f}%, rel={metrics.relevance_score:.2f}")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    total_time = time.time() - start_time
    
    return all_metrics, generated_tests, total_time

# ============================================================
# CELL 11: Summary
# ============================================================

def calculate_full_summary(metrics: List[dict], total_time: float) -> dict:
    n = len(metrics)
    
    summary = {
        'total_instances': n,
        'total_time_minutes': round(total_time / 60, 2),
        
        # Syntax
        'syntax_valid_pct': round(100 * sum(1 for m in metrics if m['syntax_valid']) / n, 2),
        'compile_success_pct': round(100 * sum(1 for m in metrics if m['compile_success']) / n, 2),
        
        # Execution
        'execution_success_pct': round(100 * sum(1 for m in metrics if m['execution_success']) / n, 2),
        'avg_tests_passed': round(sum(m['tests_passed'] for m in metrics) / n, 2),
        
        # Coverage
        'avg_line_coverage': round(sum(m['line_coverage'] for m in metrics) / n, 2),
        'avg_branch_coverage': round(sum(m['branch_coverage'] for m in metrics) / n, 2),
        
        # Quality
        'has_assertions_pct': round(100 * sum(1 for m in metrics if m['has_assertions']) / n, 2),
        'avg_relevance': round(sum(m['relevance_score'] for m in metrics) / n, 4),
        
        # Composite
        'valid_useful_pct': round(100 * sum(1 for m in metrics 
            if m['syntax_valid'] and m['has_assertions']) / n, 2),
        'executable_pct': round(100 * sum(1 for m in metrics 
            if m['syntax_valid'] and m['execution_success']) / n, 2),
    }
    
    return summary

def print_full_summary(s: dict):
    print("\n" + "="*60)
    print("📊 FULL EVALUATION SUMMARY")
    print("="*60)
    
    print(f"\n⏱️ Time: {s['total_time_minutes']} min for {s['total_instances']} instances")
    
    print(f"\n📝 SYNTAX & COMPILE:")
    print(f"   Syntax valid: {s['syntax_valid_pct']}%")
    print(f"   Compile success: {s['compile_success_pct']}%")
    
    print(f"\n⚙️ EXECUTION:")
    print(f"   Execution success: {s['execution_success_pct']}%")
    print(f"   Avg tests passed: {s['avg_tests_passed']}")
    
    print(f"\n📈 COVERAGE:")
    print(f"   Line coverage: {s['avg_line_coverage']}%")
    print(f"   Branch coverage: {s['avg_branch_coverage']}%")
    
    print(f"\n🎯 QUALITY:")
    print(f"   Has assertions: {s['has_assertions_pct']}%")
    print(f"   Relevance: {s['avg_relevance']:.2%}")
    
    print(f"\n⭐ FINAL SCORES:")
    print(f"   Valid & Useful: {s['valid_useful_pct']}%")
    print(f"   Executable: {s['executable_pct']}%")

# ============================================================
# CELL 12: Save Results
# ============================================================

def save_results(metrics, tests, summary, batch_name=None):
    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    
    # Use batch name in filenames
    suffix = f"_{batch_name}" if batch_name else ""
    
    with open(f"{OUTPUT_DIR}/summary{suffix}.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    with open(f"{OUTPUT_DIR}/detailed_metrics{suffix}.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    
    with open(f"{OUTPUT_DIR}/generated_tests{suffix}.jsonl", 'w') as f:
        for test in tests:
            f.write(json.dumps(test) + '\n')
    
    # Also save to Drive if configured
    if SAVE_TO_DRIVE:
        Path(SAVE_TO_DRIVE).mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copytree(OUTPUT_DIR, SAVE_TO_DRIVE, dirs_exist_ok=True)
        print(f"✅ Also saved to Drive: {SAVE_TO_DRIVE}")
    
    print(f"✅ Results saved to {OUTPUT_DIR}/{suffix}")

# ============================================================
# CELL 13: Run Everything
# ============================================================

print("Loading model...")
model, tokenizer = load_model()

print(f"\nRunning batch evaluation: {BATCH_NAME}...")
metrics, tests, total_time = run_full_evaluation(
    model, tokenizer, 
    start_from=START_FROM, 
    batch_size=BATCH_SIZE
)

print("\nCalculating summary...")
summary = calculate_full_summary(metrics, total_time)
summary['batch_name'] = BATCH_NAME
summary['start_from'] = START_FROM
summary['batch_size'] = BATCH_SIZE

print_full_summary(summary)

print("\nSaving results...")
save_results(metrics, tests, summary, BATCH_NAME)

print(f"\n🎉 BATCH {BATCH_NAME} COMPLETE!")

# ============================================================
# CELL 14: Merge Batches (Run after all batches complete)
# ============================================================

def merge_batches(output_dir, batch_names):
    """Merge results from multiple batch runs."""
    
    all_metrics = []
    all_tests = []
    
    print("Merging batches...")
    
    for batch in batch_names:
        # Load metrics
        metrics_file = f"{output_dir}/detailed_metrics_{batch}.json"
        if os.path.exists(metrics_file):
            with open(metrics_file) as f:
                all_metrics.extend(json.load(f))
            print(f"  ✅ Loaded {batch}")
        else:
            print(f"  ⚠️ Missing {batch}")
        
        # Load tests
        tests_file = f"{output_dir}/generated_tests_{batch}.jsonl"
        if os.path.exists(tests_file):
            with open(tests_file) as f:
                for line in f:
                    all_tests.append(json.loads(line))
    
    print(f"\nTotal instances: {len(all_metrics)}")
    
    # Calculate combined summary
    if all_metrics:
        summary = calculate_full_summary(all_metrics, 0)
        summary['total_batches'] = len(batch_names)
        summary['merged'] = True
        
        # Save merged results
        with open(f"{output_dir}/summary_merged.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        with open(f"{output_dir}/detailed_metrics_merged.json", 'w') as f:
            json.dump(all_metrics, f, indent=2)
        
        with open(f"{output_dir}/generated_tests_merged.jsonl", 'w') as f:
            for test in all_tests:
                f.write(json.dumps(test) + '\n')
        
        print_full_summary(summary)
        print(f"\n✅ Merged results saved!")
        
        return summary
    
    return None

# To merge batches after all sessions complete:
"""
batch_names = ['batch_0_100', 'batch_100_200', 'batch_200_300']
merge_batches(OUTPUT_DIR, batch_names)
# Or from Drive:
merge_batches(SAVE_TO_DRIVE, batch_names)
"""

# ============================================================
# CELL 15: Download Results
# ============================================================
"""
# Download current batch
!zip -r /content/testgen_{BATCH_NAME}.zip {OUTPUT_DIR}/
from google.colab import files
files.download(f'/content/testgen_{BATCH_NAME}.zip')

# After merging all batches:
# !zip -r /content/testgen_full_eval.zip {OUTPUT_DIR}/
# files.download('/content/testgen_full_eval.zip')
"""

# ============================================================
# HOW TO USE BATCHES:
# ============================================================
# Session 1: START_FROM=0,   BATCH_SIZE=100 → batch_0_100
# Session 2: START_FROM=100, BATCH_SIZE=100 → batch_100_200  
# Session 3: START_FROM=200, BATCH_SIZE=100 → batch_200_300
#
# After all sessions, run merge_batches() to combine results!
# ============================================================

