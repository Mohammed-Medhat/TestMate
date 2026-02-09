# ============================================================
# COMPREHENSIVE TEST GENERATION EVALUATION
# ============================================================
# Academic-grade evaluation with metrics used in research papers:
# 1. Syntax & Compilation Metrics
# 2. Execution Metrics (requires local execution)
# 3. Coverage Metrics (line, branch, statement)
# 4. Mutation Score (fault detection)
# 5. Quality Metrics (assertions, relevance)
# ============================================================
# NOTE: Full evaluation requires execution environment (not Kaggle)
# This script is designed to run locally with Docker or pytest
# ============================================================

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
from typing import List, Dict, Optional, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset

# ============================================================
# CONFIGURATION
# ============================================================

@dataclass
class EvalConfig:
    """Evaluation configuration."""
    base_model: str = "Qwen/Qwen2.5-Coder-7B"
    lora_path: str = "./testgen_model/final"  # Update this!
    output_dir: str = "./testgen_full_evaluation"
    max_instances: int = 300
    run_execution: bool = True  # Set False on Kaggle
    run_coverage: bool = True   # Requires coverage.py
    run_mutation: bool = False  # Requires mutmut, very slow
    timeout_seconds: int = 30

config = EvalConfig()

# ============================================================
# METRICS (Based on TestEval & Mutation 2024 papers)
# ============================================================

@dataclass
class TestMetrics:
    """Comprehensive test evaluation metrics."""
    instance_id: str
    repo: str
    
    # 1. Syntax Metrics
    syntax_valid: bool = False
    has_imports: bool = False
    has_test_function: bool = False
    has_assertions: bool = False
    assertion_count: int = 0
    
    # 2. Compilation Metrics
    compile_success: bool = False
    compile_error: str = ""
    
    # 3. Execution Metrics (requires running the test)
    execution_success: bool = False
    execution_error: str = ""
    execution_time_ms: float = 0.0
    
    # 4. Coverage Metrics (requires coverage.py)
    line_coverage: float = 0.0
    branch_coverage: float = 0.0
    statement_coverage: float = 0.0
    
    # 5. Mutation Score (requires mutmut)
    mutation_score: float = 0.0
    mutants_killed: int = 0
    mutants_total: int = 0
    
    # 6. Quality Metrics
    relevance_score: float = 0.0  # Word overlap with bug description
    test_length: int = 0
    test_complexity: int = 0  # McCabe complexity
    
    # 7. Generation Metrics
    generation_time_ms: float = 0.0
    
    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}

# ============================================================
# METRIC CALCULATORS
# ============================================================

def check_syntax(test_code: str) -> Tuple[bool, str]:
    """Check if test code is syntactically valid Python."""
    try:
        ast.parse(test_code)
        return True, ""
    except SyntaxError as e:
        return False, str(e)

def count_assertions(test_code: str) -> int:
    """Count number of assertions in test code."""
    patterns = [
        r'\bassert\b',
        r'\.assert\w+\(',  # unittest style
        r'pytest\.raises',
        r'self\.assert\w+',
    ]
    count = 0
    for pattern in patterns:
        count += len(re.findall(pattern, test_code))
    return count

def calculate_complexity(test_code: str) -> int:
    """Calculate McCabe cyclomatic complexity."""
    try:
        tree = ast.parse(test_code)
        complexity = 1  # Base complexity
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1
        
        return complexity
    except:
        return 0

def calculate_relevance(test_code: str, problem_statement: str) -> float:
    """Calculate relevance score based on keyword overlap."""
    # Tokenize
    stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'to', 'in', 
                  'on', 'for', 'of', 'and', 'or', 'this', 'that', 'it', 'be'}
    
    problem_words = set(re.findall(r'\w+', problem_statement.lower())) - stop_words
    test_words = set(re.findall(r'\w+', test_code.lower())) - stop_words
    
    if not problem_words:
        return 0.0
    
    overlap = len(problem_words & test_words)
    return round(overlap / len(problem_words), 4)

def compile_test(test_code: str) -> Tuple[bool, str]:
    """Try to compile the test code."""
    try:
        compile(test_code, '<test>', 'exec')
        return True, ""
    except Exception as e:
        return False, str(e)

def execute_test(test_code: str, timeout: int = 30) -> Tuple[bool, str, float]:
    """
    Execute the test code and return success status.
    
    Returns: (success, error_message, execution_time_ms)
    """
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(test_code)
        f.flush()
        temp_path = f.name
    
    try:
        start = time.time()
        result = subprocess.run(
            ['python', '-m', 'pytest', temp_path, '-v', '--tb=short'],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        exec_time = (time.time() - start) * 1000
        
        success = result.returncode == 0
        error = result.stderr if not success else ""
        
        return success, error, exec_time
        
    except subprocess.TimeoutExpired:
        return False, "Timeout", timeout * 1000
    except Exception as e:
        return False, str(e), 0.0
    finally:
        os.unlink(temp_path)

def run_coverage(test_code: str, target_module: str = None) -> Dict[str, float]:
    """
    Run test with coverage.py and return coverage metrics.
    
    Returns: {line_coverage, branch_coverage, statement_coverage}
    """
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(test_code)
        f.flush()
        temp_path = f.name
    
    try:
        # Run with coverage
        result = subprocess.run(
            ['python', '-m', 'coverage', 'run', '--branch', '-m', 'pytest', temp_path],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        # Get coverage report
        report = subprocess.run(
            ['python', '-m', 'coverage', 'json', '-o', '/tmp/cov.json'],
            capture_output=True,
            text=True
        )
        
        # Parse coverage
        if os.path.exists('/tmp/cov.json'):
            with open('/tmp/cov.json') as f:
                cov_data = json.load(f)
            
            totals = cov_data.get('totals', {})
            return {
                'line_coverage': totals.get('percent_covered', 0.0),
                'branch_coverage': totals.get('percent_covered_branches', 0.0),
                'statement_coverage': totals.get('percent_covered', 0.0)
            }
        
        return {'line_coverage': 0.0, 'branch_coverage': 0.0, 'statement_coverage': 0.0}
        
    except Exception as e:
        return {'line_coverage': 0.0, 'branch_coverage': 0.0, 'statement_coverage': 0.0}
    finally:
        os.unlink(temp_path)

def run_mutation_testing(test_code: str, target_file: str) -> Dict[str, any]:
    """
    Run mutation testing with mutmut.
    
    Returns: {mutation_score, mutants_killed, mutants_total}
    """
    # This requires the actual source code to mutate
    # Simplified version - would need full setup in practice
    try:
        result = subprocess.run(
            ['mutmut', 'run', '--paths-to-mutate', target_file],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        # Parse results
        results = subprocess.run(
            ['mutmut', 'results'],
            capture_output=True,
            text=True
        )
        
        # Extract killed/total from output
        output = results.stdout
        killed_match = re.search(r'killed:\s*(\d+)', output)
        total_match = re.search(r'total:\s*(\d+)', output)
        
        killed = int(killed_match.group(1)) if killed_match else 0
        total = int(total_match.group(1)) if total_match else 1
        
        return {
            'mutation_score': killed / total if total > 0 else 0.0,
            'mutants_killed': killed,
            'mutants_total': total
        }
        
    except Exception as e:
        return {'mutation_score': 0.0, 'mutants_killed': 0, 'mutants_total': 0}

# ============================================================
# MAIN EVALUATION
# ============================================================

def evaluate_single_test(
    test_code: str,
    instance_id: str,
    repo: str,
    problem_statement: str,
    generation_time_ms: float,
    config: EvalConfig
) -> TestMetrics:
    """Evaluate a single generated test case."""
    
    metrics = TestMetrics(instance_id=instance_id, repo=repo)
    metrics.generation_time_ms = generation_time_ms
    metrics.test_length = len(test_code)
    
    # 1. Syntax Metrics
    metrics.syntax_valid, _ = check_syntax(test_code)
    metrics.has_imports = 'import' in test_code
    metrics.has_test_function = bool(re.search(r'def test_\w+', test_code))
    metrics.has_assertions = 'assert' in test_code
    metrics.assertion_count = count_assertions(test_code)
    
    # 2. Compilation Metrics
    metrics.compile_success, metrics.compile_error = compile_test(test_code)
    
    # 3. Complexity & Relevance
    metrics.test_complexity = calculate_complexity(test_code)
    metrics.relevance_score = calculate_relevance(test_code, problem_statement)
    
    # 4. Execution (if enabled)
    if config.run_execution and metrics.compile_success:
        metrics.execution_success, metrics.execution_error, metrics.execution_time_ms = \
            execute_test(test_code, config.timeout_seconds)
    
    # 5. Coverage (if enabled)
    if config.run_coverage and metrics.execution_success:
        coverage = run_coverage(test_code)
        metrics.line_coverage = coverage['line_coverage']
        metrics.branch_coverage = coverage['branch_coverage']
        metrics.statement_coverage = coverage['statement_coverage']
    
    # 6. Mutation (if enabled - very slow)
    if config.run_mutation and metrics.execution_success:
        mutation = run_mutation_testing(test_code, f"/tmp/{repo.replace('/', '_')}.py")
        metrics.mutation_score = mutation['mutation_score']
        metrics.mutants_killed = mutation['mutants_killed']
        metrics.mutants_total = mutation['mutants_total']
    
    return metrics

def generate_test(model, tokenizer, problem_statement: str, repo: str) -> Tuple[str, float]:
    """Generate a test case and return (test_code, generation_time_ms)."""
    
    prompt = f"""=== TEST CASE GENERATION ===

## Bug Report
Repository: {repo}

{problem_statement[:1500]}

## Task
Generate a Python test case that reproduces this bug. The test should:
1. Set up the necessary conditions
2. Call the affected function/method
3. Assert the expected behavior

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

def run_full_evaluation(config: EvalConfig):
    """Run comprehensive evaluation on SWE-bench Lite."""
    
    print("="*60)
    print("COMPREHENSIVE TEST GENERATION EVALUATION")
    print("="*60)
    print(f"\nMetrics enabled:")
    print(f"  - Execution: {config.run_execution}")
    print(f"  - Coverage: {config.run_coverage}")
    print(f"  - Mutation: {config.run_mutation}")
    
    # Load model
    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(config.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    base_model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )
    
    model = PeftModel.from_pretrained(base_model, config.lora_path)
    print("✅ Model loaded!")
    
    # Load dataset
    print("\nLoading SWE-bench Lite...")
    dataset = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
    print(f"Total instances: {len(dataset)}")
    
    # Evaluate
    all_metrics = []
    generated_tests = []
    
    start_time = time.time()
    
    for i, instance in enumerate(dataset):
        if i >= config.max_instances:
            break
        
        instance_id = instance.get('instance_id', f'instance_{i}')
        repo = instance.get('repo', 'unknown')
        problem = instance.get('problem_statement', '')
        
        print(f"\n[{i+1}/{min(len(dataset), config.max_instances)}] {instance_id}")
        
        try:
            # Generate test
            test_code, gen_time = generate_test(model, tokenizer, problem, repo)
            
            # Evaluate
            metrics = evaluate_single_test(
                test_code=test_code,
                instance_id=instance_id,
                repo=repo,
                problem_statement=problem,
                generation_time_ms=gen_time,
                config=config
            )
            
            all_metrics.append(metrics.to_dict())
            generated_tests.append({
                'instance_id': instance_id,
                'test_code': test_code,
                'metrics': metrics.to_dict()
            })
            
            # Print status
            status = "✅" if metrics.syntax_valid and metrics.has_assertions else "⚠️"
            print(f"   {status} syntax={metrics.syntax_valid}, exec={metrics.execution_success}, "
                  f"asserts={metrics.assertion_count}, relevance={metrics.relevance_score:.2f}")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    total_time = time.time() - start_time
    
    # Calculate aggregate metrics
    summary = calculate_summary(all_metrics, total_time)
    
    # Print summary
    print_summary(summary)
    
    # Save results
    save_results(all_metrics, generated_tests, summary, config)
    
    return all_metrics, summary

def calculate_summary(all_metrics: List[dict], total_time: float) -> dict:
    """Calculate aggregate summary metrics."""
    
    n = len(all_metrics)
    if n == 0:
        return {}
    
    summary = {
        'total_instances': n,
        'total_time_minutes': round(total_time / 60, 2),
        'avg_generation_time_ms': round(sum(m['generation_time_ms'] for m in all_metrics) / n, 2),
        
        # Syntax Metrics
        'syntax_valid_count': sum(1 for m in all_metrics if m['syntax_valid']),
        'syntax_valid_pct': round(100 * sum(1 for m in all_metrics if m['syntax_valid']) / n, 2),
        
        'has_test_function_count': sum(1 for m in all_metrics if m['has_test_function']),
        'has_test_function_pct': round(100 * sum(1 for m in all_metrics if m['has_test_function']) / n, 2),
        
        'has_assertions_count': sum(1 for m in all_metrics if m['has_assertions']),
        'has_assertions_pct': round(100 * sum(1 for m in all_metrics if m['has_assertions']) / n, 2),
        
        'avg_assertion_count': round(sum(m['assertion_count'] for m in all_metrics) / n, 2),
        
        # Compilation
        'compile_success_count': sum(1 for m in all_metrics if m['compile_success']),
        'compile_success_pct': round(100 * sum(1 for m in all_metrics if m['compile_success']) / n, 2),
        
        # Execution
        'execution_success_count': sum(1 for m in all_metrics if m['execution_success']),
        'execution_success_pct': round(100 * sum(1 for m in all_metrics if m['execution_success']) / n, 2),
        
        # Coverage (average of successful tests)
        'avg_line_coverage': round(sum(m['line_coverage'] for m in all_metrics) / n, 2),
        'avg_branch_coverage': round(sum(m['branch_coverage'] for m in all_metrics) / n, 2),
        
        # Mutation
        'avg_mutation_score': round(sum(m['mutation_score'] for m in all_metrics) / n, 4),
        
        # Quality
        'avg_relevance_score': round(sum(m['relevance_score'] for m in all_metrics) / n, 4),
        'avg_test_complexity': round(sum(m['test_complexity'] for m in all_metrics) / n, 2),
        
        # Composite Scores
        'valid_and_useful': sum(1 for m in all_metrics if m['syntax_valid'] and m['has_assertions']),
        'valid_and_useful_pct': round(100 * sum(1 for m in all_metrics if m['syntax_valid'] and m['has_assertions']) / n, 2),
    }
    
    return summary

def print_summary(summary: dict):
    """Print formatted summary."""
    
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    
    print(f"\n📊 GENERATION STATS:")
    print(f"   Total instances: {summary['total_instances']}")
    print(f"   Total time: {summary['total_time_minutes']} minutes")
    print(f"   Avg generation time: {summary['avg_generation_time_ms']}ms")
    
    print(f"\n📝 SYNTAX METRICS:")
    print(f"   Syntax valid: {summary['syntax_valid_count']}/{summary['total_instances']} ({summary['syntax_valid_pct']}%)")
    print(f"   Has test function: {summary['has_test_function_count']}/{summary['total_instances']} ({summary['has_test_function_pct']}%)")
    print(f"   Has assertions: {summary['has_assertions_count']}/{summary['total_instances']} ({summary['has_assertions_pct']}%)")
    print(f"   Avg assertions per test: {summary['avg_assertion_count']}")
    
    print(f"\n⚙️ EXECUTION METRICS:")
    print(f"   Compile success: {summary['compile_success_count']}/{summary['total_instances']} ({summary['compile_success_pct']}%)")
    print(f"   Execution success: {summary['execution_success_count']}/{summary['total_instances']} ({summary['execution_success_pct']}%)")
    
    print(f"\n📈 COVERAGE METRICS:")
    print(f"   Avg line coverage: {summary['avg_line_coverage']}%")
    print(f"   Avg branch coverage: {summary['avg_branch_coverage']}%")
    
    print(f"\n🧬 MUTATION SCORE:")
    print(f"   Avg mutation score: {summary['avg_mutation_score']}")
    
    print(f"\n⭐ QUALITY METRICS:")
    print(f"   Avg relevance: {summary['avg_relevance_score']}")
    print(f"   Avg complexity: {summary['avg_test_complexity']}")
    
    print(f"\n🎯 FINAL SCORE:")
    print(f"   Valid & Useful: {summary['valid_and_useful']}/{summary['total_instances']} ({summary['valid_and_useful_pct']}%)")

def save_results(all_metrics, generated_tests, summary, config):
    """Save all evaluation results."""
    
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Summary
    with open(f"{config.output_dir}/summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Detailed metrics
    with open(f"{config.output_dir}/detailed_metrics.json", 'w') as f:
        json.dump(all_metrics, f, indent=2)
    
    # Generated tests
    with open(f"{config.output_dir}/generated_tests.jsonl", 'w') as f:
        for test in generated_tests:
            f.write(json.dumps(test) + '\n')
    
    print(f"\n✅ Results saved to {config.output_dir}/")

# ============================================================
# RUN EVALUATION
# ============================================================

if __name__ == "__main__":
    # Update config as needed
    config = EvalConfig(
        lora_path="./testgen_model/final",  # Update this!
        max_instances=300,
        run_execution=True,  # Set False on Kaggle
        run_coverage=False,  # Enable if coverage.py installed
        run_mutation=False   # Very slow, enable only if needed
    )
    
    all_metrics, summary = run_full_evaluation(config)
    
    print("\n🎉 COMPREHENSIVE EVALUATION COMPLETE!")
