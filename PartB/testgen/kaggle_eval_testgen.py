# ============================================================
# KAGGLE NOTEBOOK: Research-Grade Test Generation Evaluation
# ============================================================
# All metrics that can run WITHOUT requiring actual repos
# Based on TestEval (2024) and Mutation Analysis papers
# NOW WITH GRAPH-RAG INTEGRATION!
# ============================================================

# CELL 1: Install dependencies
"""
!pip install -q transformers datasets peft accelerate
"""

# CELL 2: Imports
import torch
import json
import time
import ast
import re
import pickle
import os
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, Dict, List
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset

# Config
BASE_MODEL = "Qwen/Qwen2.5-Coder-7B"
LORA_PATH = "/kaggle/input/testgen-model/testgen_model/final"  # Update this!
OUTPUT_DIR = "/kaggle/working/testgen_evaluation"
MAX_INSTANCES = 300
GRAPH_PATH = "/kaggle/input/testmate-graph/knowledge_graph.pkl"  # Upload your graph!

print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ============================================================
# CELL 2.5: Knowledge Graph Loader (THE KEY FIX!)
# ============================================================

class GraphContextExtractor:
    """Extract 2-hop context + CFG paths from Knowledge Graph."""
    
    def __init__(self, graph_path: str = None):
        self.graph = None
        if graph_path and os.path.exists(graph_path):
            self.load_graph(graph_path)
    
    def load_graph(self, path: str):
        try:
            with open(path, 'rb') as f:
                self.graph = pickle.load(f)
            print(f"✅ Loaded Graph: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
        except Exception as e:
            print(f"⚠️ Could not load graph: {e}")
            self.graph = None
    
    def get_2hop_context(self, function_name: str) -> dict:
        """Get 2-hop neighborhood + CFG paths for a function."""
        if not self.graph or function_name not in self.graph.nodes():
            return None
        
        func_node = self.graph.nodes[function_name]
        func_data = func_node.get('data', {})
        
        context = {
            'callers': [],
            'callees': [],
            'paths': func_data.get('paths', [])  # CFG paths!
        }
        
        # Get callers
        for pred in self.graph.predecessors(function_name):
            edge_data = self.graph[pred][function_name]
            for key, data in edge_data.items():
                if data.get('type') == 'CALLS':
                    node_data = self.graph.nodes[pred].get('data', {})
                    context['callers'].append(node_data.get('signature', pred))
        
        # Get callees
        for succ in self.graph.successors(function_name):
            edge_data = self.graph[function_name][succ]
            for key, data in edge_data.items():
                if data.get('type') == 'CALLS':
                    node_data = self.graph.nodes[succ].get('data', {})
                    context['callees'].append(node_data.get('signature', succ))
        
        return context
    
    def format_graph_context(self, context: dict) -> str:
        """Format context for prompt."""
        if not context:
            return ""
        
        parts = []
        
        # CFG PATHS FIRST!
        if context.get('paths'):
            parts.append("## Execution Paths (MUST COVER):")
            for i, path in enumerate(context['paths'][:5], 1):
                parts.append(f"{i}. {path}")
        
        if context.get('callers'):
            parts.append("\n## Callers:")
            for c in context['callers'][:3]:
                parts.append(f"- {c}")
        
        if context.get('callees'):
            parts.append("\n## Callees:")
            for c in context['callees'][:3]:
                parts.append(f"- {c}")
        
        return "\n".join(parts)

# Initialize graph loader
graph_loader = GraphContextExtractor(GRAPH_PATH)

def extract_function_name(code: str) -> str:
    """Extract function name from code."""
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                return node.name
    except:
        pass
    match = re.search(r'def\s+(\w+)\s*\(', code)
    return match.group(1) if match else None

# ============================================================
# CELL 3: Research Metrics (Kaggle-Compatible)
# ============================================================

@dataclass
class TestMetrics:
    """Research-grade test evaluation metrics."""
    instance_id: str = ""
    repo: str = ""
    
    # Generation
    generation_time_ms: float = 0.0
    test_length: int = 0
    
    # Syntax Metrics (TestEval 2024)
    syntax_valid: bool = False
    syntax_error: str = ""
    
    # Structure Metrics
    has_imports: bool = False
    has_test_function: bool = False
    has_class: bool = False
    has_setup: bool = False
    has_teardown: bool = False
    
    # Assertion Metrics
    has_assertions: bool = False
    assertion_count: int = 0
    assertion_types: List[str] = None  # assert, assertEqual, assertTrue, etc.
    
    # Compilation Metrics
    compile_success: bool = False
    compile_error: str = ""
    
    # Complexity Metrics (McCabe)
    cyclomatic_complexity: int = 0
    num_functions: int = 0
    num_branches: int = 0
    
    # Quality Metrics
    relevance_score: float = 0.0
    keyword_coverage: float = 0.0
    docstring_present: bool = False
    
    # Test Patterns (based on testing best practices)
    follows_aaa_pattern: bool = False  # Arrange-Act-Assert
    has_edge_cases: bool = False
    has_error_handling: bool = False
    
    def __post_init__(self):
        if self.assertion_types is None:
            self.assertion_types = []
    
    def to_dict(self):
        return {k: v for k, v in self.__dict__.items()}

# ============================================================
# CELL 4: Metric Calculators
# ============================================================

def check_syntax(code: str) -> Tuple[bool, str]:
    """Check Python syntax validity."""
    try:
        ast.parse(code)
        return True, ""
    except SyntaxError as e:
        return False, f"Line {e.lineno}: {e.msg}"

def check_compile(code: str) -> Tuple[bool, str]:
    """Check if code compiles."""
    try:
        compile(code, '<test>', 'exec')
        return True, ""
    except Exception as e:
        return False, str(e)

def count_assertions(code: str) -> Tuple[int, List[str]]:
    """Count assertions and identify types."""
    patterns = {
        'assert': r'\bassert\b',
        'assertEqual': r'\.assertEqual\(',
        'assertTrue': r'\.assertTrue\(',
        'assertFalse': r'\.assertFalse\(',
        'assertRaises': r'\.assertRaises\(',
        'assertIn': r'\.assertIn\(',
        'assertIsNone': r'\.assertIsNone\(',
        'assertIsNotNone': r'\.assertIsNotNone\(',
        'pytest.raises': r'pytest\.raises',
        'pytest.approx': r'pytest\.approx',
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
    """Calculate McCabe cyclomatic complexity."""
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
            if isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1
        
        return {
            'cyclomatic_complexity': complexity,
            'num_functions': num_functions,
            'num_branches': num_branches
        }
    except:
        return {'cyclomatic_complexity': 0, 'num_functions': 0, 'num_branches': 0}

def calculate_relevance(test_code: str, problem: str) -> Dict[str, float]:
    """Calculate relevance metrics."""
    stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'to', 'in', 
                  'on', 'for', 'of', 'and', 'or', 'this', 'that', 'it', 'be',
                  'with', 'as', 'at', 'by', 'from', 'not', 'but', 'if', 'when'}
    
    # Extract meaningful words
    problem_words = set(re.findall(r'\b[a-zA-Z_]\w+\b', problem.lower())) - stop_words
    test_words = set(re.findall(r'\b[a-zA-Z_]\w+\b', test_code.lower())) - stop_words
    
    if not problem_words:
        return {'relevance_score': 0.0, 'keyword_coverage': 0.0}
    
    overlap = problem_words & test_words
    
    return {
        'relevance_score': len(overlap) / len(problem_words),
        'keyword_coverage': len(overlap) / max(len(test_words), 1)
    }

# ============================================================
# UPGRADE 1: Semantic Similarity with Embeddings (Research-Grade!)
# ============================================================

try:
    from sentence_transformers import SentenceTransformer, util as st_util
    semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
    SEMANTIC_AVAILABLE = True
    print("✅ Semantic embeddings model loaded!")
except:
    semantic_model = None
    SEMANTIC_AVAILABLE = False
    print("⚠️ sentence-transformers not available, using word overlap")

def calculate_relevance_semantic(test_code: str, problem: str) -> float:
    """
    Calculate semantic similarity using embeddings (Research-Grade).
    This captures meaning, not just keyword overlap!
    """
    if not SEMANTIC_AVAILABLE or not semantic_model:
        # Fallback to word overlap
        result = calculate_relevance(test_code, problem)
        return result['relevance_score']
    
    try:
        # Encode both texts
        embedding1 = semantic_model.encode(test_code[:1000], convert_to_tensor=True)
        embedding2 = semantic_model.encode(problem[:1000], convert_to_tensor=True)
        
        # Compute cosine similarity
        score = st_util.cos_sim(embedding1, embedding2)
        return float(score[0][0])
    except Exception as e:
        # Fallback on error
        result = calculate_relevance(test_code, problem)
        return result['relevance_score']

# ============================================================
# UPGRADE 2: Smart Mock Detection using 2-Hop Graph (Deep Mocks!)
# ============================================================

EXTERNAL_LIBS = {'requests', 'urllib', 'sqlite3', 'pymongo', 'redis', 'boto3', 
                 'smtplib', 'httpx', 'aiohttp', 'psycopg2', 'mysql', 'socket'}

def detect_smart_mocks(func_name: str, code: str) -> list:
    """
    Find mocks even deep inside dependency trees using 2-hop graph lookup.
    This is what makes Graph-RAG special vs standard RAG!
    """
    mocks = set()
    
    # First: Check direct usage in code (basic detection)
    for lib in EXTERNAL_LIBS:
        if lib in code:
            mocks.add(lib)
    
    # Second: Use Graph to find 2-hop dependencies
    if func_name and graph_loader and graph_loader.graph:
        context = graph_loader.get_2hop_context(func_name)
        if context and context.get('callees'):
            for callee in context['callees']:
                callee_str = str(callee).lower()
                for lib in EXTERNAL_LIBS:
                    if lib in callee_str:
                        mocks.add(lib)
                        print(f"   🎯 Smart Mock: Found `{lib}` in 2-hop callee `{callee}`")
    
    return list(mocks)

def check_test_patterns(code: str) -> Dict[str, bool]:
    """Check for testing best practice patterns."""
    
    # AAA Pattern: Arrange-Act-Assert (look for comments or structure)
    has_arrange = bool(re.search(r'#\s*(arrange|setup|given)', code, re.I))
    has_act = bool(re.search(r'#\s*(act|when|exercise)', code, re.I))
    has_assert_comment = bool(re.search(r'#\s*(assert|then|verify)', code, re.I))
    follows_aaa = (has_arrange and has_act) or ('assert' in code.lower())
    
    # Edge cases
    edge_patterns = ['None', 'empty', '[]', '{}', '""', "''", '0', '-1', 'boundary']
    has_edge_cases = any(p in code for p in edge_patterns)
    
    # Error handling
    has_error_handling = 'raises' in code.lower() or 'exception' in code.lower() or 'try:' in code
    
    return {
        'follows_aaa_pattern': follows_aaa,
        'has_edge_cases': has_edge_cases,
        'has_error_handling': has_error_handling
    }

def check_structure(code: str) -> Dict[str, bool]:
    """Check test structure elements."""
    return {
        'has_imports': 'import' in code,
        'has_test_function': bool(re.search(r'def test_\w+', code)),
        'has_class': bool(re.search(r'class \w+Test', code)),
        'has_setup': 'setup' in code.lower() or 'setUp' in code,
        'has_teardown': 'teardown' in code.lower() or 'tearDown' in code,
        'docstring_present': '"""' in code or "'''" in code
    }

# ============================================================
# CELL 5: Main Evaluation Function
# ============================================================

def evaluate_test(test_code: str, instance_id: str, repo: str, 
                  problem: str, gen_time: float) -> TestMetrics:
    """Evaluate a single test with all research metrics."""
    
    metrics = TestMetrics(
        instance_id=instance_id,
        repo=repo,
        generation_time_ms=gen_time,
        test_length=len(test_code)
    )
    
    # Syntax
    metrics.syntax_valid, metrics.syntax_error = check_syntax(test_code)
    
    # Compile
    metrics.compile_success, metrics.compile_error = check_compile(test_code)
    
    # Structure
    structure = check_structure(test_code)
    metrics.has_imports = structure['has_imports']
    metrics.has_test_function = structure['has_test_function']
    metrics.has_class = structure['has_class']
    metrics.has_setup = structure['has_setup']
    metrics.has_teardown = structure['has_teardown']
    metrics.docstring_present = structure['docstring_present']
    
    # Assertions
    metrics.assertion_count, metrics.assertion_types = count_assertions(test_code)
    metrics.has_assertions = metrics.assertion_count > 0
    
    # Complexity
    complexity = calculate_complexity(test_code)
    metrics.cyclomatic_complexity = complexity['cyclomatic_complexity']
    metrics.num_functions = complexity['num_functions']
    metrics.num_branches = complexity['num_branches']
    
    # Relevance
    relevance = calculate_relevance(test_code, problem)
    metrics.relevance_score = relevance['relevance_score']
    metrics.keyword_coverage = relevance['keyword_coverage']
    
    # Patterns
    patterns = check_test_patterns(test_code)
    metrics.follows_aaa_pattern = patterns['follows_aaa_pattern']
    metrics.has_edge_cases = patterns['has_edge_cases']
    metrics.has_error_handling = patterns['has_error_handling']
    
    return metrics

# ============================================================
# CELL 6: Load Model
# ============================================================

def load_model():
    """Load test generation model."""
    print("Loading model...")
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )
    
    print(f"Loading LoRA from {LORA_PATH}...")
    model = PeftModel.from_pretrained(base_model, LORA_PATH)
    
    print("✅ Model loaded!")
    return model, tokenizer

# ============================================================
# CELL 7: Generate Test (with RAG context)
# ============================================================

def extract_paths_from_code(code: str) -> list:
    """FALLBACK: Extract execution paths from code using AST (used when graph has no match)."""
    paths = []
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.If):
                cond = ast.unparse(node.test) if hasattr(ast, 'unparse') else "condition"
                paths.append(f"if {cond}")
            elif isinstance(node, ast.For):
                paths.append("loop iteration")
            elif isinstance(node, ast.While):
                paths.append("while loop")
    except:
        pass
    return paths[:5]

def detect_external_deps(code: str) -> list:
    """Detect external dependencies that need mocking."""
    EXTERNAL = ['requests', 'urllib', 'sqlite3', 'pymongo', 'redis', 'boto3', 'smtplib']
    deps = []
    for mod in EXTERNAL:
        if mod in code:
            deps.append(mod)
    return deps

def check_syntax_valid(code: str) -> tuple:
    """Check if code has valid syntax."""
    try:
        ast.parse(code)
        return True, ""
    except SyntaxError as e:
        return False, str(e)

def generate_test(model, tokenizer, code_or_problem: str, repo: str, 
                  max_retries: int = 3) -> Tuple[str, float, int]:
    """Generate comprehensive test case with GRAPH-RAG context and self-correction."""
    
    # === GRAPH-RAG LOOKUP (THE FIX!) ===
    func_name = extract_function_name(code_or_problem)
    graph_context_str = ""
    paths = []
    graph_hit = False
    
    if func_name and graph_loader.graph:
        context_data = graph_loader.get_2hop_context(func_name)
        if context_data:
            paths = context_data.get('paths', [])
            graph_context_str = graph_loader.format_graph_context(context_data)
            graph_hit = True
            
            # === TRACEABILITY LOG FOR DEMO ===
            if paths:
                print(f"   ✨ Graph-RAG hit for `{func_name}`: Found {len(paths)} execution paths!")
                print(f"      1. {paths[0]}")
                if len(paths) > 1:
                    print(f"      2. {paths[1]} ...")
    
    # Fallback to local extraction if no graph match
    if not paths:
        paths = extract_paths_from_code(code_or_problem)
        if paths:
            print(f"   📝 Local extraction: Found {len(paths)} paths")
    
    mocks = detect_smart_mocks(func_name, code_or_problem)  # UPGRADE: 2-hop mocks!
    
    # Build path section (prioritize graph paths)
    path_section = ""
    if graph_context_str:
        path_section = f"\n{graph_context_str}\n"
    elif paths:
        path_section = "\n## Paths to Cover\n" + "\n".join(f"- {p}" for p in paths)
    
    # Build mock section  
    mock_section = ""
    if mocks:
        mock_section = "\n## Required Mocks\n" + "\n".join(f"- @patch(\"{m}.get\")" for m in mocks)
    
    prompt = f"""=== COMPREHENSIVE TEST GENERATION ===

## Source Code / Bug Description
Repository: {repo}

{code_or_problem[:2000]}

## Context
Generate comprehensive unit tests for the above code.
{path_section}
{mock_section}

## Testing Requirements
1. Basic functionality tests
2. Edge cases (None, empty, boundary values)  
3. Error handling (exceptions)
4. Path coverage (test each branch)

## Generated Tests
```python
"""
    
    start = time.time()
    retry_count = 0
    test_code = ""
    
    # Self-correction loop
    for attempt in range(max_retries):
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
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        test_code = response[len(prompt):]
        
        if "```" in test_code:
            test_code = test_code[:test_code.find("```")]
        
        test_code = test_code.strip()
        
        # Check syntax
        is_valid, error = check_syntax_valid(test_code)
        if is_valid:
            break
        
        retry_count += 1
        # Add error feedback to prompt for next attempt
        prompt += f"\n# Previous attempt had syntax error: {error}\n# Please fix:\n"
    
    gen_time = (time.time() - start) * 1000
    
    return test_code, gen_time, retry_count

# ============================================================
# CELL 8: Run Full Evaluation
# ============================================================

def run_evaluation(model, tokenizer, max_instances=300):
    """Run research-grade evaluation."""
    
    print("="*60)
    print("RESEARCH-GRADE TEST GENERATION EVALUATION")
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
            # Now returns (test_code, gen_time, retry_count)
            test_code, gen_time, retry_count = generate_test(model, tokenizer, problem, repo)
            metrics = evaluate_test(test_code, instance_id, repo, problem, gen_time)
            
            # Add retry info to metrics
            metrics_dict = metrics.to_dict()
            metrics_dict['retry_count'] = retry_count
            
            all_metrics.append(metrics_dict)
            generated_tests.append({
                'instance_id': instance_id,
                'test_code': test_code,
                'metrics': metrics_dict,
                'retry_count': retry_count
            })
            
            # Status
            retry_str = f"retries={retry_count}" if retry_count > 0 else ""
            status = "✅" if metrics.syntax_valid and metrics.has_assertions else "⚠️"
            print(f"   {status} syntax={metrics.syntax_valid}, asserts={metrics.assertion_count}, "
                  f"relevance={metrics.relevance_score:.2f} {retry_str}")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    total_time = time.time() - start_time
    
    return all_metrics, generated_tests, total_time

# ============================================================
# CELL 9: Calculate Summary
# ============================================================

def calculate_summary(metrics: List[dict], total_time: float) -> dict:
    """Calculate aggregate summary."""
    
    n = len(metrics)
    
    summary = {
        'total_instances': n,
        'total_time_minutes': round(total_time / 60, 2),
        
        # Syntax & Compilation (TestEval metrics)
        'syntax_valid_pct': round(100 * sum(1 for m in metrics if m['syntax_valid']) / n, 2),
        'compile_success_pct': round(100 * sum(1 for m in metrics if m['compile_success']) / n, 2),
        
        # Structure
        'has_test_function_pct': round(100 * sum(1 for m in metrics if m['has_test_function']) / n, 2),
        'has_imports_pct': round(100 * sum(1 for m in metrics if m['has_imports']) / n, 2),
        'has_docstring_pct': round(100 * sum(1 for m in metrics if m['docstring_present']) / n, 2),
        
        # Assertions
        'has_assertions_pct': round(100 * sum(1 for m in metrics if m['has_assertions']) / n, 2),
        'avg_assertion_count': round(sum(m['assertion_count'] for m in metrics) / n, 2),
        
        # Complexity
        'avg_complexity': round(sum(m['cyclomatic_complexity'] for m in metrics) / n, 2),
        'avg_functions': round(sum(m['num_functions'] for m in metrics) / n, 2),
        
        # Quality
        'avg_relevance': round(sum(m['relevance_score'] for m in metrics) / n, 4),
        'avg_keyword_coverage': round(sum(m['keyword_coverage'] for m in metrics) / n, 4),
        
        # Patterns
        'follows_aaa_pct': round(100 * sum(1 for m in metrics if m['follows_aaa_pattern']) / n, 2),
        'has_edge_cases_pct': round(100 * sum(1 for m in metrics if m['has_edge_cases']) / n, 2),
        'has_error_handling_pct': round(100 * sum(1 for m in metrics if m['has_error_handling']) / n, 2),
        
        # Self-Correction Statistics
        'avg_retry_count': round(sum(m.get('retry_count', 0) for m in metrics) / n, 2),
        'needed_retry_pct': round(100 * sum(1 for m in metrics if m.get('retry_count', 0) > 0) / n, 2),
        
        # Composite Scores
        'valid_useful_pct': round(100 * sum(1 for m in metrics if m['syntax_valid'] and m['has_assertions']) / n, 2),
        'high_quality_pct': round(100 * sum(1 for m in metrics if 
            m['syntax_valid'] and m['has_assertions'] and m['relevance_score'] > 0.1) / n, 2),
    }
    
    return summary

# ============================================================
# CELL 10: Print & Save Results
# ============================================================

def print_summary(summary: dict):
    """Print formatted summary."""
    
    print("\n" + "="*60)
    print("📊 EVALUATION SUMMARY (Research Metrics)")
    print("="*60)
    
    print(f"\n⏱️ GENERATION:")
    print(f"   Instances: {summary['total_instances']}")
    print(f"   Time: {summary['total_time_minutes']} min")
    
    print(f"\n📝 SYNTAX (TestEval):")
    print(f"   Valid: {summary['syntax_valid_pct']}%")
    print(f"   Compile: {summary['compile_success_pct']}%")
    
    print(f"\n🏗️ STRUCTURE:")
    print(f"   Test function: {summary['has_test_function_pct']}%")
    print(f"   Imports: {summary['has_imports_pct']}%")
    print(f"   Docstring: {summary['has_docstring_pct']}%")
    
    print(f"\n✓ ASSERTIONS:")
    print(f"   Has assertions: {summary['has_assertions_pct']}%")
    print(f"   Avg count: {summary['avg_assertion_count']}")
    
    print(f"\n📐 COMPLEXITY:")
    print(f"   Avg McCabe: {summary['avg_complexity']}")
    print(f"   Avg functions: {summary['avg_functions']}")
    
    print(f"\n🎯 QUALITY:")
    print(f"   Relevance: {summary['avg_relevance']:.2%}")
    print(f"   Keyword coverage: {summary['avg_keyword_coverage']:.2%}")
    
    print(f"\n🧪 PATTERNS:")
    print(f"   AAA pattern: {summary['follows_aaa_pct']}%")
    print(f"   Edge cases: {summary['has_edge_cases_pct']}%")
    print(f"   Error handling: {summary['has_error_handling_pct']}%")
    
    print(f"\n⭐ FINAL SCORES:")
    print(f"   Valid & Useful: {summary['valid_useful_pct']}%")
    print(f"   High Quality: {summary['high_quality_pct']}%")

def save_results(metrics, tests, summary):
    """Save all results."""
    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    
    with open(f"{OUTPUT_DIR}/summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    with open(f"{OUTPUT_DIR}/detailed_metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    
    with open(f"{OUTPUT_DIR}/generated_tests.jsonl", 'w') as f:
        for test in tests:
            f.write(json.dumps(test) + '\n')
    
    print(f"\n✅ Saved to {OUTPUT_DIR}/")

# ============================================================
# UPGRADE 3: A/B Test Comparison (For Seminar Demo!)
# ============================================================

def run_ab_comparison(model, tokenizer, code_snippet: str, func_name: str = None):
    """
    Run side-by-side comparison: With Graph RAG vs Without.
    Perfect for seminar slides to show the difference!
    """
    print("=" * 60)
    print("A/B TEST: Graph RAG vs Standard RAG")
    print("=" * 60)
    
    if not func_name:
        func_name = extract_function_name(code_snippet)
    
    # === TEST A: WITHOUT Graph RAG ===
    print("\n📋 TEST A: Standard RAG (No Graph Context)")
    print("-" * 40)
    
    prompt_a = f"""Generate a unit test for:
```python
{code_snippet}
```"""
    
    inputs = tokenizer(prompt_a, return_tensors="pt", truncation=True, max_length=1500)
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=500,
            temperature=0.7,
            do_sample=True
        )
    
    test_a = tokenizer.decode(outputs[0], skip_special_tokens=True)
    test_a = test_a[len(prompt_a):].strip()
    
    print(f"Generated test (without graph):\n{test_a[:300]}...")
    
    # === TEST B: WITH Graph RAG ===
    print("\n📊 TEST B: Graph RAG (With CFG Paths + 2-Hop Context)")
    print("-" * 40)
    
    # Get graph context
    graph_context_str = ""
    paths = []
    if func_name and graph_loader.graph:
        context_data = graph_loader.get_2hop_context(func_name)
        if context_data:
            paths = context_data.get('paths', [])
            graph_context_str = graph_loader.format_graph_context(context_data)
            print(f"   ✨ Found {len(paths)} execution paths!")
    
    mocks = detect_smart_mocks(func_name, code_snippet)
    
    prompt_b = f"""Generate a comprehensive unit test for:
```python
{code_snippet}
```

{graph_context_str}

## Smart Mocks Detected
{chr(10).join(f'- {m}' for m in mocks) if mocks else 'None required'}

## Requirements
1. Test each execution path listed above
2. Include edge cases (None, empty, boundary)
3. Test error handling paths (RAISES)
"""
    
    inputs = tokenizer(prompt_b, return_tensors="pt", truncation=True, max_length=2000)
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=500,
            temperature=0.7,
            do_sample=True
        )
    
    test_b = tokenizer.decode(outputs[0], skip_special_tokens=True)
    test_b = test_b[len(prompt_b):].strip()
    
    print(f"Generated test (with Graph RAG):\n{test_b[:300]}...")
    
    # === COMPARISON ===
    print("\n" + "=" * 60)
    print("📊 COMPARISON RESULTS")
    print("=" * 60)
    
    # Compare semantically if available
    if SEMANTIC_AVAILABLE:
        relevance_a = calculate_relevance_semantic(test_a, code_snippet)
        relevance_b = calculate_relevance_semantic(test_b, code_snippet)
        print(f"Semantic Relevance (A - Standard): {relevance_a:.2%}")
        print(f"Semantic Relevance (B - Graph RAG): {relevance_b:.2%}")
        print(f"Improvement: {(relevance_b - relevance_a) * 100:.1f}%")
    
    # Check path coverage
    paths_covered_a = sum(1 for p in paths if any(word in test_a.lower() for word in p.lower().split()[:3]))
    paths_covered_b = sum(1 for p in paths if any(word in test_b.lower() for word in p.lower().split()[:3]))
    
    print(f"\nPath Coverage (A): {paths_covered_a}/{len(paths)}")
    print(f"Path Coverage (B): {paths_covered_b}/{len(paths)}")
    
    return {
        'test_a': test_a,
        'test_b': test_b,
        'paths_found': len(paths),
        'paths_covered_a': paths_covered_a,
        'paths_covered_b': paths_covered_b
    }

# ============================================================
# CELL 11: Run Everything
# ============================================================

print("Loading model...")
model, tokenizer = load_model()

print("\nRunning evaluation...")
metrics, tests, total_time = run_evaluation(model, tokenizer, MAX_INSTANCES)

print("\nCalculating summary...")
summary = calculate_summary(metrics, total_time)

print_summary(summary)

print("\nSaving results...")
save_results(metrics, tests, summary)

print("\n🎉 EVALUATION COMPLETE!")

# ============================================================
# CELL 12: Download
# ============================================================
"""
!zip -r /kaggle/working/testgen_research_eval.zip /kaggle/working/testgen_evaluation/
"""
