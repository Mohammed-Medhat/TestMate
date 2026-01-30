# test_graphrag_complete.py - Test Graph-RAG without distillation
"""
Complete testing pipeline to validate Graph-RAG works.

Tests:
1. Graph construction quality
2. Retrieval accuracy (does it find buggy functions?)
3. Baseline comparison (vanilla model vs Graph-RAG)

No distillation needed - just validates your Graph-RAG!
"""

import json
from pathlib import Path
from ast_parser_complete import KGCompassParser
from final_graph_rag import KGCompassGraphRAG
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm import tqdm

class GraphRAGTester:
    """
    Test Graph-RAG without distillation.
    
    Validates:
    - Graph construction works
    - Retrieval finds relevant functions
    - Graph-RAG improves vanilla model
    """
    
    def __init__(self, model_name: str = "Qwen/Qwen2.5-Coder-7B"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
    
    def load_model(self):
        """Load vanilla Qwen 7B for testing"""
        print(f"📥 Loading {self.model_name}...")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        
        print("✅ Model loaded")
    
    # ========================================================================
    # TEST 1: Graph Construction Quality
    # ========================================================================
    
    def test_graph_construction(self, repo_path: str):
        """
        Test if graph construction works correctly.
        
        Checks:
        - All Python files parsed
        - Functions extracted
        - Edges created (CALLS, CONTAIN)
        """
        print("\n" + "="*60)
        print("TEST 1: Graph Construction Quality")
        print("="*60)
        
        parser = KGCompassParser()
        graph = parser.parse_repository(repo_path)
        
        stats = graph.get_stats()
        
        print(f"\n📊 Graph Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # Validation checks
        checks = {
            'Has files': stats.get('file', 0) > 0,
            'Has functions': stats.get('function', 0) > 0,
            'Has edges': graph.graph.number_of_edges() > 0,
            'Functions per file': stats.get('function', 0) / max(stats.get('file', 1), 1)
        }
        
        print(f"\n✅ Validation Checks:")
        for check, passed in checks.items():
            status = "✅" if passed else "❌"
            print(f"  {status} {check}: {passed}")
        
        return graph, all(checks.values())
    
    # ========================================================================
    # TEST 2: Retrieval Accuracy
    # ========================================================================
    
    def test_retrieval_accuracy(self, test_cases: list):
        """
        Test if retrieval finds the correct buggy functions.
        
        Test cases format:
        [
            {
                'repo_path': './project',
                'issue': {'id': 'BUG-1', 'title': '...', 'description': '...'},
                'buggy_function': 'extract_text_from_pdf',
                'buggy_file': 'utils.py'
            },
            ...
        ]
        
        Metric: Top-K recall (is buggy function in top K candidates?)
        """
        print("\n" + "="*60)
        print("TEST 2: Retrieval Accuracy")
        print("="*60)
        
        results = {
            'top_1': 0,
            'top_5': 0,
            'top_10': 0,
            'top_20': 0,
            'total': len(test_cases)
        }
        
        for i, test_case in enumerate(tqdm(test_cases, desc="Testing retrieval")):
            # Build graph
            parser = KGCompassParser()
            graph = parser.parse_repository(test_case['repo_path'])
            
            # Add issue
            parser.add_issue(test_case['issue'])
            
            # Retrieve top 20
            try:
                top_20 = graph.retrieve_top_20_candidates(
                    test_case['issue']['id'],
                    max_candidates=20
                )
            except Exception as e:
                print(f"\n⚠️  Retrieval failed for {test_case['issue']['id']}: {e}")
                continue
            
            # Check if buggy function is in top K
            buggy_func = test_case['buggy_function']
            
            candidate_names = [c['function'] for c in top_20]
            
            # Update results
            if buggy_func in candidate_names[:1]:
                results['top_1'] += 1
            if buggy_func in candidate_names[:5]:
                results['top_5'] += 1
            if buggy_func in candidate_names[:10]:
                results['top_10'] += 1
            if buggy_func in candidate_names[:20]:
                results['top_20'] += 1
        
        # Print results
        print(f"\n📊 Retrieval Accuracy:")
        for k in [1, 5, 10, 20]:
            recall = results[f'top_{k}'] / results['total'] * 100
            print(f"  Top-{k} Recall: {recall:.1f}% ({results[f'top_{k}']}/{results['total']})")
        
        # Target from papers: Top-20 recall should be 70%+
        success = results['top_20'] / results['total'] >= 0.7
        
        if success:
            print(f"\n✅ PASS: Top-20 recall >= 70%")
        else:
            print(f"\n⚠️  BELOW TARGET: Top-20 recall < 70%")
        
        return results, success
    
    # ========================================================================
    # TEST 3: Baseline Comparison (Vanilla vs Graph-RAG)
    # ========================================================================
    
    def test_vanilla_vs_graphrag(self, test_bugs: list):
        """
        Compare vanilla model vs model with Graph-RAG context.
        
        This is your KEY validation!
        
        Test bugs format:
        [
            {
                'repo_path': './project',
                'issue': {...},
                'buggy_code': '...',
                'fixed_code': '...',
                'buggy_function': '...'
            },
            ...
        ]
        """
        print("\n" + "="*60)
        print("TEST 3: Vanilla vs Graph-RAG Comparison")
        print("="*60)
        
        if self.model is None:
            self.load_model()
        
        results = {
            'vanilla': {'correct': 0, 'total': 0},
            'graphrag': {'correct': 0, 'total': 0}
        }
        
        for test_bug in tqdm(test_bugs, desc="Testing"):
            
            # ================================================================
            # Experiment A: Vanilla (no Graph-RAG)
            # ================================================================
            
            vanilla_prompt = f"""Fix this bug:

Bug Description:
{test_bug['issue']['description']}

Buggy Code:
```python
{test_bug['buggy_code']}
```

Generate ONLY the fixed code:
"""
            
            vanilla_fix = self._generate_fix(vanilla_prompt)
            vanilla_correct = self._is_fix_correct(vanilla_fix, test_bug['fixed_code'])
            
            results['vanilla']['total'] += 1
            if vanilla_correct:
                results['vanilla']['correct'] += 1
            
            # ================================================================
            # Experiment B: With Graph-RAG
            # ================================================================
            
            # Build graph
            parser = KGCompassParser()
            graph = parser.parse_repository(test_bug['repo_path'])
            parser.add_issue(test_bug['issue'])
            
            # Retrieve context
            try:
                top_20 = graph.retrieve_top_20_candidates(
                    test_bug['issue']['id'],
                    max_candidates=20
                )
                context = graph.format_kgcompass_prompt(test_bug['issue']['id'], top_20)
            except:
                context = f"Issue: {test_bug['issue']['description']}"
            
            graphrag_prompt = f"""{context}

Buggy Code:
```python
{test_bug['buggy_code']}
```

Generate ONLY the fixed code:
"""
            
            graphrag_fix = self._generate_fix(graphrag_prompt)
            graphrag_correct = self._is_fix_correct(graphrag_fix, test_bug['fixed_code'])
            
            results['graphrag']['total'] += 1
            if graphrag_correct:
                results['graphrag']['correct'] += 1
        
        # Print comparison
        print(f"\n📊 Results:")
        
        vanilla_acc = results['vanilla']['correct'] / results['vanilla']['total'] * 100
        graphrag_acc = results['graphrag']['correct'] / results['graphrag']['total'] * 100
        improvement = graphrag_acc - vanilla_acc
        
        print(f"  Vanilla (no Graph-RAG): {vanilla_acc:.1f}% ({results['vanilla']['correct']}/{results['vanilla']['total']})")
        print(f"  With Graph-RAG: {graphrag_acc:.1f}% ({results['graphrag']['correct']}/{results['graphrag']['total']})")
        print(f"  Improvement: {improvement:+.1f}%")
        
        # Success if Graph-RAG improves by at least 3%
        success = improvement >= 3.0
        
        if success:
            print(f"\n✅ PASS: Graph-RAG improves accuracy by {improvement:.1f}%")
        else:
            print(f"\n⚠️  BELOW TARGET: Graph-RAG improvement < 3%")
        
        return results, success
    
    def _generate_fix(self, prompt: str) -> str:
        """Generate fix using model"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=300,
                temperature=0.3,
                do_sample=True
            )
        
        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract code
        if "```python" in generated:
            code = generated.split("```python")[1].split("```")[0]
        elif "```" in generated:
            code = generated.split("```")[1].split("```")[0]
        else:
            code = generated
        
        return code.strip()
    
    def _is_fix_correct(self, generated_fix: str, expected_fix: str) -> bool:
        """
        Simple correctness check.
        
        In production, you'd run actual tests.
        For now, check if key patterns match.
        """
        # Normalize
        gen_normalized = generated_fix.lower().replace(" ", "")
        exp_normalized = expected_fix.lower().replace(" ", "")
        
        # Check similarity (simple heuristic)
        from difflib import SequenceMatcher
        similarity = SequenceMatcher(None, gen_normalized, exp_normalized).ratio()
        
        # Consider correct if >70% similar
        return similarity > 0.7

# ============================================================================
# QUICK TEST SCRIPT
# ============================================================================

def create_test_cases():
    """
    Create simple test cases for validation.
    
    In production, use real SWE-bench examples.
    """
    # You would load these from your actual data
    return [
        {
            'repo_path': '.',  # Your current project
            'issue': {
                'id': 'TEST-001',
                'title': 'Function fails on null input',
                'description': 'extract_text_from_pdf crashes when file_or_path is None',
                'referenced_files': ['utils.py']
            },
            'buggy_function': 'extract_text_from_pdf',
            'buggy_file': 'utils.py',
            'buggy_code': 'pdf = fitz.open(file_or_path)\nreturn text',
            'fixed_code': 'if file_or_path is None:\n    return ""\npdf = fitz.open(file_or_path)\nreturn text'
        }
    ]

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("="*60)
    print("Graph-RAG Validation Suite (No Distillation)")
    print("="*60)
    
    tester = GraphRAGTester()
    
    # Test 1: Graph construction
    print("\n🔧 Running Test 1: Graph Construction...")
    graph, test1_pass = tester.test_graph_construction(".")
    
    # Test 2: Retrieval accuracy
    print("\n🔍 Running Test 2: Retrieval Accuracy...")
    test_cases = create_test_cases()
    retrieval_results, test2_pass = tester.test_retrieval_accuracy(test_cases)
    
    # Test 3: Baseline comparison (MOST IMPORTANT!)
    print("\n⚖️  Running Test 3: Vanilla vs Graph-RAG...")
    comparison_results, test3_pass = tester.test_vanilla_vs_graphrag(test_cases)
    
    # Final summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    
    all_pass = test1_pass and test2_pass and test3_pass
    
    print(f"\nTest 1 (Graph Construction): {'✅ PASS' if test1_pass else '❌ FAIL'}")
    print(f"Test 2 (Retrieval Accuracy): {'✅ PASS' if test2_pass else '❌ FAIL'}")
    print(f"Test 3 (Graph-RAG Improvement): {'✅ PASS' if test3_pass else '✅ FAIL'}")
    
    if all_pass:
        print(f"\n🎉 ALL TESTS PASSED! Graph-RAG is working!")
        print(f"\n✅ Ready to proceed to distillation")
    else:
        print(f"\n⚠️  Some tests failed. Debug before distillation.")
    
    print("\n" + "="*60)