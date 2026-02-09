# ============================================================
# FixMate: Integrated Test Generator with Graph RAG
# ============================================================
# Combines:
#   - Layer 1: Documentation Retriever (testing patterns)
#   - Layer 2: Code Navigator (code structure, similar tests)
#   - LLM: Test generation with RAG context
# ============================================================

import ast
import re
import sys
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from layer1_docs.docs_retriever import DocsRetriever
from layer2_code.code_navigator import CodeNavigator

# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class TestGenerationRequest:
    """Input for test generation."""
    repo_path: str
    target_file: Optional[str] = None  # If None, test entire repo
    target_function: Optional[str] = None  # If None, test all functions in file
    context: str = ""  # Additional context from user
    
@dataclass 
class GeneratedTest:
    """Output from test generation."""
    function_name: str
    file_path: str
    test_code: str
    generation_time_ms: float
    syntax_valid: bool
    retry_count: int = 0
    rag_context_used: Dict = field(default_factory=dict)

@dataclass
class CFGPath:
    """A control flow path through a function."""
    path_id: int
    conditions: List[str]  # List of branch conditions
    description: str

# ============================================================
# CFG PATH ANALYZER
# ============================================================

class CFGPathAnalyzer:
    """Analyze Control Flow Graph paths in Python code."""
    
    def __init__(self):
        self.current_path = []
    
    def extract_paths(self, code: str) -> List[CFGPath]:
        """Extract all execution paths from a function."""
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return []
        
        paths = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_paths = self._analyze_function(node)
                paths.extend(func_paths)
        
        return paths
    
    def _analyze_function(self, func: ast.FunctionDef) -> List[CFGPath]:
        """Analyze a single function for paths."""
        paths = []
        conditions = self._collect_conditions(func.body)
        
        # Generate paths from conditions
        if not conditions:
            paths.append(CFGPath(
                path_id=0,
                conditions=[],
                description="Single path (no branches)"
            ))
        else:
            # Each condition creates 2 paths (true/false)
            path_id = 0
            for i, cond in enumerate(conditions):
                paths.append(CFGPath(
                    path_id=path_id,
                    conditions=[f"{cond} = True"],
                    description=f"Path when {cond}"
                ))
                path_id += 1
                paths.append(CFGPath(
                    path_id=path_id,
                    conditions=[f"{cond} = False"],
                    description=f"Path when NOT {cond}"
                ))
                path_id += 1
        
        return paths
    
    def _collect_conditions(self, body: List) -> List[str]:
        """Collect all branch conditions in a code block."""
        conditions = []
        for node in body:
            if isinstance(node, ast.If):
                cond = ast.unparse(node.test) if hasattr(ast, 'unparse') else "condition"
                conditions.append(cond)
                # Recurse into branches
                conditions.extend(self._collect_conditions(node.body))
                conditions.extend(self._collect_conditions(node.orelse))
            elif isinstance(node, ast.For):
                conditions.append(f"loop over {ast.unparse(node.iter) if hasattr(ast, 'unparse') else 'iterable'}")
            elif isinstance(node, ast.While):
                cond = ast.unparse(node.test) if hasattr(ast, 'unparse') else "condition"
                conditions.append(f"while {cond}")
        return conditions

# ============================================================
# MOCK GENERATOR
# ============================================================

class MockGenerator:
    """Generate mocks for external dependencies."""
    
    EXTERNAL_MODULES = {
        'requests': '@patch("requests.get")',
        'urllib': '@patch("urllib.request.urlopen")',
        'sqlite3': '@patch("sqlite3.connect")',
        'pymongo': '@patch("pymongo.MongoClient")',
        'redis': '@patch("redis.Redis")',
        'boto3': '@patch("boto3.client")',
        'smtplib': '@patch("smtplib.SMTP")',
    }
    
    def detect_external_deps(self, code: str) -> List[str]:
        """Detect external dependencies that need mocking."""
        deps = []
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name in self.EXTERNAL_MODULES:
                            deps.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module and node.module.split('.')[0] in self.EXTERNAL_MODULES:
                        deps.append(node.module.split('.')[0])
        except SyntaxError:
            pass
        return list(set(deps))
    
    def generate_mock_decorators(self, deps: List[str]) -> str:
        """Generate mock decorators for dependencies."""
        decorators = []
        for dep in deps:
            if dep in self.EXTERNAL_MODULES:
                decorators.append(self.EXTERNAL_MODULES[dep])
        return '\n'.join(decorators)

# ============================================================
# INTEGRATED TEST GENERATOR
# ============================================================

class IntegratedTestGenerator:
    """
    Test generator with Graph RAG integration.
    
    Uses:
    - Layer 1: Documentation for testing patterns
    - Layer 2: Code structure for context
    - CFG: Path analysis for coverage
    - Self-correction: Retry on syntax errors
    """
    
    def __init__(self, model=None, tokenizer=None):
        self.model = model
        self.tokenizer = tokenizer
        self.docs_retriever = None
        self.code_navigator = None
        self.cfg_analyzer = CFGPathAnalyzer()
        self.mock_generator = MockGenerator()
        self.max_retries = 3
        
    def init_layers(self, docs_index_path: str = None, repo_path: str = None):
        """Initialize RAG layers."""
        # Layer 1: Documentation
        self.docs_retriever = DocsRetriever(docs_index_path or "docs_index")
        try:
            self.docs_retriever.load()
            print("✅ Layer 1: Documentation loaded")
        except:
            print("⚠️ Layer 1: No index found, creating empty")
        
        # Layer 2: Code Navigator
        if repo_path:
            self.code_navigator = CodeNavigator(repo_path)
            print(f"✅ Layer 2: Code indexed ({self.code_navigator.get_stats()['total_files']} files)")
    
    def generate_tests_for_repo(self, request: TestGenerationRequest) -> List[GeneratedTest]:
        """Generate tests for an entire repository."""
        
        # Initialize Layer 2 with repo
        if request.repo_path:
            self.code_navigator = CodeNavigator(request.repo_path)
        
        results = []
        
        # Get all Python files
        files = self.code_navigator.search_file("*.py")
        
        for file_result in files:
            file_path = file_result.location.file_path
            
            # Skip test files
            if 'test' in file_path.lower():
                continue
            
            # Generate tests for this file
            file_tests = self.generate_tests_for_file(file_path)
            results.extend(file_tests)
        
        return results
    
    def generate_tests_for_file(self, file_path: str) -> List[GeneratedTest]:
        """Generate tests for all functions in a file."""
        
        content = self.code_navigator.get_file_content(file_path)
        if not content:
            return []
        
        # Extract function definitions
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return []
        
        results = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Skip private/magic methods
                if node.name.startswith('_'):
                    continue
                
                # Get function source
                func_source = ast.get_source_segment(content, node) or ""
                
                # Generate test for this function
                test = self.generate_test_with_rag(
                    function_name=node.name,
                    function_code=func_source,
                    file_path=file_path
                )
                results.append(test)
        
        return results
    
    def generate_test_with_rag(self, function_name: str, function_code: str, 
                                file_path: str) -> GeneratedTest:
        """
        Generate test with full RAG context and self-correction.
        """
        start_time = time.time()
        
        # ===== GATHER RAG CONTEXT =====
        rag_context = {}
        
        # Layer 1: Documentation (testing patterns)
        if self.docs_retriever:
            doc_results = self.docs_retriever.retrieve(
                f"python unit test {function_name}",
                top_k=3
            )
            rag_context['docs'] = [r.chunk.content[:300] for r in doc_results]
        
        # Layer 2: Similar code/existing tests
        if self.code_navigator:
            similar = self.code_navigator.search_bm25(function_name, top_k=3)
            rag_context['similar_code'] = [r.location.content[:300] for r in similar]
        
        # CFG: Path analysis
        paths = self.cfg_analyzer.extract_paths(function_code)
        rag_context['paths'] = [p.description for p in paths]
        
        # Mocks: Detect external deps
        external_deps = self.mock_generator.detect_external_deps(function_code)
        mock_decorators = self.mock_generator.generate_mock_decorators(external_deps)
        rag_context['mocks'] = mock_decorators
        
        # ===== SELF-CORRECTION LOOP =====
        test_code = ""
        syntax_valid = False
        retry_count = 0
        
        for attempt in range(self.max_retries):
            # Build prompt with RAG context
            prompt = self._build_rag_prompt(
                function_name=function_name,
                function_code=function_code,
                rag_context=rag_context,
                previous_error=None if attempt == 0 else "Syntax error - please fix"
            )
            
            # Generate
            test_code = self._generate(prompt)
            
            # Check syntax
            syntax_valid, error = self._check_syntax(test_code)
            
            if syntax_valid:
                break
            
            retry_count += 1
        
        gen_time = (time.time() - start_time) * 1000
        
        return GeneratedTest(
            function_name=function_name,
            file_path=file_path,
            test_code=test_code,
            generation_time_ms=gen_time,
            syntax_valid=syntax_valid,
            retry_count=retry_count,
            rag_context_used=rag_context
        )
    
    def _build_rag_prompt(self, function_name: str, function_code: str,
                          rag_context: Dict, previous_error: str = None) -> str:
        """Build prompt with RAG context."""
        
        prompt = f"""=== COMPREHENSIVE TEST GENERATION ===

## Function to Test
```python
{function_code}
```

"""
        
        # Add documentation context
        if rag_context.get('docs'):
            prompt += "## Testing Patterns (from documentation)\n"
            for doc in rag_context['docs'][:2]:
                prompt += f"- {doc[:200]}...\n"
            prompt += "\n"
        
        # Add similar code context
        if rag_context.get('similar_code'):
            prompt += "## Similar Code Examples\n"
            for code in rag_context['similar_code'][:2]:
                prompt += f"```python\n{code[:200]}\n```\n"
            prompt += "\n"
        
        # Add path coverage requirements
        if rag_context.get('paths'):
            prompt += "## Paths to Cover\n"
            for path in rag_context['paths']:
                prompt += f"- {path}\n"
            prompt += "\n"
        
        # Add mock decorators
        if rag_context.get('mocks'):
            prompt += f"## Required Mocks\n{rag_context['mocks']}\n\n"
        
        # Add error feedback for self-correction
        if previous_error:
            prompt += f"## ⚠️ Previous Attempt Failed\n{previous_error}\nPlease fix the syntax.\n\n"
        
        prompt += f"""## Generate Tests
Generate comprehensive unit tests for `{function_name}`:
1. Basic functionality test
2. Edge cases (empty, None, boundaries)
3. Error handling (exceptions)
4. Path coverage (test each branch)

```python
import pytest
from unittest.mock import patch, MagicMock

"""
        
        return prompt
    
    def _generate(self, prompt: str) -> str:
        """Generate test code using the model."""
        if self.model is None or self.tokenizer is None:
            # Return template if no model
            return self._generate_template()
        
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=500,
                temperature=0.7,
                do_sample=True,
                top_p=0.95,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        test_code = response[len(prompt):]
        
        if "```" in test_code:
            test_code = test_code[:test_code.find("```")]
        
        return test_code.strip()
    
    def _generate_template(self) -> str:
        """Generate a template test (when no model available)."""
        return '''def test_basic():
    """Test basic functionality."""
    result = function_under_test()
    assert result is not None

def test_edge_cases():
    """Test edge cases."""
    assert function_under_test(None) is None
    assert function_under_test([]) == []

def test_error_handling():
    """Test error handling."""
    with pytest.raises(ValueError):
        function_under_test(invalid_input)
'''
    
    def _check_syntax(self, code: str) -> Tuple[bool, str]:
        """Check if generated code has valid syntax."""
        try:
            ast.parse(code)
            return True, ""
        except SyntaxError as e:
            return False, f"Line {e.lineno}: {e.msg}"

# ============================================================
# PRIORITY SCORER
# ============================================================

class TestPriorityScorer:
    """Score and prioritize functions for testing."""
    
    def score_function(self, func_code: str, func_name: str) -> float:
        """Calculate priority score (0-1, higher = more important to test)."""
        score = 0.0
        
        # Complexity score
        try:
            tree = ast.parse(func_code)
            for node in ast.walk(tree):
                if isinstance(node, (ast.If, ast.While, ast.For)):
                    score += 0.1  # More branches = higher priority
                if isinstance(node, ast.ExceptHandler):
                    score += 0.15  # Error handling = important
                if isinstance(node, ast.Return):
                    score += 0.05
        except SyntaxError:
            pass
        
        # Name-based priority
        if func_name.startswith('get_') or func_name.startswith('set_'):
            score += 0.1  # Accessors
        if 'validate' in func_name or 'check' in func_name:
            score += 0.2  # Validation functions
        if 'process' in func_name or 'handle' in func_name:
            score += 0.15  # Processing functions
        
        # Length score (longer = more complex)
        lines = func_code.count('\n')
        score += min(lines * 0.01, 0.2)
        
        return min(score, 1.0)  # Cap at 1.0

# ============================================================
# CLI / DEMO
# ============================================================

def demo():
    """Demo the integrated test generator."""
    print("="*60)
    print("FixMate: Integrated Test Generator with Graph RAG")
    print("="*60)
    
    # Sample code to test
    sample_code = '''
def calculate_discount(price, customer_type, quantity):
    """Calculate discount based on customer type and quantity."""
    if price <= 0:
        raise ValueError("Price must be positive")
    
    if customer_type == "premium":
        base_discount = 0.2
    elif customer_type == "regular":
        base_discount = 0.1
    else:
        base_discount = 0.0
    
    if quantity >= 10:
        volume_discount = 0.05
    else:
        volume_discount = 0.0
    
    total_discount = base_discount + volume_discount
    return price * (1 - total_discount)
'''
    
    print("\n📝 Input Code:")
    print(sample_code)
    
    # Initialize generator (without model for demo)
    generator = IntegratedTestGenerator()
    
    # CFG Path Analysis
    print("\n🔀 CFG Path Analysis:")
    paths = generator.cfg_analyzer.extract_paths(sample_code)
    for path in paths:
        print(f"  Path {path.path_id}: {path.description}")
    
    # Mock Detection
    print("\n🔧 Mock Detection:")
    deps = generator.mock_generator.detect_external_deps(sample_code)
    print(f"  External deps: {deps if deps else 'None'}")
    
    # Priority Scoring
    print("\n📊 Priority Score:")
    scorer = TestPriorityScorer()
    score = scorer.score_function(sample_code, "calculate_discount")
    print(f"  calculate_discount: {score:.2f}")
    
    # Generate test (template mode)
    print("\n🧪 Generated Test (template mode):")
    test = generator.generate_test_with_rag(
        function_name="calculate_discount",
        function_code=sample_code,
        file_path="pricing.py"
    )
    print(f"  Syntax valid: {test.syntax_valid}")
    print(f"  Retry count: {test.retry_count}")
    print(f"  RAG context used: {list(test.rag_context_used.keys())}")
    print(f"\n{test.test_code}")

if __name__ == "__main__":
    demo()
