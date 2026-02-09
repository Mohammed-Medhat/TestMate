# agent.py - Main SWE-bench Agent (3-Layer RAG)
"""
SWE-bench Lite Agent combining:
- Layer 1: Documentation Retriever (RAGFix)
- Layer 2: Code Navigator (KGCompass)
- Layer 3: LLM Reasoning Engine (Knowledge Distillation)

Based on papers:
- KGCompass (arXiv 2025): Multi-hop graph traversal
- RAGFix (NSF/IEEE 2025): External knowledge retrieval
- RAG Traceability (MCSE 2025): Requirement-code linking
- Knowledge Distillation (arXiv 2025): Efficient model training
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass

# Add parent paths for imports
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

# ============================================================================
# ISSUE DATA STRUCTURE
# ============================================================================

@dataclass
class SWEBenchIssue:
    """A SWE-bench issue instance."""
    instance_id: str
    repo: str
    problem_statement: str
    hints_text: str = ""
    base_commit: str = ""
    patch: str = ""  # Ground truth (not available at inference)

# ============================================================================
# AGENT
# ============================================================================

class SWEBenchAgent:
    """
    3-Layer RAG Agent for SWE-bench Lite.
    
    Execution Flow:
    1. Receive issue
    2. Layer 2: Navigate codebase to find relevant files
    3. Layer 1: Retrieve documentation for APIs/errors mentioned
    4. Layer 3: Generate patch using LLM with combined context
    """
    
    def __init__(self, model_path: str = None, docs_index_path: str = None):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        
        # Initialize layers
        self.docs_retriever = None
        self.code_navigator = None
        self.graph_rag = None
        
        # Lazy load components
        self._init_layer1(docs_index_path)
    
    def _init_layer1(self, docs_index_path: str = None):
        """Initialize Layer 1: Documentation Retriever."""
        try:
            from layer1_docs.docs_retriever import DocsRetriever
            self.docs_retriever = DocsRetriever()
            if docs_index_path:
                self.docs_retriever.load(docs_index_path)
            print("Layer 1 (Docs Retriever) initialized")
        except ImportError as e:
            print(f"Warning: Could not initialize Layer 1: {e}")
    
    def _init_layer2(self, repo_path: str):
        """Initialize Layer 2: Code Navigator for a specific repo."""
        try:
            from layer2_code.code_navigator import CodeNavigator
            self.code_navigator = CodeNavigator()
            self.code_navigator.index_repository(repo_path)
            print(f"Layer 2 (Code Navigator) indexed: {self.code_navigator.get_stats()}")
        except ImportError as e:
            print(f"Warning: Could not initialize Layer 2: {e}")
    
    def _init_layer3(self):
        """Initialize Layer 3: LLM with LoRA weights."""
        if self.model is not None:
            return
        
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
            from peft import PeftModel
            
            base_model = "Qwen/Qwen2.5-Coder-7B"
            
            print("Loading LLM...")
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4"
            )
            
            self.tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            model = AutoModelForCausalLM.from_pretrained(
                base_model,
                quantization_config=quant_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16
            )
            
            # Load LoRA weights if available
            if self.model_path and Path(self.model_path).exists():
                print(f"Loading LoRA weights from {self.model_path}")
                self.model = PeftModel.from_pretrained(model, self.model_path)
            else:
                self.model = model
            
            print("Layer 3 (LLM) initialized")
            
        except Exception as e:
            print(f"Warning: Could not initialize Layer 3: {e}")
    
    # ========================================================================
    # MAIN SOLVE METHOD
    # ========================================================================
    
    def solve(self, issue: SWEBenchIssue, repo_path: str = None) -> str:
        """
        Solve a SWE-bench issue.
        
        Args:
            issue: The issue to solve
            repo_path: Path to the cloned repository
            
        Returns:
            Generated patch as a string
        """
        print(f"\n{'='*60}")
        print(f"Solving: {issue.instance_id}")
        print(f"{'='*60}")
        
        # Step 1: Initialize Layer 2 for this repo
        if repo_path:
            self._init_layer2(repo_path)
        
        # Step 2: Find relevant code (Layer 2)
        code_context = self._retrieve_code_context(issue)
        
        # Step 3: Find relevant documentation (Layer 1)
        docs_context = self._retrieve_docs_context(issue)
        
        # Step 4: Build prompt
        prompt = self._build_prompt(issue, code_context, docs_context)
        
        # Step 5: Generate patch (Layer 3)
        patch = self._generate_patch(prompt)
        
        return patch
    
    def _retrieve_code_context(self, issue: SWEBenchIssue) -> Dict:
        """Layer 2: Retrieve relevant code from repository."""
        context = {
            'files': [],
            'functions': [],
            'repo_map': ""
        }
        
        if not self.code_navigator:
            return context
        
        # Search for relevant code
        query = f"{issue.problem_statement[:500]}"
        results = self.code_navigator.search_bm25(query, top_k=10)
        
        for r in results[:5]:
            context['functions'].append({
                'name': r.location.symbol_name,
                'file': r.location.file_path,
                'content': r.location.content,
                'score': r.score
            })
        
        # Get repo map
        context['repo_map'] = self.code_navigator.get_repo_map(max_files=30)
        
        # Search for specific files mentioned
        for word in issue.problem_statement.split():
            if '.py' in word:
                matches = self.code_navigator.search_file(word)
                context['files'].extend(matches[:3])
        
        context['files'] = list(set(context['files']))[:5]
        
        return context
    
    def _retrieve_docs_context(self, issue: SWEBenchIssue) -> Dict:
        """Layer 1: Retrieve relevant documentation."""
        context = {
            'api_docs': [],
            'stackoverflow': []
        }
        
        if not self.docs_retriever:
            return context
        
        # Search for API docs
        query = f"{issue.repo} {issue.problem_statement[:300]}"
        results = self.docs_retriever.retrieve(query, top_k=3, repo_filter=issue.repo)
        
        for r in results:
            if r.chunk.source == "stackoverflow":
                context['stackoverflow'].append({
                    'title': r.chunk.title,
                    'content': r.chunk.content[:500],
                    'relevance': r.relevance
                })
            else:
                context['api_docs'].append({
                    'title': r.chunk.title,
                    'content': r.chunk.content[:500],
                    'relevance': r.relevance
                })
        
        return context
    
    def _build_prompt(self, issue: SWEBenchIssue, code_context: Dict, docs_context: Dict) -> str:
        """Build the prompt for the LLM."""
        
        prompt = f"""=== BUG REPAIR TASK ===

## Issue
Repository: {issue.repo}
Instance: {issue.instance_id}

{issue.problem_statement[:1500]}

"""
        
        # Add relevant code (Layer 2)
        if code_context['functions']:
            prompt += "## Relevant Code (from codebase search)\n"
            for i, func in enumerate(code_context['functions'][:3], 1):
                prompt += f"""
### {i}. {func['name']}
File: {func['file']}
```python
{func['content'][:400]}
```
"""
        
        # Add documentation (Layer 1)
        if docs_context['api_docs']:
            prompt += "\n## API Documentation\n"
            for doc in docs_context['api_docs'][:2]:
                prompt += f"**{doc['title']}**: {doc['content'][:300]}...\n\n"
        
        if docs_context['stackoverflow']:
            prompt += "\n## Similar Issues (Stack Overflow)\n"
            for so in docs_context['stackoverflow'][:1]:
                prompt += f"{so['content'][:300]}...\n\n"
        
        # Generation instruction
        prompt += """
## Task
Generate a patch to fix this issue. Output a valid unified diff format:

```diff
"""
        
        return prompt
    
    def _generate_patch(self, prompt: str) -> str:
        """Layer 3: Generate patch using LLM."""
        
        # Initialize LLM if not loaded
        self._init_layer3()
        
        if not self.model or not self.tokenizer:
            return "# Error: LLM not available"
        
        import torch
        
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract the patch from response
        patch = response[len(prompt):]
        
        # Try to extract just the diff
        if "```diff" in patch:
            start = patch.find("```diff") + 7
            end = patch.find("```", start)
            if end > start:
                patch = patch[start:end].strip()
        
        return patch
    
    # ========================================================================
    # BATCH PROCESSING
    # ========================================================================
    
    def solve_batch(self, issues: List[SWEBenchIssue], repos_dir: str) -> List[Dict]:
        """Solve multiple issues."""
        results = []
        
        for issue in issues:
            repo_path = Path(repos_dir) / issue.repo.replace('/', '_')
            
            try:
                patch = self.solve(issue, str(repo_path) if repo_path.exists() else None)
                results.append({
                    'instance_id': issue.instance_id,
                    'model_patch': patch,
                    'error': None
                })
            except Exception as e:
                results.append({
                    'instance_id': issue.instance_id,
                    'model_patch': '',
                    'error': str(e)
                })
        
        return results

# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    print("SWE-bench Agent")
    print("="*60)
    
    # Demo usage
    agent = SWEBenchAgent()
    
    # Create a test issue
    test_issue = SWEBenchIssue(
        instance_id="django__django-11001",
        repo="django",
        problem_statement="""
        The @ character in username validation is incorrectly escaped.
        The regex pattern ^[\\w.@+-]+$ should use \\A and \\Z anchors instead.
        """
    )
    
    print(f"Test issue: {test_issue.instance_id}")
    print("(Run with repo path to fully test)")
