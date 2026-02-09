# ============================================================
# KAGGLE NOTEBOOK: Train GraphRAG for Comprehensive Test Generation
# ============================================================
# Trains the model WITH KNOWLEDGE GRAPH CONTEXT
# Uses 2-hop neighbors, callers/callees from the graph!
# ============================================================

# CELL 1: Install dependencies
"""
!pip install -q transformers datasets peft bitsandbytes accelerate trl sentence-transformers networkx
"""

# CELL 2: Imports
import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset, Dataset
from datetime import datetime
import json
import os
import re
import pickle
import ast

# Config
MODEL_NAME = "Qwen/Qwen2.5-Coder-7B"
MAX_LENGTH = 2048
OUTPUT_DIR = "/kaggle/working/testgen_model"
GRAPH_PATH = "/kaggle/input/testmate-graph/knowledge_graph.pkl"  # Upload your graph!

LORA_CONFIG = {
    "r": 16,
    "lora_alpha": 32,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "lora_dropout": 0.05,
    "bias": "none",
    "task_type": "CAUSAL_LM"
}

print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ============================================================
# CELL 2.5: Knowledge Graph Loader (THE KEY FIX!)
# ============================================================

class GraphContextExtractor:
    """Extract 2-hop context from Knowledge Graph for training prompts."""
    
    def __init__(self, graph_path: str = None):
        self.graph = None
        if graph_path and os.path.exists(graph_path):
            self.load_graph(graph_path)
    
    def load_graph(self, path: str):
        """Load the knowledge graph from pickle file."""
        try:
            with open(path, 'rb') as f:
                self.graph = pickle.load(f)
            print(f"✅ Loaded Knowledge Graph: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
        except Exception as e:
            print(f"⚠️ Could not load graph: {e}")
            self.graph = None
    
    def get_2hop_context(self, function_name: str) -> dict:
        """
        Get 2-hop neighborhood for a function.
        
        Returns callers, callees, parent class, and requirements.
        """
        if not self.graph or function_name not in self.graph.nodes():
            return None
        
        # Get the function's own data (including CFG paths!)
        func_node = self.graph.nodes[function_name]
        func_data = func_node.get('data', {})
        
        context = {
            'callers': [],      # Who calls this function
            'callees': [],      # Who this function calls
            'parent': None,     # Class/module containing this
            'requirements': [], # Team A requirements
            'siblings': [],     # Other functions in same class
            'paths': func_data.get('paths', [])  # CFG execution paths!
        }
        
        # Get callers (incoming CALLS edges)
        for pred in self.graph.predecessors(function_name):
            edge_data = self.graph[pred][function_name]
            for key, data in edge_data.items():
                if data.get('type') == 'CALLS':
                    node_data = self.graph.nodes[pred].get('data', {})
                    context['callers'].append({
                        'name': pred,
                        'signature': node_data.get('signature', ''),
                        'body': node_data.get('body', '')[:200]
                    })
        
        # Get callees (outgoing CALLS edges)
        for succ in self.graph.successors(function_name):
            edge_data = self.graph[function_name][succ]
            for key, data in edge_data.items():
                if data.get('type') == 'CALLS':
                    node_data = self.graph.nodes[succ].get('data', {})
                    context['callees'].append({
                        'name': succ,
                        'signature': node_data.get('signature', ''),
                        'body': node_data.get('body', '')[:200]
                    })
        
        # Get parent (CONTAIN edge)
        for pred in self.graph.predecessors(function_name):
            edge_data = self.graph[pred][function_name]
            for key, data in edge_data.items():
                if data.get('type') == 'CONTAIN':
                    parent_node = self.graph.nodes[pred]
                    context['parent'] = {
                        'name': pred,
                        'type': parent_node.get('type', 'unknown')
                    }
                    break
        
        # Get requirements (TRACES_TO edge)
        for succ in self.graph.successors(function_name):
            edge_data = self.graph[function_name][succ]
            for key, data in edge_data.items():
                if data.get('type') == 'TRACES_TO':
                    req_node = self.graph.nodes[succ]
                    req_data = req_node.get('data', {})
                    context['requirements'].append({
                        'id': succ,
                        'description': req_data.get('description', '')
                    })
        
        return context
    
    def format_graph_context(self, context: dict) -> str:
        """Format 2-hop context into prompt-friendly string."""
        if not context:
            return ""
        
        parts = []
        
        # CFG PATHS FIRST - This is the "Depth" upgrade!
        if context.get('paths'):
            parts.append("## Execution Paths (MUST COVER IN TESTS):")
            for i, path in enumerate(context['paths'], 1):
                parts.append(f"{i}. {path}")
        
        if context['callers']:
            parts.append("\n## Callers (who calls this function):")
            for c in context['callers'][:3]:
                parts.append(f"- `{c['name']}`: {c['signature']}")
        
        if context['callees']:
            parts.append("\n## Callees (dependencies this function uses):")
            for c in context['callees'][:3]:
                parts.append(f"- `{c['name']}`: {c['signature']}")
        
        if context['parent']:
            parts.append(f"\n## Parent: {context['parent']['type']} `{context['parent']['name']}`")
        
        if context['requirements']:
            parts.append("\n## Requirements (from Team A):")
            for r in context['requirements'][:2]:
                parts.append(f"- {r['id']}: {r['description'][:100]}")
        
        return "\n".join(parts)

# Initialize graph extractor
graph_extractor = GraphContextExtractor(GRAPH_PATH)

def extract_function_name(code: str) -> str:
    """Extract function name from code."""
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                return node.name
    except:
        pass
    # Fallback: regex
    match = re.search(r'def\s+(\w+)\s*\(', code)
    return match.group(1) if match else None

# ============================================================
# CELL 3: Training Data Format - WITH GRAPH CONTEXT
# ============================================================

def format_code_to_test(code: str, test: str, context: str = "", 
                        paths: list = None, mocks: list = None,
                        graph_context: str = "") -> str:
    """Format code → test generation sample with GRAPH RAG context."""
    
    # Build RAG context section
    rag_section = ""
    
    # Add GRAPH context (2-hop neighbors!)
    if graph_context:
        rag_section += f"\n## Graph Context (2-Hop Neighborhood)\n{graph_context}\n"
    
    if paths:
        rag_section += "\n## Paths to Cover\n"
        for path in paths[:5]:
            rag_section += f"- {path}\n"
    
    if mocks:
        rag_section += "\n## Required Mocks\n"
        for mock in mocks[:3]:
            rag_section += f"- @patch(\"{mock}\")\n"
    
    prompt = f"""=== GRAPH-RAG TEST GENERATION ===

## Source Code
```python
{code[:2000]}
```

## Context
{context if context else "Generate comprehensive unit tests for the above code."}
{rag_section}
## Testing Requirements
1. Basic functionality tests
2. Edge cases (None, empty, boundary values)
3. Error handling (exceptions)
4. Path coverage (test each branch)
5. Integration tests for callers/callees

## Generated Tests
```python
{test}
```"""
    
    return prompt

def format_function_test(func_code: str, func_name: str, docstring: str = "") -> str:
    """Format a function for test generation."""
    
    # Create expected test structure
    test_template = f'''import pytest

class Test{func_name.title().replace("_", "")}:
    """Tests for {func_name} function."""
    
    def test_{func_name}_basic(self):
        """Test basic functionality."""
        # TODO: Add basic test
        result = {func_name}()
        assert result is not None
    
    def test_{func_name}_edge_cases(self):
        """Test edge cases."""
        # TODO: Test with None, empty, boundary values
        pass
    
    def test_{func_name}_error_handling(self):
        """Test error conditions."""
        # TODO: Test exceptions
        pass
'''
    
    prompt = f"""=== COMPREHENSIVE TEST GENERATION ===

## Source Code
```python
{func_code}
```

## Context
Generate comprehensive unit tests for the function `{func_name}`.
{f"Docstring: {docstring}" if docstring else ""}
Include: basic tests, edge cases, error handling.

## Generated Tests
```python
{test_template}
```"""
    
    return prompt

# ============================================================
# CELL 4: Load Training Data - Multiple Sources
# ============================================================

def load_comprehensive_training_data(swe_limit=400, magic_limit=400, mbpp_limit=200):
    """
    Load data for GRAPH-RAG test generation:
    - SWE-bench: Real bug tests WITH GRAPH CONTEXT
    - Magicoder: General Python code → tests WITH GRAPH CONTEXT
    - MBPP: Programming problems with tests
    """
    samples = []
    graph_hits = 0  # Track how many samples got graph context
    
    print("="*50)
    print("LOADING GRAPH-RAG TEST GENERATION DATA")
    print("="*50)
    
    # 1. SWE-bench (bug-specific tests WITH GRAPH)
    print("\n1. Loading SWE-bench Train WITH GRAPH CONTEXT...")
    swe_count = 0
    try:
        swe_ds = load_dataset("princeton-nlp/SWE-bench", split="train")
        
        for item in swe_ds:
            if swe_count >= swe_limit:
                break
            
            patch = item.get('patch', '')
            problem = item.get('problem_statement', '')
            repo = item.get('repo', 'unknown')
            
            # Extract test code from patch
            test_code = extract_test_from_patch(patch)
            
            if test_code and len(test_code) > 50:
                # Extract function name and get graph context
                func_name = extract_function_name(patch)
                graph_context_str = ""
                
                if func_name and graph_extractor.graph:
                    context_data = graph_extractor.get_2hop_context(func_name)
                    if context_data:
                        graph_context_str = graph_extractor.format_graph_context(context_data)
                        graph_hits += 1
                
                # Build prompt WITH graph context
                prompt = f"""=== GRAPH-RAG TEST GENERATION ===

## Bug Report
Repository: {repo}
{problem[:1000]}

## Patch Context
{patch[:500]}

## Graph Context (2-Hop Neighborhood)
{graph_context_str if graph_context_str else "No graph context available"}

## Testing Requirements
1. Basic functionality tests
2. Edge cases (None, empty, boundary values)
3. Error handling (exceptions)
4. Test callers/callees if shown above

## Generated Tests
```python
{test_code}
```"""
                samples.append({'type': 'swe_bench', 'text': prompt})
                swe_count += 1
                
        print(f"   Loaded {swe_count} SWE-bench samples ({graph_hits} with graph context)")
        
    except Exception as e:
        print(f"   Error: {e}")
    
    # 2. Magicoder (general code → tests)
    print("\n2. Loading Magicoder...")
    magic_count = 0
    try:
        magic_ds = load_dataset("ise-uiuc/Magicoder-Evol-Instruct-110K", split="train")
        
        for item in magic_ds:
            if magic_count >= magic_limit:
                break
            
            instruction = item.get('instruction', '')
            response = item.get('response', '')
            
            # Filter for Python with functions
            if '```python' in response.lower() or 'def ' in response:
                # Extract code
                code_match = re.search(r'```python\s*(.*?)\s*```', response, re.DOTALL)
                if code_match:
                    code = code_match.group(1)
                else:
                    code = response
                
                # Extract function names
                func_names = re.findall(r'def (\w+)\(', code)
                
                if func_names and len(code) > 50:
                    # Generate test template based on the code
                    main_func = func_names[0]
                    
                    test_code = f'''import pytest

def test_{main_func}_basic():
    """Test basic functionality of {main_func}."""
    result = {main_func}()
    assert result is not None

def test_{main_func}_edge_cases():
    """Test edge cases."""
    # Test with edge inputs
    pass

def test_{main_func}_invalid_input():
    """Test error handling."""
    with pytest.raises(Exception):
        {main_func}(None)
'''
                    
                    prompt = format_code_to_test(code, test_code, instruction[:500])
                    samples.append({'type': 'magicoder', 'text': prompt})
                    magic_count += 1
                    
        print(f"   Loaded {magic_count} Magicoder samples")
        
    except Exception as e:
        print(f"   Error: {e}")
    
    # 3. MBPP (programming problems with tests)
    print("\n3. Loading MBPP...")
    mbpp_count = 0
    try:
        mbpp_ds = load_dataset("mbpp", split="train")
        
        for item in mbpp_ds:
            if mbpp_count >= mbpp_limit:
                break
            
            code = item.get('code', '')
            test_list = item.get('test_list', [])
            text = item.get('text', '')
            
            if code and test_list:
                # Combine tests
                test_code = '\n'.join(test_list)
                
                prompt = f"""=== COMPREHENSIVE TEST GENERATION ===

## Task Description
{text}

## Source Code
```python
{code}
```

## Generated Tests
```python
{test_code}
```"""
                
                samples.append({'type': 'mbpp', 'text': prompt})
                mbpp_count += 1
                
        print(f"   Loaded {mbpp_count} MBPP samples")
        
    except Exception as e:
        print(f"   Error: {e}")
    
    print(f"\n✅ Total samples: {len(samples)}")
    print(f"   SWE-bench: {swe_count}, Magicoder: {magic_count}, MBPP: {mbpp_count}")
    return samples

def extract_test_from_patch(patch: str) -> str:
    """Extract test code from patch."""
    test_content = []
    in_test_file = False
    
    lines = patch.split('\n')
    for line in lines:
        if line.startswith('diff --git') and ('test' in line.lower() or '_test' in line):
            in_test_file = True
        elif line.startswith('diff --git'):
            in_test_file = False
        elif in_test_file and line.startswith('+') and not line.startswith('+++'):
            test_content.append(line[1:])
    
    return '\n'.join(test_content) if test_content else None

# ============================================================
# CELL 5: Train Model
# ============================================================

def train_testgen(epochs=1, batch_size=1, swe_samples=400, magic_samples=400, mbpp_samples=200):
    """Train comprehensive test generation model."""
    
    print("="*60)
    print("COMPREHENSIVE TEST GENERATION TRAINING")
    print("="*60)
    
    # Load data
    samples = load_comprehensive_training_data(swe_samples, magic_samples, mbpp_samples)
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Tokenize
    print("Tokenizing...")
    def tokenize(sample):
        tokens = tokenizer(
            sample['text'], 
            truncation=True, 
            max_length=MAX_LENGTH, 
            padding="max_length",
            return_tensors=None
        )
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens
    
    tokenized = [tokenize(s) for s in samples]
    dataset = Dataset.from_list(tokenized)
    print(f"Dataset: {len(dataset)} samples")
    
    # Load model
    print("\nLoading model with 4-bit quantization...")
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16
    )
    
    model = prepare_model_for_kbit_training(model)
    
    # Add LoRA
    lora_config = LoraConfig(**LORA_CONFIG)
    model = get_peft_model(model, lora_config)
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    
    # Training
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    run_name = f"testgen_{datetime.now().strftime('%Y%m%d_%H%M')}"
    
    training_args = TrainingArguments(
        output_dir=f"{OUTPUT_DIR}/{run_name}",
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        warmup_ratio=0.1,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        fp16=True,
        optim="paged_adamw_8bit",
        report_to="none",
        gradient_checkpointing=True,
        remove_unused_columns=False,
    )
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )
    
    print("\n🚀 Starting comprehensive test generation training...")
    trainer.train()
    
    # Save
    final_path = f"{OUTPUT_DIR}/final"
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    
    # Save metadata
    with open(f"{final_path}/testgen_meta.json", 'w') as f:
        json.dump({
            'task': 'comprehensive_test_generation',
            'total_samples': len(samples),
            'swe_samples': swe_samples,
            'magic_samples': magic_samples,
            'mbpp_samples': mbpp_samples,
            'epochs': epochs
        }, f, indent=2)
    
    print(f"\n✅ TRAINING COMPLETE!")
    print(f"   Model: {final_path}")
    
    return model, tokenizer

# ============================================================
# CELL 6: Run Training
# ============================================================

model, tokenizer = train_testgen(
    epochs=1,
    batch_size=1,
    swe_samples=600,    # Bug-specific tests (real world)
    magic_samples=200,  # General code tests
    mbpp_samples=200    # Problem + test pairs
)

# ============================================================
# CELL 7: Test the Model - Generate Tests for ANY Code
# ============================================================

def generate_tests_for_code(model, tokenizer, code: str, context: str = ""):
    """Generate comprehensive tests for any Python code."""
    
    prompt = f"""=== COMPREHENSIVE TEST GENERATION ===

## Source Code
```python
{code}
```

## Context
{context if context else "Generate comprehensive unit tests for the above code."}
Include: basic tests, edge cases, error handling.

## Generated Tests
```python
"""
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
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
    
    return test_code.strip()

# Test with sample code
sample_code = '''
def calculate_discount(price, discount_percent):
    """Calculate discounted price."""
    if price < 0 or discount_percent < 0:
        raise ValueError("Values must be non-negative")
    if discount_percent > 100:
        raise ValueError("Discount cannot exceed 100%")
    return price * (1 - discount_percent / 100)
'''

print("="*50)
print("GENERATING TESTS FOR SAMPLE CODE:")
print("="*50)
print(sample_code)
print("\n" + "="*50)
print("GENERATED TESTS:")
print("="*50)
generated_tests = generate_tests_for_code(model, tokenizer, sample_code)
print(generated_tests)

# ============================================================
# CELL 8: Download
# ============================================================
"""
!zip -r /kaggle/working/testgen_comprehensive.zip /kaggle/working/testgen_model/final
"""
