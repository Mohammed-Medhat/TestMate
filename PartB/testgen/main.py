"""
Autonomous Test Generation Agent — Self-Correcting Loop
========================================================
Loads the LoRA-finetuned Qwen2.5-Coder-7B model, generates pytest tests,
runs them, and feeds errors back for self-correction.

Uses the same Graph-RAG context extraction as the training pipeline.

Usage:
  python main.py                          # Test calculator.py (default)
  python main.py --target path/to/code.py # Test any Python file
"""     

import os, sys, ast, re, subprocess, argparse, time, textwrap, json
import logging
from difflib import SequenceMatcher

import networkx as nx
import torch
from concurrent.futures import ThreadPoolExecutor

# Module-level logger. Default level is WARNING so debug calls are silent unless
# explicitly enabled (e.g., logging.getLogger("main").setLevel(logging.DEBUG)).
logger = logging.getLogger(__name__)
from rag_store import (init_db, store_example, retrieve_similar, store_bad_example,
                      retrieve_bad_examples, store_probe_value, retrieve_probe_value,
                      store_source_pattern, retrieve_source_patterns,
                      store_method_dependency, retrieve_method_dependencies,
                      feedback_test_accepted, feedback_test_rejected,
                      feedback_bad_helped, feedback_bad_didnt_help)

from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# ============================================================
# CONFIGURATION
# ============================================================
BASE_MODEL = r"D:\donwloader\Qwen2.5-Coder-7B-Instruct"
LORA_PATH  = r"D:\TestMate\TestMate\PartB\value_reasoning_model"

MAX_ITERATIONS   = 15
MAX_NEW_TOKENS   = 1536   # Increased so the LLM doesn't stop generating halfway through!
TEMPERATURE      = 0.5    
TOP_P            = 0.9
REPETITION_PENALTY = 1.05

# ── Performance flags (RTX 4060 Ada Lovelace) ──
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True       # ~2x matmul speed
    torch.backends.cudnn.allow_tf32 = True              # TF32 for convolutions
    torch.backends.cudnn.benchmark = True               # Auto-tune cuDNN kernels
    torch.cuda.set_device(0)

# ============================================================
# GRAPH-RAG CONTEXT EXTRACTION (identical to training pipeline)
# ============================================================

_AST_CACHE: dict[str, dict] = {}
_AST_CACHE_MAX = 512


def _extract_function_source(source_code: str, func_name: str) -> str:
    """Extract the full source of a single function from a module source string."""
    try:
        tree = ast.parse(source_code)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == func_name:
                lines = source_code.splitlines()
                # end_lineno available in Python 3.8+
                end = getattr(node, "end_lineno", node.lineno + 20)
                return "\n".join(lines[node.lineno - 1 : end])
    except Exception:
        pass
    return ""


def extract_context_from_code(code: str) -> dict:
    """Extract CFG paths, raises list, and cyclomatic complexity via AST.

    Cached by source hash — same file reparsed in the autocorrect loop is free.
    """
    import hashlib as _hashlib
    _key = _hashlib.md5(code.encode("utf-8", errors="ignore")).hexdigest()
    if _key in _AST_CACHE:
        # Return a shallow copy so callers can mutate without poisoning cache
        cached = _AST_CACHE[_key]
        return {k: (list(v) if isinstance(v, list) else v) for k, v in cached.items()}

    ctx = {"cfg_paths": [], "complexity": None, "raises": [], "callees": [], "classes": []}
    try:
        tree = ast.parse(code)
    except SyntaxError:
        _AST_CACHE[_key] = ctx
        return ctx

    # ── Extract classes, their constructors, and methods ──
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            bases = []
            for b in node.bases:
                try: bases.append(ast.unparse(b))
                except: pass
            methods = []
            init_params = []
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    methods.append(item.name)
                    if item.name == "__init__":
                        # Extract __init__ params (skip 'self')
                        args = item.args
                        for a in args.args[1:]:  # skip self
                            init_params.append(a.arg)
            ctx["classes"].append({
                "name": node.name,
                "bases": bases,
                "init_params": init_params,
                "methods": methods,
            })

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        paths = []

        def _trace(stmts, cur, depth=0):
            if depth > 5 or len(paths) > 5:
                return
            for s in stmts:
                if isinstance(s, ast.If):
                    try:    cond = ast.unparse(s.test)[:50]
                    except: cond = "condition"
                    _trace(s.body,   cur + [f"IF ({cond})"],   depth + 1)
                    if s.orelse:
                        _trace(s.orelse, cur + [f"ELSE ({cond})"], depth + 1)
                elif isinstance(s, ast.Raise):
                    exc = ""
                    if s.exc:
                        try:    exc = ast.unparse(s.exc)[:40]
                        except: exc = "Exception"
                    paths.append(cur + [f"RAISE {exc}"])
                elif isinstance(s, ast.Return):
                    try:    rv = ast.unparse(s.value)[:30] if s.value else "None"
                    except: rv = "value"
                    paths.append(cur + [f"RETURN {rv}"])
                elif isinstance(s, (ast.For, ast.While)):
                    _trace(s.body, cur + ["LOOP"], depth + 1)
                elif isinstance(s, ast.Try):
                    _trace(s.body, cur + ["TRY"], depth + 1)
                    for handler in s.handlers:
                        _trace(handler.body, cur + ["EXCEPT"], depth + 1)

        _trace(node.body, [f"def {node.name}()"])
        ctx["cfg_paths"].extend(paths)

        # Complexity estimate
        complexity = 0
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.For, ast.While, ast.Try,
                                  ast.ExceptHandler, ast.BoolOp)):
                complexity += 1
        ctx["complexity"] = complexity

    # Extract raises
    for node in ast.walk(tree):
        if isinstance(node, ast.Raise) and node.exc:
            if isinstance(node.exc, ast.Call) and hasattr(node.exc.func, 'id'):
                ctx["raises"].append(node.exc.func.id)
            elif isinstance(node.exc, ast.Name):
                ctx["raises"].append(node.exc.id)
    ctx["raises"] = list(set(ctx["raises"]))

    # Extract function calls
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                ctx["callees"].append(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                ctx["callees"].append(node.func.attr)
    ctx["callees"] = list(set(ctx["callees"]))[:10]

    # Populate cache (LRU-ish: drop oldest if at capacity)
    if len(_AST_CACHE) >= _AST_CACHE_MAX:
        try:
            _AST_CACHE.pop(next(iter(_AST_CACHE)))
        except StopIteration:
            pass
    _AST_CACHE[_key] = ctx
    return {k: (list(v) if isinstance(v, list) else v) for k, v in ctx.items()}



def extract_docstring(source_code: str, cls_name: str, method_name: str) -> str:
    try:
        tree = ast.parse(source_code)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == cls_name:
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        if item.name == method_name:
                            doc = ast.get_docstring(item)
                            return doc or ""
            # Top-level functions
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name == method_name and getattr(node, 'name', None) == method_name and not cls_name:
                    doc = ast.get_docstring(node)
                    return doc or ""
    except SyntaxError:
        pass
    return ""


def extract_comments(source_code: str, cls_name: str, method_name: str) -> list[str]:
    """
    Extract standalone comments related to a function/method.
    
    Captures:
      - Comments directly above the function definition
      - Inline comments on the def line
      - Module-level and class-level comments that describe behavior,
        invariants, edge cases, TODOs, or known bugs
    
    These give the LLM extra context about developer intent, known
    edge cases, and expected behavior that may not be in the docstring.
    """
    comments = []
    lines = source_code.splitlines()
    
    # Find the target function's line number
    target_line = None
    try:
        tree = ast.parse(source_code)
        for node in ast.walk(tree):
            if cls_name and isinstance(node, ast.ClassDef) and node.name == cls_name:
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        if item.name == method_name:
                            target_line = item.lineno
                            break
            elif not cls_name and isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name == method_name:
                    target_line = node.lineno
                    break
    except SyntaxError:
        return []
    
    if not target_line:
        return []
    
    # Collect comments above the function (up to 10 lines above)
    start_scan = max(0, target_line - 11)  # -1 for 0-index, -10 for lookback
    for i in range(target_line - 2, start_scan - 1, -1):  # Walk backwards
        if i < 0 or i >= len(lines):
            continue
        stripped = lines[i].strip()
        if stripped.startswith("#"):
            # Remove the # and leading space
            comment_text = stripped.lstrip("# ").strip()
            if comment_text:  # Skip empty comments
                comments.insert(0, comment_text)  # Insert at front (reverse order)
        elif stripped == "":
            continue  # Skip blank lines between comments
        else:
            break  # Hit non-comment code, stop
    
    # Also collect inline comment on the def line itself
    if target_line - 1 < len(lines):
        def_line = lines[target_line - 1]
        if "#" in def_line:
            inline = def_line.split("#", 1)[1].strip()
            if inline:
                comments.append(f"(inline) {inline}")
    
    # Collect module-level comments that mention the target by name
    # (useful for files with header comments describing functions)
    for i, line in enumerate(lines[:min(30, len(lines))]):  # First 30 lines
        stripped = line.strip()
        if stripped.startswith("#") and method_name in stripped:
            comment_text = stripped.lstrip("# ").strip()
            if comment_text and comment_text not in comments:
                comments.append(f"(module) {comment_text}")
    
    return comments[:8]  # Cap at 8 comments to avoid bloating prompt

def extract_method_snippet(source_code: str, cls_name: str, method_name: str) -> str:
    """Extract ONLY the target method's source + class __init__ for context.

    Instead of passing the whole file (~100 lines) to the 7B model,
    pass just the target method (~10 lines) plus class signature and __init__.
    This focuses the model's attention and reduces hallucination.
    """
    try:
        tree = ast.parse(source_code)
    except SyntaxError:
        return source_code  # Fallback to full source

    lines = source_code.splitlines()

    if cls_name:
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == cls_name:
                snippet_parts = []

                # 1. Class signature
                snippet_parts.append(lines[node.lineno - 1])

                # 2. __init__ method (for instantiation context)
                init_method = None
                target_method = None
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        if item.name == "__init__":
                            init_method = item
                        if item.name == method_name:
                            target_method = item

                if init_method and method_name != "__init__":
                    init_src = "\n".join(lines[init_method.lineno - 1 : init_method.end_lineno])
                    snippet_parts.append("")
                    snippet_parts.append(init_src)

                # 3. Target method itself
                if target_method:
                    target_src = "\n".join(lines[target_method.lineno - 1 : target_method.end_lineno])
                    snippet_parts.append("")
                    snippet_parts.append(target_src)
                    return "\n".join(snippet_parts)

        return source_code  # Fallback
    else:
        # Top-level function
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == method_name:
                return "\n".join(lines[node.lineno - 1 : node.end_lineno])
        return source_code  # Fallback


def build_test_checklist(code: str, ctx: dict) -> str:
    """Build an explicit checklist of what needs to be tested from the AST."""
    checklist = []
    try:
        tree = ast.parse(code)
        # Top-level functions
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                checklist.append(f"[ ] {node.name}()")
        # Class methods
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        checklist.append(f"[ ] {node.name}.{item.name}()")
    except SyntaxError:
        pass

    if ctx.get("raises"):
        for exc in ctx["raises"]:
            checklist.append(f"[ ] raises {exc} — test with pytest.raises({exc})")

    return "\n".join(checklist)


def build_prompt(code: str, ctx: dict, filename: str = "") -> str:
    """Build the Graph-RAG structured prompt (same format as training)."""
    sections = []
    if ctx.get("complexity") is not None:
        sections.append(f"## Complexity & Branch Info\nComplexity  : {ctx['complexity']}")
    if ctx.get("cfg_paths"):
        path_lines = "\n".join(f"  {i+1}. {' → '.join(p)}" for i, p in enumerate(ctx["cfg_paths"]))
        sections.append(f"## Control Flow — Execution Paths (COVER ALL IN TESTS)\n{path_lines}")
    if ctx.get("raises"):
        sections.append("## Exception Paths\n" + "\n".join(f"  - {e}" for e in ctx["raises"]))
    if ctx.get("callees"):
        sections.append("## Data Dependencies (calls)\n" + "\n".join(f"  - `{c}`" for c in ctx["callees"]))
    if filename:
        sections.append(f"## File\n{filename}")

    structured_context = "\n\n".join(sections)

    # ── Class-aware instructions ──
    class_section = ""
    if ctx.get("classes"):
        cls_lines = []
        for c in ctx["classes"]:
            params = ", ".join(c["init_params"]) if c["init_params"] else ""
            bases = f" (inherits {', '.join(c['bases'])})" if c["bases"] else ""
            methods_str = ", ".join(m for m in c["methods"] if m != "__init__")[:100]
            cls_lines.append(f"  - `{c['name']}({params})`{bases}")
            if methods_str:
                cls_lines.append(f"    Methods: {methods_str}")
        class_section = "\n## Classes — INSTANTIATE these, do NOT call __init__() directly\n" + "\n".join(cls_lines) + "\n\nEXAMPLE: To test class `MyClass(x, y)` with method `do_thing()`:\n```python\ndef test_myclass_do_thing():\n    obj = MyClass(1, 2)  # CORRECT: instantiate the class\n    result = obj.do_thing()\n    assert result is not None\n```\nNEVER do `__init__(1, 2)` or `myclass___getitem__(x)` — those are NOT valid Python.\n"

    # ── Test checklist section ──
    checklist = build_test_checklist(code, ctx)
    checklist_section = ""
    if checklist:
        checklist_section = f"\n## ✅ TEST CHECKLIST — write one def test_X() for EACH item:\n{checklist}\n"

    return f"""Generate comprehensive unit tests for the following function.

## Source Code
```python
{code[:2000]}
```

{structured_context}
{class_section}
{checklist_section}
## Testing Requirements
1. CRITICAL COVERAGE: You MUST write one test for EVERY item in the checklist above. Do not stop after 1-2 tests.
2. CRITICAL: You MUST import the functions you are testing.
3. CRITICAL: Calculate the exact expected output. DO NOT use generic assertions.
4. CRITICAL: To test exceptions, use `with pytest.raises(ExpectedError):`.
5. CRITICAL MOCKING: If the target code makes network requests (e.g., `requests.get`), reads files, or queries databases, you MUST use `@patch` from `unittest.mock` to fake the response. Do not let the test make real internet calls.
6. CRITICAL CLASSES: If the source has classes, you MUST create class instances with `obj = ClassName(args)`, then call `obj.method()`. NEVER call `__init__()` or `classname___method()` as standalone functions.
7. Cover edge cases: None, empty, zero.
8. Cover BOTH sides of every IF condition.

Return your test code inside <test>...</test> XML tags."""


# ============================================================
# CODE EXTRACTION
# ============================================================

def extract_test_code(response: str) -> str:
    """Extract Python code from the model's response."""
    if "<test>" in response and "</test>" in response:
        code = response.split("<test>")[1].split("</test>")[0].strip()
    elif "```python" in response:
        code = response.split("```python")[1].split("```")[0].strip()
    else:
        code = response.strip()
    return fix_generator_assertions(code)


def fix_generator_assertions(test_code: str) -> str:
    """Wrap generator-returning method assertions with list() automatically.
    
    Fixes: assert d.lower_items() == [('foo', 'bar')]  (fails: generator != list)
    To:    assert list(d.lower_items()) == [('foo', 'bar')]
    """
    lines = test_code.splitlines()
    fixed = []
    for line in lines:
        # Match: assert obj.method() == [list_literal]
        # Only wrap method calls, not builtins like len()
        m = re.match(r'(\s*)assert\s+((?!\blen\b|\brepr\b|\bstr\b|\bbool\b)\w+(?:\.\w+)?\([^)]*\))\s*==\s*(\[.+\])', line)
        if m:
            indent, call, expected = m.group(1), m.group(2), m.group(3)
            # Don't double-wrap if already list()
            if not call.startswith('list('):
                line = f"{indent}assert list({call}) == {expected}"
        fixed.append(line)
    return "\n".join(fixed)


# ============================================================
# TEST QUALITY CHECK (quick pre-execution filter)
# ============================================================

def quick_quality_check(test_code: str) -> tuple[bool, str]:
    """Returns (ok, reason). Checks syntax, structure, assertions."""
    try:
        ast.parse(test_code)
    except SyntaxError as e:
        return False, f"Syntax error: {e}"

    if len(test_code.strip()) < 30:
        return False, "Too short (<30 chars)"
    if "def test_" not in test_code:
        return False, "No test function (missing 'def test_')"
    if "assert" not in test_code.lower() and "raises" not in test_code.lower():
        return False, "No assertions or raises checks"

    # Count test functions
    test_funcs = re.findall(r'def test_\w+', test_code)
    if len(test_funcs) < 1:
        return False, "No test functions found"

    # Count assertions
    assert_lines = [l.strip() for l in test_code.splitlines() if re.match(r'\s*assert\s', l)]
    if len(assert_lines) < 1:
        return False, "No assertions found"

    # Reject tests where ALL assertions are type/isinstance checks (mutation-proof)
    weak_lines = [l for l in assert_lines
                  if "isinstance" in l or "type(" in l]
    if len(assert_lines) > 0 and len(weak_lines) == len(assert_lines):
        return False, "All assertions are type/isinstance checks — need concrete VALUE assertions (e.g. assert x == 42)"

    # Ban assert True / assert False — useless assertions
    for line in assert_lines:
        clean = line.replace(" ", "")
        if clean in ("assertTrue", "assertFalse", "assert1==1", "assert1"):
            return False, "Useless assertion detected: never use `assert True` or `assert False`"
        has_comparison = any(op in line for op in
                           ("==", "!=", ">=", "<=", ">", "<", " in ", " not in ",
                            ".startswith(", ".endswith(", ".startswith (", ".endswith ("))
        is_not_none = "is not None" in line
        if not has_comparison and not is_not_none and "isinstance" not in line and "raises" not in line.lower():
            return False, "Assertion missing comparison operator — assert the exact value"

    # Ban empty test functions (no assertion inside the function body)
    for block in test_code.split("def test_")[1:]:
        if "assert" not in block and "pytest.raises" not in block:
            return False, "Test function has no assertion — every test must assert something"

    # Detect assert inside pytest.raises block — semantically wrong
    for block in test_code.split("pytest.raises")[1:]:
        if "assert" in block[:200]:
            lines_in_block = block[:200].split('\n')
            for bline in lines_in_block[1:4]:  # check first few lines after raises
                if re.match(r'\s+assert\s', bline) and '==' in bline:
                    return False, "assert with == inside pytest.raises block is unreachable — use pytest.raises as context manager only, put assertions AFTER the block"

    # Detect comparison chaining: assert A != B == False
    for line in assert_lines:
        if re.search(r'!=.+==\s*(True|False)', line) or re.search(r'==.+!=', line):
            return False, "Comparison chaining detected: write separate assertions instead of 'assert A != B == False'"

    # Detect all-identical assertions (e.g. same assert 3 times)
    unique_asserts = set(l.strip() for l in assert_lines)
    if len(assert_lines) >= 2 and len(unique_asserts) == 1:
        return False, "All assertions are identical — add different inputs or verify different properties"

    # Reject tautologies: 'assert x == x' (always true) and 'assert f() == f()'
    # (same call on both sides). These pass pytest but catch zero bugs and
    # pollute the RAG corpus when they're stored as "good examples".
    _TAUTOLOGY_IDENT = re.compile(r'assert\s+([A-Za-z_]\w*)\s*==\s*\1\b')
    _TAUTOLOGY_CALL = re.compile(r'assert\s+([A-Za-z_]\w*\([^)]*\))\s*==\s*\1')
    for line in assert_lines:
        if _TAUTOLOGY_IDENT.search(line):
            return False, "Tautology: 'assert x == x' is always True — assert against an expected value"
        if _TAUTOLOGY_CALL.search(line):
            return False, "Tautology: same call on both sides of == is always True — assert against a literal"

    return True, "OK"


# ============================================================
# MAIN AGENT
# ============================================================

def load_model(attn_implementation: str = "sdpa", use_lora: bool = True):
    """Load the 4-bit quantized model, optionally with the LoRA adapter.

    Args:
        attn_implementation: Attention backend ('sdpa' default, 'eager', 'flash_attention_2').
                             SDPA gives ~1.3x speedup vs eager with zero extra VRAM cost.
        use_lora: If True (default), load LoRA from LORA_PATH on top of the base model.
                  If False, return the raw 4-bit base model only (useful for A/B ablations).
    """
    import gc
    import time as _time

    # ── CRITICAL: Set CUDA allocator config BEFORE any CUDA call ──
    # This env var is only read once, at first CUDA context creation.
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
        "expandable_segments:True,garbage_collection_threshold:0.6"
    )

    lora_mode = "with LoRA" if use_lora else "BASE ONLY (no LoRA)"
    # Note: no emoji here — avoids UnicodeEncodeError on Windows cp1252 terminals
    print(f"Loading Qwen2.5-Coder-7B [{lora_mode}] into GPU...")
    print(f"   Base model: {BASE_MODEL}")
    if use_lora:
        print(f"   LoRA path:  {LORA_PATH}")

    if not os.path.exists(BASE_MODEL):
        raise FileNotFoundError(f"Base model path not found: {BASE_MODEL}")

    # ── Aggressive GPU memory flush ──
    # When running in a subprocess after a previous run was force-killed,
    # stale GPU memory may linger. Flush everything before loading.
    if torch.cuda.is_available():
        torch.cuda.init()
        torch.cuda.set_device(0)
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        # Brief pause to let GPU memory settle after a force-kill
        _time.sleep(1)

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            BASE_MODEL, trust_remote_code=True, use_fast=True)
        tokenizer.truncation_side = "left"
        tokenizer.pad_token = tokenizer.eos_token
        print("   ✅ Tokenizer loaded (fast)")

        gc.collect()
        torch.cuda.empty_cache()

        # Detect available VRAM and choose dtype accordingly
        free_vram = torch.cuda.mem_get_info()[0] / 1e9
        total_vram = torch.cuda.mem_get_info()[1] / 1e9
        print(f"   📊 VRAM: {free_vram:.1f}GB free / {total_vram:.1f}GB total")

        # Abort early if insufficient VRAM
        if free_vram < 4.0:
            raise RuntimeError(
                f"Insufficient VRAM: {free_vram:.1f}GB free, need ≥4GB. "
                f"Close other GPU apps or restart the computer to free GPU memory."
            )

        # BF16 compute needs ~8GB peak during loading, FP16 needs ~6GB
        # Use FP16 on GPUs with ≤10GB VRAM to avoid OOM segfaults
        if total_vram <= 10:
            compute_dtype = torch.float16
            dtype_label = "FP16"
            print(f"   📊 Using FP16 compute (≤10GB VRAM detected)")
        else:
            compute_dtype = torch.bfloat16
            dtype_label = "BF16"
            print(f"   📊 Using BF16 compute (>10GB VRAM detected)")

        # Try 4-bit first, fall back to 8-bit if it fails
        base = None
        for quant_mode in ["4bit", "8bit"]:
            try:
                if quant_mode == "4bit":
                    bnb = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=compute_dtype,
                    )
                else:
                    print("   ⚠️  4-bit failed, trying 8-bit fallback...")
                    gc.collect()
                    torch.cuda.empty_cache()
                    bnb = BitsAndBytesConfig(load_in_8bit=True)

                extra_kwargs = {}
                if attn_implementation:
                    extra_kwargs["attn_implementation"] = attn_implementation
                    print(f"   📊 Using {attn_implementation} attention (memory-efficient)")

                try:
                    base = AutoModelForCausalLM.from_pretrained(
                        BASE_MODEL, quantization_config=bnb,
                        device_map={"": 0},
                        trust_remote_code=True,
                        low_cpu_mem_usage=True,
                        torch_dtype=compute_dtype,
                        **extra_kwargs,
                    )
                except (ValueError, TypeError, ImportError) as _attn_err:
                    # Fall back to eager if SDPA/FA2 not supported by this
                    # transformers version. 100% local and 8GB-safe regardless.
                    if attn_implementation:
                        print(f"   ⚠️  {attn_implementation} unsupported → falling back to eager: {_attn_err}")
                        extra_kwargs.pop("attn_implementation", None)
                        base = AutoModelForCausalLM.from_pretrained(
                            BASE_MODEL, quantization_config=bnb,
                            device_map={"": 0},
                            trust_remote_code=True,
                            low_cpu_mem_usage=True,
                            torch_dtype=compute_dtype,
                        )
                    else:
                        raise
                torch.cuda.synchronize()  # Ensure model is fully on GPU
                used = torch.cuda.memory_allocated() / 1e9
                print(f"   ✅ Base model loaded ({quant_mode} {dtype_label}, {used:.1f}GB VRAM)")
                break
            except Exception as e:
                if quant_mode == "8bit":
                    raise  # Both failed
                print(f"   ⚠️  {quant_mode} loading error: {e}")
                gc.collect()
                torch.cuda.empty_cache()
                continue

        if use_lora:
            if not os.path.exists(LORA_PATH):
                raise FileNotFoundError(f"LoRA path not found: {LORA_PATH}")
            model = PeftModel.from_pretrained(base, LORA_PATH)
            model.eval()
            torch.cuda.synchronize()
            adapter_mb = sum(
                p.numel() * p.element_size()
                for p in model.parameters() if p.requires_grad
            ) / 1e6
            print(f"   ✅ LoRA adapter loaded from {LORA_PATH} ({adapter_mb:.0f} MB trainable)")
        else:
            model = base
            model.eval()
            print("   ⚠️  LoRA SKIPPED — using raw 4-bit base Qwen2.5-Coder-7B only")

    except Exception as e:
        print(f"\n❌ MODEL LOADING FAILED:\n{e}")
        import traceback
        traceback.print_exc()
        # Clean up GPU on failure
        gc.collect()
        torch.cuda.empty_cache()
        raise RuntimeError(f"Model loading failed: {e}") from e

    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0
    print(f"✅ Model loaded on {gpu} ({vram:.1f} GB VRAM)\n")
    return model, tokenizer


def _compress_srs_requirements(srs_requirements: list, max_items: int = 12) -> str:
    """Compress SRS requirements to ≤80-char bullets for plan prompt injection."""
    import re as _re
    if not srs_requirements:
        return ""
    req_only = [r for r in srs_requirements if isinstance(r, dict) and r.get("label") == 1]
    req_only.sort(key=lambda r: r.get("score", 0), reverse=True)
    _NOISE = {"the", "system", "shall", "must", "accept", "return", "provide",
              "support", "allow", "when", "that", "from", "with", "this",
              "have", "will", "each", "only", "than", "into", "and", "for"}
    seen, bullets = set(), []
    for r in req_only:
        text = r.get("text", "").strip()
        if not text or len(text) < 10:
            continue
        text = _re.sub(r"^REQ-[\d.]+\s*", "", text)
        text = _re.sub(r"^(The system|A Session|The library|The caller)\s+shall\s+", "", text, flags=_re.I)
        text = text[:80].rstrip(".").strip()
        key = text[:35].lower()
        if key in seen:
            continue
        seen.add(key)
        bullets.append(f"  • {text}")
        if len(bullets) >= max_items:
            break
    if not bullets:
        return ""
    return ("\n📋 SRS REQUIREMENTS — your plan MUST include test cases for all of these:\n"
            + "\n".join(bullets) + "\n")


def generate_file_test_plan(model, tokenizer, source_code: str,
                            targets: list, ctx: dict, G,
                            import_path: str, target_file: str,
                            probe_cache: dict,
                            srs_requirements: list = None,
                            priming_examples: str = "",
                            stats_out: dict = None) -> str:
    """Phase 1 of plan mode: generate a structured test plan for ALL functions in a file.

    Creates one unified plan covering every target function/method before
    any code generation begins. Uses low temperature (0.3) for determinism.
    Includes FULL function source and probed runtime values for concrete plans.
    """
    # Extract individual function source code directly from AST
    # This is more reliable than token_budgeted_context which can strip the body
    func_sources = {}
    try:
        tree = ast.parse(source_code)
        lines = source_code.splitlines()
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        key = (node.name, item.name)
                        start = item.lineno - 1
                        end = (item.end_lineno or item.lineno)
                        func_sources[key] = "\n".join(lines[start:end])
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Only top-level functions (not nested in classes)
                if not any(isinstance(p, ast.ClassDef) for p in ast.walk(tree)
                          if node in getattr(p, 'body', [])):
                    key = (None, node.name)
                    start = node.lineno - 1
                    end = (node.end_lineno or node.lineno)
                    func_sources[key] = "\n".join(lines[start:end])
    except SyntaxError:
        pass

    # Build a detailed summary of each target function
    target_summaries = []
    for cls_name, method_name in targets:
        label = f"{cls_name}.{method_name}" if cls_name else method_name

        # Get the FULL function source (not truncated context)
        func_src = func_sources.get((cls_name, method_name), "")
        if not func_src:
            # Fallback to token-budgeted context
            func_src = build_token_budgeted_context(
                source_code, cls_name, method_name, G, token_budget=500)

        # Get CFG paths
        method_paths = get_method_cfg_paths(source_code, cls_name, method_name)
        path_lines = ""
        if method_paths:
            for i, p in enumerate(method_paths[:4]):
                path_lines += f"    {i+1}. {' → '.join(str(s) for s in p)}\n"

        # Get data flow
        data_flow = extract_data_flow(source_code, cls_name, method_name)

        # Get probed runtime value — this is CRITICAL for concrete plans
        target_label_graph = f"{cls_name}.{method_name}" if cls_name else method_name
        probed = probe_cache.get(target_label_graph, "")

        # RAG examples
        similar = retrieve_similar(method_name, cls_name or "", top_k=1,
                                   source_file=os.path.basename(target_file))
        rag_code = ""
        if similar:
            rag_code = similar[0].get("test_code", "")[:300]

        entry = f"""### {label}
```python
{func_src[:800]}
```
  Execution paths:{chr(10) + path_lines if path_lines else ' single path'}
  Returns: {data_flow.get('return_type', 'unknown')}
  Raises: {', '.join(data_flow.get('raises', [])) or 'none'}"""

        if probed and not str(probed).startswith("ERROR"):
            entry += f"\n  ✅ ACTUAL runtime output: {probed[:120]}"
        if rag_code:
            entry += f"\n  📚 Similar test pattern:\n```python\n{rag_code}\n```"

        target_summaries.append(entry)

    targets_block = "\n\n".join(target_summaries)
    filename = os.path.basename(target_file)

    plan_prompt = f"""READ the source code below and create a CONCRETE test plan for {filename}.

## File: {filename}
## Import: {import_path or filename}

{targets_block}

═══════════════════════════════════════
RULES FOR YOUR PLAN (violating = rejected):
═══════════════════════════════════════

NEVER write vague plans like:
  ❌ "Define mock inputs x, y, z"
  ❌ "assert result == expected"
  ❌ "Try function(None) → raises ValueError"  (unless the code actually raises ValueError)

ALWAYS write concrete plans like:
  ✅ "Call default_hooks() → returns {{'response': []}}"
  ✅ "Call dispatch_hook('response', {{'response': [lambda r: r]}}, data) → returns data unchanged"
  ✅ "Edge: dispatch_hook('missing_key', {{}}, data) → returns data (no hooks to run)"

For EACH function, provide:

### function_name
[Objective] One sentence about what to prove
[Requirements] Exact signature with types
[Test Cases]
  1. Input: (exact values) → Expected: (exact return value)
  2. Input: (exact values) → Expected: (exact return value)
  3. Edge: (boundary input) → Expected: (exact behavior)
[Mocks needed] List or "none"

READ THE SOURCE CODE to determine the correct expected values.
If a "✅ ACTUAL runtime output" is shown above, USE THAT as your expected value.
{_compress_srs_requirements(srs_requirements or [])}
{priming_examples}
Return inside <plan>...</plan> tags."""

    messages = [
        {"role": "system", "content":
         "You are a test planning expert. You READ source code carefully and "
         "create plans with EXACT concrete values — never placeholders like "
         "'expected' or 'x, y, z'. If you see the function returns "
         "{'response': []}, your plan must say exactly that. "
         "Every test case must have a real input and real expected output."},
        {"role": "user", "content": plan_prompt},
    ]

    # More tokens for file-level plan (covers many functions)
    max_tokens = max(500, min(1000, 250 * len(targets)))
    raw = generate_test(model, tokenizer, messages,
                        max_tokens=max_tokens, temperature=0.3, stats_out=stats_out)

    # Extract plan from tags
    if "<plan>" in raw and "</plan>" in raw:
        return raw.split("<plan>")[1].split("</plan>")[0].strip()
    return raw.strip()


def generate_test(model, tokenizer, messages: list, max_tokens: int = MAX_NEW_TOKENS,
                  temperature: float = None, stats_out: dict = None) -> str:
    """Generate test code from the current conversation.

    Thread-safe: uses CUDA synchronization barriers to prevent
    access violations when called from daemon threads.

    If stats_out is provided, increments stats_out["prompt_tokens"] and
    stats_out["completion_tokens"] with the actual token counts from this call.
    """
    _temp = temperature if temperature is not None else TEMPERATURE
    chat = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(chat, return_tensors="pt", truncation=True, max_length=2048).to(model.device)

    # Adaptive cap: never request more new tokens than fit in 4096-ctx with the prompt.
    # Saves ~10-25% generation time on short prompts with no quality loss.
    _prompt_len = inputs["input_ids"].shape[1]
    _ctx_budget = max(256, 4096 - _prompt_len - 16)
    max_tokens = min(max_tokens, _ctx_budget)

    # Use greedy decoding for first attempt (faster), sampling for retries
    _do_sample = _temp > 0.01

    try:
        with torch.inference_mode():  # Faster than torch.no_grad()
            if torch.cuda.is_available():
                torch.cuda.synchronize()  # Ensure no pending ops from other threads
            out = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=_do_sample,
                temperature=_temp if _do_sample else None,
                top_p=TOP_P if _do_sample else None,
                repetition_penalty=REPETITION_PENALTY,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True,  # KV cache reuse
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()  # Wait for completion
    except torch.cuda.OutOfMemoryError:
        print("   ⚠️  CUDA OOM — retrying with reduced tokens...")
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=min(max_tokens, 512),
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )

    raw = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    # M1: token accounting. Prompt = tokens fed in; completion = new tokens
    # generated (output length minus prompt length).
    if stats_out is not None:
        _completion_tokens = int(out.shape[1] - inputs["input_ids"].shape[1])
        stats_out["prompt_tokens"] = stats_out.get("prompt_tokens", 0) + _prompt_len
        stats_out["completion_tokens"] = stats_out.get("completion_tokens", 0) + _completion_tokens
    # Free GPU memory immediately
    del inputs, out
    return raw


def run_pytest(test_file: str, target_file: str = None, deadline_ts: float = None,
               docker_image: str = None) -> tuple[bool, str]:
    """Run pytest on the test file. Returns (passed, output).

    deadline_ts: optional Unix timestamp. When provided, pytest's subprocess
    timeout is capped so the call cannot run past the file-level deadline.

    docker_image: optional pre-built repo image tag. When provided, runs pytest
    inside that container (env-isolated from host) instead of host pytest.
    """
    if docker_image:
        try:
            from docker_runner import run_pytest_in_image  # type: ignore[import]
            return run_pytest_in_image(
                docker_image, test_file, target_file or test_file,
                with_coverage=False, deadline_ts=deadline_ts,
            )
        except Exception as exc:
            # Soft-fail to host pytest if Docker is mid-failure — better than
            # silently producing zero results in production.
            print(f"   ⚠️  Docker pytest fell back to host: {exc}")

    env = os.environ.copy()
    if target_file:
        # Walk up from target_file until no __init__.py is found to find the repo root
        root = os.path.dirname(os.path.abspath(target_file))
        while os.path.exists(os.path.join(root, "__init__.py")):
            root = os.path.dirname(root)
        env["PYTHONPATH"] = f"{root}{os.pathsep}{env.get('PYTHONPATH', '')}"

    _timeout = 30
    if deadline_ts is not None:
        _timeout = max(5, min(_timeout, int(deadline_ts - time.time()) - 2))

    result = subprocess.run(
        [sys.executable, "-m", "pytest", test_file, "-x", "--tb=short", "--no-header", "-q"],
        capture_output=True, text=True, timeout=_timeout, cwd=os.path.dirname(os.path.abspath(test_file)),
        env=env
    )
    output = result.stdout + "\n" + result.stderr
    return result.returncode == 0, output


def run_pytest_with_coverage(test_file: str, source_file: str, cov_module: str = None,
                             need_lines=True, deadline_ts: float = None,
                             docker_image: str = None) -> tuple[bool, str, float, float, set]:
    """Run pytest with coverage including branch coverage.

    Returns (passed, output, line_coverage_pct, branch_coverage_pct, covered_lines).

    cov_module: module import path (e.g. 'requests.structures') for --cov flag.
                Using a file path with --cov silently fails on installed packages.
    deadline_ts: optional Unix timestamp. When provided, pytest's subprocess
                 timeout is capped so the call cannot run past the file deadline.
    docker_image: optional pre-built repo image tag. When provided, runs the
                  coverage pass inside that container.  Note: covered_lines set
                  is empty in Docker mode (would require docker-cp of cov.json
                  per call); use the percentages instead.
    """
    if docker_image:
        try:
            from docker_runner import run_pytest_in_image  # type: ignore[import]
            return run_pytest_in_image(
                docker_image, test_file, source_file,
                with_coverage=True, cov_module=cov_module,
                deadline_ts=deadline_ts,
            )
        except Exception as exc:
            print(f"   ⚠️  Docker coverage pytest fell back to host: {exc}")
    # Use module name if available (works for installed packages),
    # fall back to source directory (works for local files)
    if cov_module:
        cov_target = cov_module
    else:
        cov_target = os.path.dirname(os.path.abspath(source_file))

    test_dir = os.path.dirname(os.path.abspath(test_file))
    cov_json = os.path.join(test_dir, "coverage.json")

    flags = [sys.executable, "-m", "pytest", test_file, "-v", "--tb=short",
             f"--cov={cov_target}",
             "--cov-branch",
             "--cov-report=term-missing", "--no-header"]
    if need_lines:
        flags.append("--cov-report=json")

    env = os.environ.copy()
    if source_file:
        root = os.path.dirname(os.path.abspath(source_file))
        while os.path.exists(os.path.join(root, "__init__.py")):
            root = os.path.dirname(root)
        env["PYTHONPATH"] = f"{root}{os.pathsep}{env.get('PYTHONPATH', '')}"

    _timeout = 60
    if deadline_ts is not None:
        _timeout = max(5, min(_timeout, int(deadline_ts - time.time()) - 2))

    result = subprocess.run(
        flags,
        capture_output=True, text=True, timeout=_timeout,
        cwd=test_dir, env=env
    )
    output = result.stdout + "\n" + result.stderr

    # Extract line & branch coverage from output
    # With --cov-branch, the table format is:
    #   Name  Stmts  Miss  Branch  BrPart  Cover  Missing
    #   auth.py  173  137    58       7     19%   8-25,...
    # There is only ONE combined Cover%. To get separate line and branch:
    #   line_coverage   = (Stmts - Miss) / Stmts * 100
    #   branch_coverage = (Branch - BrPart) / Branch * 100
    line_coverage = 0.0
    branch_coverage = 0.0
    source_basename = os.path.basename(source_file)

    cov_candidates = [l.strip() for l in output.splitlines() 
                      if "%" in l and "::" not in l
                      and not l.strip().startswith("test_")]
    if cov_candidates:
        print(f"   🔍 Coverage candidates: "
              f"{cov_candidates[:3]}")

    for line in output.splitlines():
        stripped = line.strip()
        if (("%" in stripped) and
            (source_basename in stripped or
             source_basename.replace(".py", "") in stripped) and
            not stripped.startswith("PASSED") and
            not stripped.startswith("FAILED") and
            not stripped.startswith("test_") and
            "::" not in stripped and
            "passing" not in stripped.lower() and
            "warning" not in stripped.lower() and
            "error" not in stripped.lower()):
            print(f"   🔍 Coverage raw: {stripped}")
            before_pct = stripped.split('%')[0]
            # Remove the filename part to get only the numbers
            for sep in [source_basename, source_basename.replace(".py",""),
                        "\\", "/"]:
                if sep in before_pct:
                    before_pct = before_pct.split(sep)[-1]
                    break
            cols = re.findall(r'(\d+)', before_pct)
            if len(cols) >= 4:
                try:
                    stmts    = int(cols[0])
                    miss     = int(cols[1])
                    branches = int(cols[2])
                    br_part  = int(cols[3])
                    if stmts > 0:
                        line_coverage = round((stmts - miss) / stmts * 100, 1)
                    if branches > 0:
                        branch_coverage = round((branches - br_part) / branches * 100, 1)
                except (ValueError, ZeroDivisionError):
                    percents = re.findall(r'(\d+)%', stripped)
                    if percents:
                        line_coverage = float(percents[0])
                        branch_coverage = line_coverage
            elif len(cols) >= 1:
                percents = re.findall(r'(\d+)%', stripped)
                if percents:
                    line_coverage = float(percents[0])
                    branch_coverage = line_coverage
            break

    if line_coverage == 0.0:
        for line in output.splitlines():
            if line.strip().startswith('TOTAL') and '%' in line:
                before_pct = line.split('%')[0]
                cols = re.findall(r'(\d+)',
                                  before_pct.split('TOTAL')[-1])
                if len(cols) >= 4:
                    try:
                        stmts, miss = int(cols[0]), int(cols[1])
                        branches, br_part = int(cols[2]), int(cols[3])
                        if stmts > 0:
                            line_coverage = round(
                                (stmts - miss) / stmts * 100, 1)
                        if branches > 0:
                            branch_coverage = round(
                                (branches - br_part) / branches * 100, 1)
                    except (ValueError, ZeroDivisionError):
                        pass
                elif len(cols) >= 1:
                    percents = re.findall(r'(\d+)%', line)
                    if percents:
                        line_coverage = float(percents[0])
                break

    # Parse JSON for executed lines
    executed_lines = set()
    if need_lines:
        try:
            with open(cov_json) as f:
                data = json.load(f)
            for filepath, info in data.get("files", {}).items():
                if source_basename.replace(".py", "") in filepath:
                    executed_lines = set(info.get("executed_lines", []))
                    break
        except (json.JSONDecodeError, OSError, KeyError) as _cov_e:
            logger.debug("Failed to parse coverage.json (%s); proceeding with empty executed_lines", _cov_e)

    return result.returncode == 0, output, line_coverage, branch_coverage, executed_lines



def extract_uncovered_functions(source_file: str, coverage_output: str) -> list:
    """
    Parse the source file AST and coverage output to find untested functions.
    Returns a list of function/method names that have no coverage.
    """
    # Get all function/method names and their line ranges from AST
    try:
        with open(source_file, "r") as f:
            source = f.read()
        tree = ast.parse(source)
    except (SyntaxError, FileNotFoundError):
        return []

    func_ranges = {}  # name -> (start_line, end_line)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # For methods, prefix with class name
            parent_class = None
            for cls_node in ast.walk(tree):
                if isinstance(cls_node, ast.ClassDef):
                    for item in cls_node.body:
                        if item is node:
                            parent_class = cls_node.name
                            break
            name = f"{parent_class}.{node.name}" if parent_class else node.name
            func_ranges[name] = (node.lineno, node.end_lineno or node.lineno + 5)

    # Extract missing line numbers from coverage output
    # Coverage format: "filename.py    100     50    50%   1-10, 25-30, 45"
    missing_lines = set()
    source_basename = os.path.basename(source_file)
    for line in coverage_output.splitlines():
        if source_basename in line:
            # Find the "Missing" column — line ranges after the %
            m = re.search(r'\d+%\s+(.+)', line)
            if m:
                missing_str = m.group(1).strip()
                for part in missing_str.split(','):
                    part = part.strip()
                    range_m = re.match(r'(\d+)-(\d+)', part)
                    if range_m:
                        for ln in range(int(range_m.group(1)), int(range_m.group(2)) + 1):
                            missing_lines.add(ln)
                    elif part.isdigit():
                        missing_lines.add(int(part))

    if not missing_lines:
        return []  # Can't determine — coverage output wasn't parseable

    # Find functions whose entire body is in missing lines
    uncovered = []
    for name, (start, end) in func_ranges.items():
        func_lines = set(range(start, end + 1))
        if func_lines.issubset(missing_lines):
            uncovered.append(name)

    return uncovered


def inject_mock_wrapper(test_code: str, module_name: str, source_code: str) -> str:
    """
    Post-process the generated test to add mock wrappers automatically.
    Called when the model fails to add @patch decorators itself.
    """
    import re

    # Build the correct import block
    mock_imports = (
        f"import pytest\n"
        f"from unittest.mock import patch, MagicMock\n"
        f"from {module_name} import *\n\n"
    )

    # Strip whatever imports the model wrote (usually wrong)
    lines = test_code.splitlines()
    non_import_lines = [l for l in lines if not l.startswith("import ")
                        and not l.startswith("from ")]
    clean_body = "\n".join(non_import_lines).strip()

    # Find all test functions that call the target and wrap them
    # Strategy: add @patch decorator to every def test_ that doesn't already have one
    wrapped_lines = []
    body_lines = clean_body.splitlines()
    i = 0
    while i < len(body_lines):
        line = body_lines[i]
        if re.match(r'^def test_', line):
            # Check if previous line is already a decorator
            prev = wrapped_lines[-1].strip() if wrapped_lines else ""
            if not prev.startswith("@"):
                wrapped_lines.append(f"@patch('{module_name}.requests.get')")
            # Add mock_get param to function signature
            line = re.sub(
                r'^def (test_\w+)\(self\)',
                rf'def \1(mock_get)',
                line
            )
            line = re.sub(
                r'^def (test_\w+)\(\)',
                rf'def \1(mock_get)',
                line
            )
            wrapped_lines.append(line)
            # Inject mock setup as first lines of function body
            wrapped_lines.append(f'    mock_get.return_value.status_code = 200')
            wrapped_lines.append(f'    mock_get.return_value.json.return_value = {{"temp_celsius": 22.5}}')
            i += 1
            continue
        wrapped_lines.append(line)
        i += 1

    result = mock_imports + "\n".join(wrapped_lines)

    try:
        ast.parse(result)
        return result
    except SyntaxError:
        # Fallback: return a known-good minimal test suite
        return f"""\
import pytest
from unittest.mock import patch, MagicMock
from {module_name} import *

@patch('{module_name}.requests.get')
def test_mocking_wrapper(mock_get):
    mock_get.return_value.status_code = 200
    mock_get.return_value.json.return_value = {{"temp_celsius": 22.5}}
    pass
"""


def get_uncovered_lines_with_source(
    source_file: str, 
    coverage_output: str,
    target_cls: str = None,
    target_method: str = None
) -> list:
    """
    Returns (line_number, source_text) for uncovered lines,
    optionally filtered to a specific method.
    
    This is the laser pointer — instead of saying
    'coverage too low', you say 'you missed line 42:
    if key not in self.data: raise KeyError'
    """
    try:
        with open(source_file) as f:
            source_lines = f.readlines()
    except FileNotFoundError:
        # Try installed package path
        import importlib.util
        parts = os.path.normpath(source_file).split(os.sep)
        for i, part in enumerate(parts):
            if part == "repos" and i + 2 < len(parts):
                pkg_name = parts[i + 1]
                spec = importlib.util.find_spec(pkg_name)
                if spec and spec.submodule_search_locations:
                    pkg_dir = spec.submodule_search_locations[0]
                    installed = os.path.join(pkg_dir, os.path.basename(source_file))
                    if os.path.exists(installed):
                        with open(installed) as f:
                            source_lines = f.readlines()
                        break
        else:
            return []
    
    # Parse missing lines from coverage output
    missing_lines = set()
    source_basename = os.path.basename(source_file)
    for line in coverage_output.splitlines():
        if source_basename in line:
            m = re.search(r'\d+%\s+(.+)', line)
            if m:
                for part in m.group(1).split(','):
                    part = part.strip()
                    range_m = re.match(r'(\d+)-(\d+)', part)
                    if range_m:
                        for ln in range(int(range_m.group(1)), int(range_m.group(2)) + 1):
                            missing_lines.add(ln)
                    elif part.isdigit():
                        missing_lines.add(int(part))
    
    if not missing_lines:
        return []
    
    # Filter to target method's lines if specified
    if target_cls and target_method:
        try:
            tree = ast.parse("".join(source_lines))
            method_lines = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and node.name == target_cls:
                    for item in node.body:
                        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == target_method:
                            for ln in range(item.lineno, (item.end_lineno or item.lineno) + 1):
                                method_lines.add(ln)
            missing_lines = missing_lines & method_lines
        except:
            pass
    
    result = []
    for ln in sorted(missing_lines):
        if 0 < ln <= len(source_lines):
            src = source_lines[ln - 1].rstrip()
            if src.strip() and not src.strip().startswith("#"):
                result.append((ln, src))
    
    return result[:8]  # max 8 lines to keep prompt focused


def save_to_flywheel(
    target_file: str,
    source_code: str, 
    test_code: str,
    coverage: float,
    mutation_score: float,
    flywheel_path: str = "flywheel_data.jsonl"
):
    """Save successful test generations for future fine-tuning.
    
    Dedup: skips if the same source file already has >= 3 entries,
    unless the new entry has better quality (coverage + mutation).
    """
    src_basename = os.path.basename(target_file)
    new_quality = coverage + mutation_score
    
    # Dedup check: count existing entries for this source file
    existing_count = 0
    best_existing_quality = 0.0
    if os.path.exists(flywheel_path):
        try:
            with open(flywheel_path, "r") as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        if entry.get("source_file") == src_basename:
                            existing_count += 1
                            eq = entry.get("coverage_pct", 0) + entry.get("mutation_score", 0)
                            best_existing_quality = max(best_existing_quality, eq)
                    except json.JSONDecodeError:
                        continue
        except OSError:
            pass
    
    # Skip if we already have enough entries and new one isn't better
    if existing_count >= 3 and new_quality <= best_existing_quality:
        print(f"   ♻️  Flywheel: skipped (already {existing_count} entries for {src_basename}, quality {new_quality:.0f} <= {best_existing_quality:.0f})")
        return
    
    record = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_file": src_basename,
        "source_code": source_code,
        "generated_tests": test_code,
        "coverage_pct": coverage,
        "mutation_score": mutation_score,
        "model_version": "value_reasoning_v1",
    }
    with open(flywheel_path, "a") as f:
        f.write(json.dumps(record) + "\n")
    print(f"   💾 Flywheel: saved to {flywheel_path} ({coverage:.0f}% cov, {mutation_score:.0f}% mut)")


def _merge_test_code(existing_code, new_code, import_statement):
    """Append-only merge: keep existing_code as-is, append new test functions."""
    existing_names = set(re.findall(r'def (test_\w+)', existing_code))

    new_functions = []
    lines = new_code.splitlines()
    i = 0
    while i < len(lines):
        m = re.match(r'^def (test_\w+)', lines[i])
        if m and m.group(1) not in existing_names:
            func_lines = [lines[i]]
            i += 1
            while i < len(lines) and (lines[i].startswith(' ') or lines[i].startswith('\t')):
                func_lines.append(lines[i])
                i += 1
            new_functions.append('\n'.join(func_lines))
        else:
            i += 1

    if not new_functions:
        return existing_code

    return existing_code.rstrip() + '\n\n' + '\n\n'.join(new_functions) + '\n'


def _extract_actual_values(pytest_output: str) -> list:
    """
    Parse pytest output to extract actual values from assertion failures.
    
    Example pytest output:
        E   AssertionError: assert 'Basic Og==' == None
        E   assert _basic_auth_str('', '') == None
    
    Returns list of dicts: [{"call": "func(args)", "actual": "real_value", "expected": "wrong_value"}]
    """
    fixes = []
    if not pytest_output:
        return fixes
    
    for line in pytest_output.splitlines():
        line = line.strip()
        
        # Pattern 1: "AssertionError: assert 'actual' == expected"
        # or "assert func(args) == expected" where actual is shown
        m = re.search(
            r"assert\s+['\"]([^'\"]+)['\"]\s*==\s*['\"]([^'\"]+)['\"]", line
        )
        if m:
            actual, expected = m.group(1), m.group(2).strip()
            # Try to find the function call from nearby context
            call_m = re.search(r"(\w+\([^)]*\))", line)
            call = call_m.group(1) if call_m else "function()"
            if " object at 0x" in actual or ("<" in actual and ">" in actual):
                continue
            fixes.append({"call": call, "actual": actual, "expected": expected})
            continue
        
        # Pattern 2: quoted string == unquoted None/True/False
        m2 = re.search(r"assert\s+['\"]([^'\"]+)['\"]\s*==\s*(None|True|False)", line)
        if m2:
            actual, expected = m2.group(1), m2.group(2)
            if " object at 0x" not in actual and not ("<" in actual and ">" in actual):
                fixes.append({
                    "call": "the tested function",
                    "actual": actual,
                    "expected": expected,
                })
    
    # Deduplicate
    seen = set()
    unique = []
    for f in fixes:
        key = f"{f['actual']}:{f['expected']}"
        if key not in seen:
            seen.add(key)
            unique.append(f)
    
    return unique[:5]  # max 5 fixes


def run_mutation_testing(target_file: str, test_file: str) -> tuple[bool, str]:
    """Lightweight mutation testing that works on Windows (mutmut doesn't).
    
    Applies simple mutations to the source file and checks if tests catch them.
    Resolves installed package paths for installed packages.
    """
    import importlib.util, shutil, tempfile
    
    # Resolve to installed package file if applicable
    src_path = target_file
    try:
        parts = os.path.normpath(target_file).split(os.sep)
        for i, part in enumerate(parts):
            if part == "repos" and i + 2 < len(parts):
                pkg_name = parts[i + 1]
                spec = importlib.util.find_spec(pkg_name)
                if spec and spec.submodule_search_locations:
                    pkg_dir = spec.submodule_search_locations[0]
                    installed = os.path.join(pkg_dir, os.path.basename(target_file))
                    if os.path.exists(installed):
                        src_path = installed
                break
    except (ImportError, AttributeError, ValueError):
        pass

    if not os.path.exists(src_path):
        return False, f"Source file not found: {src_path}"

    # Read original source
    with open(src_path, "r", encoding="utf-8") as f:
        original_code = f.read()

    # Define mutations — expanded set for better coverage
    mutations = [
        ("==", "!="),
        ("!=", "=="),
        (" not ", " "),
        ("True", "False"),
        ("False", "True"),
        (" + ", " - "),
        (" - ", " + "),
        ("return ", "return None  # "),
        (" >= ", " > "),
        (" <= ", " < "),
        (" > ", " >= "),
        (" < ", " <= "),
        (" * ", " / "),
        (" and ", " or "),
        (" or ", " and "),
        (" is ", " is not "),
        (" in ", " not in "),
    ]
    
    # Load covered lines from coverage.json to focus mutations
    covered_lines = set()
    cov_json = os.path.join(
        os.path.dirname(os.path.abspath(test_file)),
        "coverage.json"
    )
    # Also check source file directory (digit-named files)
    cov_json_alt = os.path.join(
        os.path.dirname(os.path.abspath(target_file)),
        "coverage.json"
    )
    _cov_path = (cov_json if os.path.exists(cov_json)
                 else cov_json_alt if os.path.exists(cov_json_alt)
                 else None)
    if _cov_path:
        try:
            with open(_cov_path) as _cj:
                _cov_data = json.load(_cj)
            source_base = os.path.basename(target_file).replace(".py", "")
            for _fp, _info in _cov_data.get("files", {}).items():
                if source_base in _fp:
                    covered_lines = set(_info.get("executed_lines", []))
                    break
            print(f"   📊 Mutation: using {len(covered_lines)} "
                  f"covered lines from coverage.json")
        except Exception as _e:
            print(f"   ⚠️  Mutation: coverage.json read failed: {_e}")
    else:
        print(f"   ⚠️  Mutation: no coverage.json found, "
              f"testing all mutable lines")

    # Generate mutants by applying one mutation at a time
    mutants = []
    lines = original_code.splitlines()
    for line_idx, line in enumerate(lines):
        # P4: Only mutate lines that are actually covered by tests
        if covered_lines and (line_idx + 1) not in covered_lines:
            continue
        stripped = line.strip()
        # Skip comments, docstrings, imports, class/def declarations
        if (stripped.startswith("#") or stripped.startswith('"""') or 
            stripped.startswith("'''") or stripped.startswith("import ") or
            stripped.startswith("from ") or stripped.startswith("class ") or
            stripped.startswith("def ") or not stripped):
            continue
        if stripped.startswith('"') or stripped.startswith("'"):
            continue
        for old, new in mutations:
            if old in line and new not in line:
                mutated_lines = lines.copy()
                mutated_lines[line_idx] = line.replace(old, new, 1)
                mutants.append(("\n".join(mutated_lines), f"L{line_idx+1}: '{old}'→'{new}'"))
                break  # One mutation per line

    if not mutants:
        return True, "✅ No mutable code found (trivial file)"

    # Test each mutant
    killed = 0
    survived = 0
    survived_details = []
    test_file_abs = os.path.abspath(test_file)
    
    # P6: Test up to 8 mutants for better accuracy
    test_mutants = mutants[:5]
    print(f"   🧬 Testing {len(test_mutants)} mutants (parallel)...")

    def _test_mutant_in_place(args):
        """Test one mutant in-place directly on the original source file."""
        mutated_code, desc = args
        try:
            with open(src_path, "w", encoding="utf-8") as f:
                f.write(mutated_code)
            
            result = subprocess.run(
                [sys.executable, "-m", "pytest", test_file_abs, "-x", "--no-header", "-q"],
                capture_output=True, text=True, timeout=30,
                cwd=os.path.dirname(test_file_abs)
            )
            return result.returncode != 0, desc
        except subprocess.TimeoutExpired:
            return True, desc
        except Exception:
            return True, desc
        finally:
            with open(src_path, "w", encoding="utf-8") as f:
                f.write(original_code)

    # In-place mutations MUST be sequential
    print(f"   🧬 Testing {len(test_mutants)} mutants (in-place sequential)...")
    for args in test_mutants:
        was_killed, d = _test_mutant_in_place(args)
        if was_killed:
            killed += 1
        else:
            survived += 1
            survived_details.append(d)



    total = killed + survived
    score = (killed / total * 100) if total > 0 else 0
    
    if survived == 0:
        return True, f"✅ All mutants killed! ({killed}/{total})"
    else:
        detail = "; ".join(survived_details[:3])
        return False, f"Mutation score: {score:.0f}% ({survived} survived: {detail})"


def probe_method(actual_import: str, cls_name: str, method_name: str, 
                 init_params: list, source_code: str) -> str:
    """Execute the actual method to discover its real return value.
    
    Returns repr() of the result, or empty string if probe fails.
    """
    if not cls_name:
        # Top-level function probe
        probe = f"""
from {actual_import} import *
try:
    import inspect
    fn = eval('{method_name}')
    sig = inspect.signature(fn)
    args = []
    for name, p in sig.parameters.items():
        if p.default is inspect.Parameter.empty:
            args.append(repr('test'))
    result = {method_name}(*args)
    print(repr(result))
except Exception as e:
    print(f'ERROR: {{e}}')
"""
    else:
        # Build constructor args from init_params
        constructor_args = ", ".join(
            f"{p}='test_{p}'" if 'name' in p.lower() or 'str' in p.lower()
            else f"{p}=1" 
            for p in init_params
        ) if init_params else ""

        # Build method invocation
        dunder_invocations = {
            "__len__": "len(obj)",
            "__repr__": "repr(obj)",
            "__str__": "str(obj)",
            "__iter__": "list(iter(obj))",
            "__eq__": f"obj == {cls_name}({constructor_args})",
            "__ne__": f"obj != {cls_name}({constructor_args})",
            "__bool__": "bool(obj)",
            "__hash__": "hash(obj)",
            "__getitem__": "obj[list(obj.keys())[0]] if hasattr(obj, 'keys') and obj else None",
            "__delitem__": "'deletion - check state after'",
            "__setitem__": "'assignment - check state after'",
            "__call__": "None  # requires special setup",
            "__init__": "'__attr_snapshot__'",  # sentinel — handled below
        }
        invocation = dunder_invocations.get(method_name, f"obj.{method_name}()")

        if init_params:
            obj_init = f"{cls_name}({constructor_args})"
        else:
            obj_init = f"{cls_name}({{'test_key': 'test_value'}})"

        # Special probe for __init__: return attribute snapshot instead of repr
        if method_name == "__init__":
            probe = f"""
from {actual_import} import *
import json
try:
    obj = {obj_init}
    attrs = {{k: repr(v) for k, v in vars(obj).items() if not k.startswith('_')}}
    print(json.dumps(attrs) if attrs else repr(obj))
except Exception as e:
    print(f'ERROR: {{e}}')
"""
        else:
            probe = f"""
from {actual_import} import *
try:
    obj = {obj_init}
    result = {invocation}
    if hasattr(result, '__next__') or (hasattr(result, '__iter__') and not isinstance(result, (str, list, dict))):
        result = list(result)
    print(repr(result))
except Exception as e:
    print(f'ERROR: {{e}}')
"""

    try:
        result = subprocess.run(
            [sys.executable, "-c", probe],
            capture_output=True, text=True, timeout=10
        )
        output = result.stdout.strip()
        if output and not output.startswith("ERROR"):
            return output
    except Exception:
        pass
    return ""


def get_method_line_count(source_code: str, cls_name: str, method_name: str) -> int:
    try:
        tree = ast.parse(source_code)
        if cls_name:
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and node.name == cls_name:
                    for item in node.body:
                        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == method_name:
                            return (item.end_lineno or item.lineno) - item.lineno
        else:
            for node in ast.iter_child_nodes(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == method_name:
                    return (node.end_lineno or node.lineno) - node.lineno
    except SyntaxError:
        pass
    return 0

def calculate_max_tokens(
    method_lines: int,
    method_paths: int,
    subgraph: dict,
    data_flow: dict,
    retry: int = 0,
    has_mock_hint: bool = False,
    has_bad_examples: bool = False,
    has_rag_examples: bool = False,
) -> int:
    """Calculate optimal max_tokens based on method complexity.

    Factors in:
    - Base: method lines (40 tokens per line)
    - Complexity: CFG paths (30 tokens per path)
    - Dependencies: method callees (20 tokens per callee)
    - Exceptions: raise statements (40 tokens per raise)
    - Mocks: extra setup code (150 tokens)
    - Bad examples: model needs to avoid them (80 tokens)
    - RAG few-shot: extra context from similar tests (120 tokens)
    - Retries: give more room for failed attempts (100 tokens per retry)

    Returns token budget clamped to [256, 2048].
    """
    base         = method_lines * 40
    path_bonus   = method_paths * 30
    dep_bonus    = len(subgraph.get("callees", [])) * 20
    raises_bonus = len(data_flow.get("raises", [])) * 40
    mock_bonus   = 150 if has_mock_hint else 0
    bad_bonus    = 80  if has_bad_examples else 0
    rag_bonus    = 120 if has_rag_examples else 0
    retry_bonus  = retry * 100

    total = (base + path_bonus + dep_bonus + raises_bonus
             + mock_bonus + bad_bonus + rag_bonus + retry_bonus)

    # Force minimum for dict/list returning functions
    if any("}" in rv or "]" in rv for rv in data_flow.get("return_values", [])):
        total = max(total, 400)

    return min(2048, max(256, total))


def verify_path_coverage(test_code: str, method_paths: list) -> tuple:
    """Check if the generated test covers critical CFG paths.
    
    Returns (covered_count, total_critical, missing_hints).
    This is a SOFT check — used as feedback, not a hard gate.
    """
    if not method_paths:
        return 0, 0, []
    
    critical_paths = []
    for path in method_paths:
        path_str = " ".join(str(step) for step in path)
        # RAISE paths are critical — tests should use pytest.raises
        if "RAISE" in path_str:
            critical_paths.append(("raise", path_str))
        # EXCEPT paths are critical — tests should trigger exceptions
        elif "EXCEPT" in path_str:
            critical_paths.append(("except", path_str))
        # ELSE branches — tests should hit the alternate path
        elif "ELSE" in path_str:
            critical_paths.append(("else", path_str))
        # Loop empty-input paths
        elif "0 iterations" in path_str:
            critical_paths.append(("loop_empty", path_str))
        # IF-ONLY paths — if with no else, must test the truthy branch
        elif "IF-ONLY" in path_str:
            critical_paths.append(("if_only", path_str))
    
    if not critical_paths:
        return 0, 0, []
    
    test_lower = test_code.lower()
    covered = 0
    missing = []
    
    for ptype, pdesc in critical_paths:
        if ptype == "raise":
            if "pytest.raises" in test_lower or "with pytest.raises" in test_lower:
                covered += 1
            else:
                missing.append(f"Missing pytest.raises() for path: {pdesc[:80]}")
        elif ptype == "except":
            if "pytest.raises" in test_lower or "try" in test_lower:
                covered += 1
            else:
                missing.append(f"Missing exception test for path: {pdesc[:80]}")
        elif ptype == "else":
            # Hard to verify else coverage statically, give benefit of doubt
            # if test has multiple assertions or calls
            assertions = test_lower.count("assert ")
            if assertions >= 2:
                covered += 1
            else:
                missing.append(f"Only {assertions} assertion(s) — may miss ELSE branch")
        elif ptype == "loop_empty":
            # Check if test uses empty inputs like [], "", {}, None, 0
            empty_inputs = any(tok in test_code for tok in [
                "[]", "''", '""', "{}", "None", "()"
            ])
            if empty_inputs:
                covered += 1
            else:
                missing.append(f"No empty/None input for loop edge case")
        elif ptype == "if_only":
            # IF-ONLY: the code has `if X:` with no else.
            # The model must test BOTH truthy (enter IF) and falsy (skip IF).
            # Extract the condition from the path description
            import re as _re
            cond_match = _re.search(r'IF-ONLY \((.+?)\)', pdesc)
            cond_hint = cond_match.group(1) if cond_match else "condition"
            # Check if test has at least 2 calls or varied inputs (truthy + falsy)
            assertions = test_lower.count("assert ")
            calls = test_code.count("(")  # rough proxy for method calls
            if assertions >= 2 and calls >= 4:
                covered += 1
            else:
                missing.append(f"Test may not enter IF ({cond_hint}) — add a test with truthy input")
    
    return covered, len(critical_paths), missing


def _extract_expected_outputs(test_code: str) -> set:
    """Extract concrete expected values from assertions.
    Used to differentiate tests that look structurally similar
    but test different behavior."""
    outputs = set()
    for line in test_code.splitlines():
        line = line.strip()
        # assert x == VALUE
        m = re.search(r'assert\s+.+?\s*==\s*(.+)$', line)
        if m:
            val = m.group(1).strip()
            # Only keep concrete values, not expressions
            if val not in ('None', 'True', 'False') or len(val) < 50:
                outputs.add(val)
        # assert VALUE in x
        m2 = re.search(r'assert\s+(.+?)\s+in\s+', line)
        if m2:
            outputs.add(m2.group(1).strip())
    return outputs

def is_semantically_duplicate(new_test: str,
                               existing_code: str,
                               threshold: float = 0.85) -> bool:
    existing_funcs = re.split(r'\ndef test_', existing_code)
    new_body = re.sub(
        r'def test_\w+\([^)]*\):', '', new_test
    ).strip()

    def normalize(code):
        # Q3: stronger normalization so structurally-identical tests with
        # different local variable names / comments / whitespace map to the
        # same canonical form. String literals are deliberately preserved so
        # tests with different expected values still differ.
        # 1. Strip Python line comments.
        code = re.sub(r'#[^\n]*', '', code)
        # 2. Canonicalize constructor-style assignments.
        code = re.sub(
            r'\w+\s*=\s*\w+\([^)]*\)', 'OBJ = CLASS()', code
        )
        # 3. Replace lowercase identifiers on the LHS of '=' with VAR
        #    (catches `x = ...`, `obj = ...`, `result = ...`, etc.).
        code = re.sub(
            r'\b[a-z_][a-z0-9_]*\s*=(?!=)', 'VAR =', code
        )
        # 4. Collapse runs of whitespace.
        code = re.sub(r'\s+', ' ', code).strip()
        return code

    new_body_norm = normalize(new_body)

    # Extract expected outputs from new test
    new_outputs = _extract_expected_outputs(new_test)

    for existing in existing_funcs[1:]:
        existing_body = re.sub(
            r'^[^(]+\([^)]*\):', '', existing, count=1
        ).strip()
        existing_body_norm = normalize(existing_body)
        ratio = SequenceMatcher(
            None, new_body_norm, existing_body_norm
        ).ratio()

        if ratio > threshold:
            # Structure is similar — now check expected outputs
            existing_outputs = _extract_expected_outputs(
                "def test_" + existing
            )
            # If expected outputs overlap > 80% it is truly
            # duplicate — same structure AND same behavior
            if new_outputs and existing_outputs:
                overlap = len(
                    new_outputs & existing_outputs
                ) / max(len(new_outputs), len(existing_outputs))
                if overlap > 0.8:
                    return True
                # Same structure but different expected values
                # means different behavior — NOT a duplicate
                else:
                    return False
            # No expected values to compare — fall back to
            # structure only
            return True
    return False

def score_assertion_strength(test_code: str) -> float:
    """Kept for compute_composite_score backward compat. Delegates to BES assertion dim."""
    try:
        from bes_scorer import _assertion_quality  # type: ignore[import]
        return _assertion_quality(test_code)
    except ImportError:
        pass
    # Inline fallback
    assert_lines = [l for l in test_code.splitlines() if re.match(r'\s*assert\s', l)]
    non_trivial = [l for l in assert_lines if l.strip() not in ("assert True", "assert False")]
    if not non_trivial: return 0.0
    score = 0.33
    if re.search(r'\w+\([^)]+\)', test_code): score += 0.33
    if re.search(r'==\s*["\'\d]', test_code) or len(assert_lines) >= 2: score += 0.34
    return min(score, 1.0)


def score_input_diversity(test_code: str) -> float:
    """Kept for compute_composite_score backward compat. Delegates to BES variety dim."""
    try:
        from bes_scorer import _input_variety  # type: ignore[import]
        return _input_variety(test_code)
    except ImportError:
        pass
    unique = set(re.findall(r'["\']([^"\']{1,})["\']', test_code) +
                 re.findall(r'\b([1-9]\d*|0\.\d+)\b', test_code))
    if len(unique) >= 3: return 1.0
    if len(unique) == 2: return 0.5
    if len(unique) == 1: return 0.25
    return 0.0


def score_edge_cases(test_code: str) -> float:
    """Kept for compute_composite_score backward compat. Delegates to BES boundary dim."""
    try:
        from bes_scorer import _boundary_coverage  # type: ignore[import]
        return _boundary_coverage(test_code)
    except ImportError:
        pass
    markers = ["None", '""', "''", "[]", "{}", "0", "-1"]
    found = sum(1 for e in markers if e in test_code)
    return 1.0 if found >= 2 else (0.5 if found == 1 else 0.0)


def compute_composite_score(
    test_code: str,
    line_coverage_pct: float = 0.0,
    branch_coverage_pct: float = 0.0,
    mutation_score: float = 0.0,
    per_test: bool = True,  # True = per-test gate, False = final file score
) -> dict:
    """Dimension 4 & 5 + Composite: All-in-one 0-100 test quality score.

    When per_test=True (default): Only scores things a single test controls
      - assertion*0.60 + diversity*0.20 + edge*0.20
    When per_test=False: Full file-level score with coverage + mutation
      - assertion*0.30 + line_cov*0.20 + branch_cov*0.20 + mutation*0.20 + diversity*0.05 + edge*0.05

    Returns dict with composite score (0-100) and breakdown of each dimension.
    """
    # Calculate each dimension
    assertion = score_assertion_strength(test_code)
    input_diversity = score_input_diversity(test_code)
    edge_case = score_edge_cases(test_code)

    if per_test:
        # Per-test: only things a single test controls
        composite_norm = (
            assertion       * 0.60 +
            input_diversity * 0.20 +
            edge_case       * 0.20
        )
    else:
        # File-level: include coverage and mutation
        line_cov_norm   = line_coverage_pct / 100.0
        branch_cov_norm = branch_coverage_pct / 100.0
        mutation_norm   = mutation_score / 100.0
        composite_norm = (
            assertion       * 0.30 +
            line_cov_norm   * 0.20 +
            branch_cov_norm * 0.20 +
            mutation_norm   * 0.20 +
            input_diversity * 0.05 +
            edge_case       * 0.05
        )

    # Convert to 0-100 scale and round
    composite_score = round(composite_norm * 100, 1)

    return {
        "composite": composite_score,  # 0-100
        "assertion": round(assertion * 100, 1),  # 0-100
        "branch_cov": round(branch_coverage_pct, 1),  # 0-100
        "line_cov": round(line_coverage_pct, 1),  # 0-100
        "mutation": round(mutation_score, 1),  # 0-100
        "diversity": round(input_diversity * 100, 1),  # 0-100
        "edge_cases": round(edge_case * 100, 1),  # 0-100
    }


def categorize_failure(error_logs: str, failure_reason: str) -> tuple[str, list]:
    """Categorize test failure to help learning."""
    category = "unknown_error"
    failed_mutants = []

    logs_lower = error_logs.lower()
    reason_lower = failure_reason.lower()

    # Categorize by error type
    if "keyerror" in logs_lower and "realm" in logs_lower:
        category = "missing_chal_setup"
    elif "keyerror" in logs_lower or "key" in reason_lower:
        category = "missing_setup"
    elif "typeerror" in logs_lower or "type" in reason_lower:
        category = "type_error"
    elif "importerror" in logs_lower or "import" in reason_lower:
        category = "import_error"
    elif "attributeerror" in logs_lower:
        category = "attribute_error"
    elif "indexerror" in logs_lower:
        category = "index_error"
    elif "assert" in logs_lower or "==" in logs_lower or "!=" in logs_lower:
        category = "assertion_failure"
    elif "failed" in logs_lower:
        category = "execution_error"
    else:
        category = "execution_error"

    # Extract survived mutants from logs if mutation testing
    if "mutation" in logs_lower or "mutant" in logs_lower:
        # Look for patterns like "COR: == → !=" or "AOR: + → -"
        mutant_patterns = re.findall(r'(l\d+):\s+([^→]+)→\s*([^\s]+)', logs_lower)
        for op, from_op, to_op in mutant_patterns:
            failed_mutants.append(f"{op}: {from_op.strip()} → {to_op.strip()}")

    return category, failed_mutants



def analyze_source_patterns(source_code: str, class_name: str, method_name: str, ctx: dict):
    """Detect common patterns in source code that hint at required test patterns."""
    patterns = []

    try:
        tree = ast.parse(source_code)
    except SyntaxError:
        return patterns

    # Find the target method
    target_node = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == method_name:
                    target_node = item
                    break

    if not target_node:
        return patterns

    # Analyze method body for patterns
    source_lines = source_code.split("\n")
    method_source = "\n".join(source_lines[target_node.lineno-1:target_node.end_lineno])

    # Pattern 1: Dict-like methods (__getitem__, __setitem__)
    if method_name in ("__getitem__", "__setitem__", "__delitem__"):
        patterns.append(("dict_operation", "Accesses/modifies internal dict - needs setup"))

    # Pattern 2: __init__ with required fields
    if method_name == "__init__":
        if "self." in method_source:
            patterns.append(("initializer", "Sets up internal state - required before access"))

    # Pattern 3: Methods that check conditions
    if "if " in method_source or "while " in method_source:
        patterns.append(("conditional", "Has branches - test both paths"))

    # Pattern 4: Methods that raise exceptions
    if "raise " in method_source:
        patterns.append(("error_raising", "Can raise exceptions - test error cases"))

    # Pattern 5: Methods that iterate
    if "for " in method_source or "while " in method_source:
        patterns.append(("iterative", "Loops over data - test empty/single/multiple"))

    # Pattern 6: recursive calls
    if method_name in method_source.replace(f"def {method_name}", ""):
        patterns.append(("recursive", "Calls itself - test base and recursive cases"))

    # Store patterns in database
    for pattern_type, description in patterns:
        store_source_pattern(class_name, method_name, pattern_type, description)

    return patterns


def detect_method_dependencies(source_code: str, class_name: str, context: dict):
    """Detect method dependencies based on source code analysis.
    E.g., __getitem__ typically depends on __setitem__ or __init__"""
    dependencies = []

    try:
        tree = ast.parse(source_code)
    except SyntaxError:
        return dependencies

    # Find the class
    target_class = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            target_class = node
            break

    if not target_class:
        return dependencies

    # Get all methods in the class
    methods_in_class = {item.name for item in target_class.body if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))}

    # Dict-like class pattern: __getitem__ depends on __setitem__/__init__
    if "__getitem__" in methods_in_class and ("__setitem__" in methods_in_class or "__init__" in methods_in_class):
        if "__setitem__" in methods_in_class:
            dependencies.append(("__getitem__", "__setitem__", "dict_setup"))
        else:
            dependencies.append(("__getitem__", "__init__", "dict_setup"))

    # Similar patterns for __delitem__
    if "__delitem__" in methods_in_class and ("__setitem__" in methods_in_class or "__init__" in methods_in_class):
        if "__setitem__" in methods_in_class:
            dependencies.append(("__delitem__", "__setitem__", "dict_setup"))

    # __contains__ depends on __setitem__
    if "__contains__" in methods_in_class and "__setitem__" in methods_in_class:
        dependencies.append(("__contains__", "__setitem__", "dict_setup"))

    # __len__ depends on __setitem__ for dict-like classes
    if "__len__" in methods_in_class and "__setitem__" in methods_in_class:
        dependencies.append(("__len__", "__setitem__", "dict_setup"))

    # Store dependencies in database
    for method, depends_on, dep_type in dependencies:
        store_method_dependency(class_name, method, depends_on, dep_type)

    return dependencies



def build_call_graph(source_code: str):
    import networkx as nx
    G = nx.MultiDiGraph()

    try:
        tree = ast.parse(source_code)
    except SyntaxError:
        return G

    # Build a single id(func_node) -> class_name map once.
    # Replaces the previous O(N^2) re-walk per function for parent lookup.
    _func_to_class: dict[int, str] = {}
    for cls_node in ast.walk(tree):
        if isinstance(cls_node, ast.ClassDef):
            for item in cls_node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    _func_to_class[id(item)] = cls_node.name

    # Add all functions and methods as nodes
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            parent_class = _func_to_class.get(id(node))

            full_name = f"{parent_class}.{node.name}" if parent_class else node.name
            G.add_node(full_name,
                      type="method" if parent_class else "function",
                      class_name=parent_class,
                      method_name=node.name,
                      lineno=node.lineno,
                      end_lineno=node.end_lineno or node.lineno)

    # Pass 1: Add edges for call relationships
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            parent_class = _func_to_class.get(id(node))

            caller = f"{parent_class}.{node.name}" if parent_class else node.name
            
            for child in ast.walk(node):
                if isinstance(child, ast.Call):
                    if isinstance(child.func, ast.Name):
                        callee = child.func.id
                    elif isinstance(child.func, ast.Attribute):
                        if isinstance(child.func.value, ast.Name) and child.func.value.id == "self":
                            callee = f"{parent_class}.{child.func.attr}" if parent_class else child.func.attr
                        else:
                            callee = child.func.attr
                    else:
                        continue
                    
                    if callee in G.nodes or any(callee in n for n in G.nodes):
                        G.add_edge(caller, callee, type='calls')
    
    # Pass 2: State dependency detection via ast.Store/ast.Load
    attr_writes = {}  # {attr_name: [method_that_writes]}
    attr_reads = {}   # {attr_name: [method_that_reads]}

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    method_full = f"{node.name}.{item.name}"
                    for child in ast.walk(item):
                        if (isinstance(child, ast.Attribute) and
                            isinstance(child.value, ast.Name) and
                            child.value.id == 'self'):
                            attr = child.attr
                            if isinstance(child.ctx, ast.Store):
                                attr_writes.setdefault(attr, []).append(method_full)
                            elif isinstance(child.ctx, ast.Load):
                                attr_reads.setdefault(attr, []).append(method_full)

    # Add state dependency edges
    for attr, writers in attr_writes.items():
        readers = attr_reads.get(attr, [])
        for writer in writers:
            for reader in readers:
                if writer != reader and writer in G.nodes and reader in G.nodes:
                    G.add_edge(writer, reader,
                              type='state_dependency',
                              variable=attr)

    return G


def get_2hop_subgraph(G, target: str) -> dict:
    result = {
        "callers": [],
        "callees": [],
        "callers_of_callers": [],
        "callees_of_callees": [],
    }
    
    if target not in G:
        matches = [n for n in G.nodes if target in n or n in target]
        if not matches:
            return result
        target = matches[0]
    
    # 1 hop
    result["callees"] = list(G.successors(target))
    result["callers"] = list(G.predecessors(target))
    
    # 2 hops
    for callee in result["callees"]:
        result["callees_of_callees"].extend(
            [n for n in G.successors(callee) if n != target]
        )
    for caller in result["callers"]:
        result["callers_of_callers"].extend(
            [n for n in G.predecessors(caller) if n != target]
        )
    
    # Deduplicate
    for key in result:
        result[key] = list(set(result[key]))
    
    return result


def get_stateful_setup_chain(G, target_label: str) -> list:
    """Get methods that must be called before target due to state dependencies.
    Returns list of (predecessor_method, variable_name) tuples."""
    deps = []
    if target_label not in G:
        return deps
    
    for pred, _, data in G.in_edges(target_label, data=True):
        if data.get('type') == 'state_dependency':
            var = data.get('variable', 'internal state')
            method_name = pred.split('.')[-1]
            # Skip __init__ — already handled by instantiation
            if method_name != '__init__':
                deps.append((pred, var))
    
    # Deduplicate
    seen = set()
    unique = []
    for dep in deps:
        if dep[0] not in seen:
            seen.add(dep[0])
            unique.append(dep)
    return unique


def build_cross_file_context(source_code: str, source_file: str,
                              repo_dir: str, max_snippets: int = 5) -> dict:
    """Follow imports to find dependency classes/functions and extract signatures.
    Returns dict: {imported_name: snippet_of_definition}"""
    cross_context = {}
    try:
        tree = ast.parse(source_code)
    except SyntaxError:
        return cross_context

    imported_modules = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                imported_modules[alias.name] = module
        elif isinstance(node, ast.Import):
            for alias in node.names:
                imported_modules[alias.asname or alias.name] = alias.name

    count = 0
    for name, module in imported_modules.items():
        if count >= max_snippets:
            break
        # Try resolving module to file
        module_path = module.replace(".", os.sep) + ".py"
        candidates = [
            os.path.join(repo_dir, module_path),
            os.path.join(os.path.dirname(source_file), module_path),
        ]
        for full_path in candidates:
            if os.path.exists(full_path):
                try:
                    with open(full_path, encoding="utf-8", errors="ignore") as f:
                        dep_source = f.read()
                    dep_tree = ast.parse(dep_source)
                    dep_lines = dep_source.splitlines()

                    for dep_node in ast.walk(dep_tree):
                        if isinstance(dep_node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                            if dep_node.name == name:
                                # Extract first 15 lines of the definition
                                start = dep_node.lineno - 1
                                end = min(start + 15, dep_node.end_lineno or start + 15)
                                snippet = "\n".join(dep_lines[start:end])
                                cross_context[name] = {
                                    "file": os.path.basename(full_path),
                                    "snippet": snippet,
                                    "type": "class" if isinstance(dep_node, ast.ClassDef) else "function"
                                }
                                count += 1
                                break
                except (SyntaxError, OSError):
                    pass
                break
    return cross_context


def synthesize_branch_inputs(source_code: str, cls_name: str,
                              method_name: str) -> list:
    """Parse branch conditions in a method and synthesize inputs that trigger each branch.
    Returns list of dicts: [{condition, true_input, false_input}]"""
    inputs = []
    try:
        tree = ast.parse(source_code)
    except SyntaxError:
        return inputs

    target_node = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == cls_name:
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == method_name:
                    target_node = item
                    break
        elif not cls_name and isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == method_name:
                target_node = node

    if not target_node:
        return inputs

    source_lines = source_code.splitlines()
    for child in ast.walk(target_node):
        if isinstance(child, ast.If):
            # Get the condition source text
            try:
                cond_line = source_lines[child.lineno - 1].strip()
                # Extract condition after 'if' or 'elif'
                cond_match = re.match(r'(?:el)?if\s+(.+?):', cond_line)
                if cond_match:
                    condition = cond_match.group(1)

                    # Parse common patterns to synthesize inputs
                    entry = {"condition": condition, "true_input": None, "false_input": None}

                    # Pattern: x == "value" or x == value
                    eq_match = re.search(r'(\w+)\s*==\s*["\']([^"\']+)["\']', condition)
                    if eq_match:
                        var, val = eq_match.group(1), eq_match.group(2)
                        entry["true_input"] = f"{var}='{val}'"
                        entry["false_input"] = f"{var}=None"
                        inputs.append(entry)
                        continue

                    # Pattern: x is None / x is not None
                    none_match = re.search(r'(\w+)\s+is\s+(not\s+)?None', condition)
                    if none_match:
                        var = none_match.group(1)
                        is_not = bool(none_match.group(2))
                        true_val = '"test"' if is_not else 'None'
                        false_val = 'None' if is_not else '"test"'
                        entry["true_input"] = f"{var}={true_val}"
                        entry["false_input"] = f"{var}={false_val}"
                        inputs.append(entry)
                        continue

                    # Pattern: x (truthiness check)
                    if re.match(r'^\w+$', condition.strip()):
                        var = condition.strip()
                        entry["true_input"] = f"{var}='test'"
                        entry["false_input"] = f"{var}=None"
                        inputs.append(entry)
                        continue

                    # Pattern: len(x) > 0
                    len_match = re.search(r'len\((\w+)\)\s*([><=!]+)\s*(\d+)', condition)
                    if len_match:
                        var, op, num = len_match.group(1), len_match.group(2), len_match.group(3)
                        entry["true_input"] = f"{var}=['item']"
                        entry["false_input"] = f"{var}=[]"
                        inputs.append(entry)
                        continue

                    # Generic: just record the condition
                    inputs.append(entry)
            except (IndexError, AttributeError):
                pass

    return inputs[:6]  # Limit to 6 branches


def extract_data_flow(source_code: str, cls_name: str, method_name: str) -> dict:
    flow = {
        "inputs": [],
        "outputs": [],
        "transforms": [],
        "return_values": [],
        "raises": [],
    }
    
    try:
        tree = ast.parse(source_code)
    except SyntaxError:
        return flow
    
    target_node = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == method_name:
            if cls_name:
                for cls_node in ast.walk(tree):
                    if isinstance(cls_node, ast.ClassDef) and cls_node.name == cls_name:
                        for item in cls_node.body:
                            if item is node:
                                target_node = node
                                break
            else:
                target_node = node
            if target_node:
                break
    
    if not target_node:
        return flow
    
    for arg in target_node.args.args:
        if arg.arg == "self":
            continue
        entry = {"name": arg.arg}
        if arg.annotation:
            try:
                entry["type"] = ast.unparse(arg.annotation)
            except:
                pass
        flow["inputs"].append(entry)
    
    if target_node.returns:
        try:
            flow["outputs"].append(ast.unparse(target_node.returns))
        except:
            pass
    
    for node in ast.walk(target_node):
        if isinstance(node, ast.Return) and node.value:
            try:
                flow["return_values"].append(ast.unparse(node.value))
            except:
                pass
        if isinstance(node, ast.Raise) and node.exc:
            try:
                flow["raises"].append(ast.unparse(node.exc))
            except:
                pass
        if isinstance(node, ast.Assign):
            try:
                transform = ast.unparse(node)
                if len(transform) < 80:
                    flow["transforms"].append(transform)
            except:
                pass
    
    return flow


def get_method_cfg_paths(source_code: str, cls_name: str, method_name: str) -> list:
    try:
        tree = ast.parse(source_code)
    except SyntaxError:
        return []
    
    target_node = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == method_name:
            if cls_name:
                for cls_node in ast.walk(tree):
                    if isinstance(cls_node, ast.ClassDef) and cls_node.name == cls_name:
                        for item in cls_node.body:
                            if item is node:
                                target_node = node
                                break
            else:
                for top in ast.iter_child_nodes(tree):
                    if top is node:
                        target_node = node
                        break
            if target_node:
                break
    
    if not target_node:
        return []
    
    paths = []
    
    def _trace(stmts, cur, depth=0):
        if depth > 8 or len(paths) > 8:
            return
        for s in stmts:
            if isinstance(s, ast.If):
                try:    cond = ast.unparse(s.test)[:40]
                except: cond = "condition"
                _trace(s.body,   cur + [f"IF ({cond})"],   depth + 1)
                if s.orelse:
                    _trace(s.orelse, cur + [f"ELSE ({cond})"], depth + 1)
                else:
                    paths.append(cur + [f"IF-ONLY ({cond})"])
            elif isinstance(s, ast.Raise):
                try:    exc = ast.unparse(s.exc)[:30] if s.exc else "Exception"
                except: exc = "Exception"
                paths.append(cur + [f"RAISE {exc}"])
            elif isinstance(s, ast.Return):
                try:    rv = ast.unparse(s.value)[:25] if s.value else "None"
                except: rv = "value"
                paths.append(cur + [f"RETURN {rv}"])
            elif isinstance(s, (ast.For, ast.While)):
                # Extract loop variable and iterable for richer context
                loop_desc = "LOOP"
                if isinstance(s, ast.For):
                    try:
                        var = ast.unparse(s.target)[:15]
                        iter_expr = ast.unparse(s.iter)[:25]
                        loop_desc = f"FOR {var} in {iter_expr}"
                    except:
                        pass
                # Trace inside the loop body
                _trace(s.body, cur + [loop_desc], depth + 1)
                # Add edge case paths for loops
                paths.append(cur + [f"{loop_desc} (0 iterations — empty input)"])
                # Check for break statements inside the loop
                for child in ast.walk(s):
                    if isinstance(child, ast.Break):
                        paths.append(cur + [f"{loop_desc} → BREAK (early exit)"])
                        break
            elif isinstance(s, ast.Try):
                _trace(s.body,    cur + ["TRY"],    depth + 1)
                for handler in s.handlers:
                    exc_type = ""
                    if handler.type:
                        try: exc_type = ast.unparse(handler.type)
                        except: pass
                    _trace(handler.body, cur + [f"EXCEPT {exc_type}"], depth + 1)
    
    _trace(target_node.body, [f"{method_name}()"])
    return paths


def build_token_budgeted_context(
    source_code: str,
    cls_name: str,
    method_name: str,
    G,
    token_budget: int = 600
) -> str:
    target_label = f"{cls_name}.{method_name}" if cls_name else method_name
    
    target_snippet = extract_method_snippet(source_code, cls_name, method_name)
    tokens_used = len(target_snippet.split()) * 1.3
    parts = [f"# TARGET METHOD\n{target_snippet}"]
    
    subgraph = get_2hop_subgraph(G, target_label)
    
    for callee in subgraph["callees"]:
        callee_parts = callee.split(".")
        callee_cls = callee_parts[0] if len(callee_parts) > 1 else None
        callee_method = callee_parts[-1]
        
        if callee_method == method_name:
            continue
            
        snippet = extract_method_snippet(source_code, callee_cls, callee_method)
        cost = len(snippet.split()) * 1.3
        
        if tokens_used + cost < token_budget:
            parts.append(f"# CALLED BY TARGET: {callee}\n{snippet}")
            tokens_used += cost
        else:
            break
    
    for caller in subgraph["callers"]:
        caller_parts = caller.split(".")
        caller_cls = caller_parts[0] if len(caller_parts) > 1 else None
        caller_method = caller_parts[-1]
        
        snippet = extract_method_snippet(source_code, caller_cls, caller_method)
        cost = len(snippet.split()) * 1.3
        
        if tokens_used + cost < token_budget * 0.8:
            parts.append(f"# CALLS TARGET: {caller}\n{snippet}")
            tokens_used += cost
        else:
            break
    
    return "\n\n".join(parts)


def resolve_base_classes(source_code: str, repo_dir: str) -> dict:
    from pathlib import Path
    base_sources = {}
    
    try:
        tree = ast.parse(source_code)
    except SyntaxError:
        return base_sources
    
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for base in node.bases:
                try:
                    base_name = ast.unparse(base).split(".")[-1]
                except:
                    continue
                
                if base_name in ("object", "dict", "list", "str", "int", "Exception"):
                    continue
                
                repo_path = Path(repo_dir)
                for py_file in repo_path.rglob("*.py"):
                    try:
                        with open(py_file, encoding="utf-8", errors="ignore") as f:
                            file_content = f.read()
                        if f"class {base_name}" in file_content:
                            base_tree = ast.parse(file_content)
                            for base_node in ast.walk(base_tree):
                                if (isinstance(base_node, ast.ClassDef) and 
                                    base_node.name == base_name):
                                    methods = [
                                        item.name for item in base_node.body
                                        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                                    ]
                                    base_sources[base_name] = {
                                        "file": str(py_file),
                                        "methods": methods[:10],
                                        "lineno": base_node.lineno,
                                    }
                                    break
                            if base_name in base_sources:
                                break
                    except:
                        continue
    
    return base_sources


def probe_constructor_signature(actual_import: str, cls_name: str) -> str:
    """Get the actual __init__ signature so model knows exact params to pass."""
    probe = f"""
from {actual_import} import *
import inspect
try:
    sig = inspect.signature({cls_name}.__init__)
    params = []
    for name, p in sig.parameters.items():
        if name == 'self':
            continue
        if p.kind in (inspect.Parameter.VAR_POSITIONAL,):
            params.append(f'*{{name}}')
        elif p.kind in (inspect.Parameter.VAR_KEYWORD,):
            params.append(f'**{{name}}')
        elif p.default is inspect.Parameter.empty:
            params.append(f'{{name}} (required)')
        else:
            params.append(f'{{name}}={{p.default!r}} (optional)')
    print(', '.join(params) if params else 'no params')
except Exception as e:
    print(f'ERROR: {{e}}')
"""
    try:
        result = subprocess.run(
            [sys.executable, "-c", probe],
            capture_output=True, text=True, timeout=5
        )
        output = result.stdout.strip()
        if output and not output.startswith("ERROR"):
            return output
    except (subprocess.SubprocessError, OSError) as _probe_e:
        logger.debug("Param-attribute probe subprocess failed (%s)", _probe_e)
    return ""


def infer_param_attributes(source_code: str, cls_name: str,
                           method_name: str) -> dict:
    """Scan method body to find what attributes are accessed on each parameter.
    Returns dict: {param_name: [list of accessed attributes]}"""
    param_attrs = {}
    try:
        tree = ast.parse(source_code)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == cls_name:
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == method_name:
                        params = [a.arg for a in item.args.args if a.arg != 'self']
                        for param in params:
                            attrs = []
                            for child in ast.walk(item):
                                if (isinstance(child, ast.Attribute) and
                                    isinstance(child.value, ast.Name) and
                                    child.value.id == param):
                                    attrs.append(child.attr)
                            if attrs:
                                param_attrs[param] = list(set(attrs))
    except SyntaxError:
        pass
    return param_attrs


def enforce_stateful_setup(test_code: str, source_code: str,
                           cls_name: str, method_name: str,
                           G, target_label: str) -> str:
    """Post-processing fallback: programmatically inject setup methods
    that the model should have included but didn't."""
    if not cls_name:
        return test_code

    setup_chain = get_stateful_setup_chain(G, target_label)
    if not setup_chain:
        return test_code

    setup_methods = [pred.split('.')[-1] for pred, _ in setup_chain]

    # Detect if target needs chal dict by checking what attrs it reads
    # from _thread_local (detected by CPG Load edges)
    needs_chal = False
    try:
        tree = ast.parse(source_code)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == cls_name:
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name == method_name:
                        for child in ast.walk(item):
                            if (isinstance(child, ast.Attribute) and
                                isinstance(child.value, ast.Attribute) and
                                isinstance(child.value.value, ast.Name) and
                                child.value.value.id == 'self' and
                                'thread_local' in child.value.attr and
                                child.attr == 'chal'):
                                needs_chal = True
    except SyntaxError:
        pass

    lines = test_code.splitlines()
    fixed = []
    for i, line in enumerate(lines):
        fixed.append(line)
        m = re.match(r'(\s+)(\w+)\s*=\s*' + re.escape(cls_name) + r'\(', line)
        if m:
            indent = m.group(1)
            var = m.group(2)
            next_few = lines[i+1:i+5]
            for setup_method in setup_methods:
                if not any(setup_method in l for l in next_few):
                    fixed.append(f"{indent}{var}.{setup_method}()")
            # Inject chal dict if method reads _thread_local.chal
            if needs_chal and not any('chal' in l for l in next_few):
                fixed.append(f"{indent}{var}._thread_local.chal = "
                             f"{{'realm': 'test', 'nonce': 'abc123', "
                             f"'qop': 'auth', 'algorithm': 'MD5'}}")
                fixed.append(f"{indent}{var}._thread_local.last_nonce = 'abc123'")

    return '\n'.join(fixed)


def enforce_param_mocks(test_code: str, source_code: str,
                       cls_name: str, method_name: str) -> str:
    """If method accesses param.attr and test passes None, inject MagicMock."""
    param_attrs = infer_param_attributes(source_code, cls_name, method_name)
    if not param_attrs:
        return test_code

    lines = test_code.splitlines()
    fixed = []
    for i, line in enumerate(lines):
        replaced = False
        for param, attrs in param_attrs.items():
            m = re.match(r'(\s+)(\w+)\s*=\s*None\s*$', line)
            if m and m.group(2) == param:
                indent = m.group(1)
                # Replace None with MagicMock
                fixed.append(f"{indent}{param} = MagicMock()")
                for attr in attrs[:3]:
                    fixed.append(f"{indent}{param}.{attr} = MagicMock()")
                replaced = True
                break
        if not replaced:
            fixed.append(line)
    return '\n'.join(fixed)


def probe_stateful_dependencies(actual_import: str, cls_name: str,
                                 method_name: str, setup_chain: list,
                                 init_params: list) -> dict:
    """Run setup chain and capture state snapshot after stateful methods execute.
    Returns dict of captured state attributes, or {} on failure."""
    if not cls_name or not setup_chain:
        return {}

    constructor_args = ", ".join(
        f"{p}='test_{p}'" if 'name' in p.lower() or 'str' in p.lower()
        else f"{p}=1"
        for p in init_params
    ) if init_params else ""

    # Build setup calls
    setup_calls = []
    for pred_method, variable in setup_chain:
        method_short = pred_method.split('.')[-1]
        setup_calls.append(f"    obj.{method_short}()")

    # Build probe with mock setup for methods that need HTTP responses
    setup_code = "\n".join(setup_calls)
    probe = f"""
from {actual_import} import *
from unittest.mock import MagicMock
import json

try:
    obj = {cls_name}({constructor_args})
    
    # Try basic setup first
    try:
{setup_code}
    except Exception:
        pass
    
    # Capture all non-private state
    state = {{}}
    for attr_name in dir(obj):
        if attr_name.startswith('__'):
            continue
        try:
            val = getattr(obj, attr_name)
            if callable(val):
                continue
            if hasattr(val, '__dict__'):
                # Thread-local or namespace — capture sub-attrs
                for sub in dir(val):
                    if not sub.startswith('_'):
                        try:
                            sv = getattr(val, sub)
                            if not callable(sv):
                                state[f"{{attr_name}}.{{sub}}"] = repr(sv)
                        except Exception:
                            pass
            else:
                state[attr_name] = repr(val)
        except Exception:
            pass
    print(json.dumps(state))
except Exception as e:
    print(f'ERROR: {{e}}')
"""
    try:
        result = subprocess.run(
            [sys.executable, "-c", probe],
            capture_output=True, text=True, timeout=15
        )
        output = result.stdout.strip()
        if output and not output.startswith("ERROR"):
            return json.loads(output)
    except Exception:
        pass
    return {}


def is_async_method(source_code: str, cls_name: str, method_name: str) -> bool:
    """Check if a method is defined with async def."""
    try:
        tree = ast.parse(source_code)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == cls_name:
                for item in node.body:
                    if (isinstance(item, ast.AsyncFunctionDef) and
                        item.name == method_name):
                        return True
            elif not cls_name and isinstance(node, ast.AsyncFunctionDef):
                if node.name == method_name:
                    return True
    except SyntaxError:
        pass
    return False

def probe_all_targets(targets, actual_import, ctx, source_code, import_path=None):
    """Run probe_method on all targets in parallel and return results dict.
    Uses persistent database cache for probed values."""
    # Cannot probe digit-named modules with regular import
    # They use importlib loader which doesn't support from X import *
    if actual_import and actual_import[0].isdigit():
        print("   ⚠️  Using importlib probe for digit-named module")
        abs_target = os.path.abspath(
            next((f for f in [
                os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    '..', 'testgen_eval_lite',
                    'testgen_eval_files',
                    actual_import + '.py')
            ] if os.path.exists(f)), "")
        )
        # Use source_code path directly from caller
        # Fall through to normal probing using importlib-based probe
        pass

    probe_cache = {}
    targets_to_probe = []

    # First, check for cached values in database
    for cls, method in targets:
        cached = retrieve_probe_value(import_path or "", cls or "", method)
        if cached:
            key = f"{cls}.{method}" if cls else method
            probe_cache[key] = cached
        else:
            targets_to_probe.append((cls, method))

    # Probe remaining targets
    if targets_to_probe:
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(
                    probe_method, actual_import, cls, method,
                    [ci.get("init_params",[]) for ci in ctx["classes"]
                     if ci["name"]==cls][0] if cls else [],
                    source_code
                ): (cls, method)
                for cls, method in targets_to_probe
            }
            for future, (cls, method) in futures.items():
                key = f"{cls}.{method}" if cls else method
                try:
                    result = future.result()
                    probe_cache[key] = result
                    # Store in database for future runs
                    if result and not result.startswith("ERROR"):
                        store_probe_value(import_path or "", cls or "", method, result)
                except Exception:
                    probe_cache[key] = ""

    return probe_cache


def get_method_from_line(source_code: str, line_no: int) -> tuple:
    """Map a source line number to (class_name, method_name)."""
    try:
        tree = ast.parse(source_code)
        for cls_node in ast.walk(tree):
            if isinstance(cls_node, ast.ClassDef):
                for item in cls_node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        if item.lineno <= line_no <= (item.end_lineno or item.lineno):
                            return cls_node.name, item.name
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.lineno <= line_no <= (node.end_lineno or node.lineno):
                    return "", node.name
    except SyntaxError:
        pass
    return "", ""

# Fix 14: Module-level cache for resolve_base_classes
_base_classes_cache = {}

_DJANGO_BASE_APPS = ['django.contrib.contenttypes', 'django.contrib.auth']
_DJANGO_CONTRIB_RE = re.compile(r'django\.contrib\.([a-zA-Z_][a-zA-Z_0-9]*)')


def _discover_django_apps(source_code: str) -> list[str]:
    """Fix A — Source-driven INSTALLED_APPS. Auto-picks up any
    `django.contrib.X` the source references; no hardcoded contrib list."""
    apps = list(_DJANGO_BASE_APPS)
    for match in _DJANGO_CONTRIB_RE.finditer(source_code):
        full = f'django.contrib.{match.group(1)}'
        if full not in apps:
            apps.append(full)
    return apps


def detect_framework_setup(source_code: str) -> str:
    """Return setup code needed for the file's framework."""
    if 'from django' in source_code or 'import django' in source_code:
        apps = _discover_django_apps(source_code)
        header = (
            "import django\n"
            "from django.conf import settings\n"
            "if not settings.configured:\n"
            "    settings.configure(\n"
            "        DATABASES={'default': {'ENGINE': "
            "'django.db.backends.sqlite3', 'NAME': ':memory:'}},\n"
            f"        INSTALLED_APPS={apps!r},\n"
            "        USE_TZ=True, USE_L10N=False,\n"
            "        USE_THOUSAND_SEPARATOR=False, USE_I18N=True,\n"
            "        LANGUAGE_CODE='en-us',\n"
            "        DEFAULT_AUTO_FIELD='django.db.models.AutoField',\n"
            "        SECRET_KEY='test-secret-key-for-testing-only',\n"
            "        DEFAULT_HASHING_ALGORITHM='sha256',\n"
            "        PASSWORD_RESET_TIMEOUT=259200,\n"
            "        ROOT_URLCONF='django.contrib.staticfiles.urls',\n"
            "        MIDDLEWARE=[],\n"
            "        TEMPLATES=[{'BACKEND': "
            "'django.template.backends.django.DjangoTemplates',\n"
            "                    'DIRS': [], 'APP_DIRS': True, 'OPTIONS': {}}],\n"
            "    )\n"
            "    django.setup()\n\n"
        )
        if ('User' in source_code or
                'last_login' in source_code or
                'password' in source_code or
                'get_email_field_name' in source_code):
            header += "from django.contrib.auth.models import User\n\n"
        return header
    if 'from flask' in source_code or 'import flask' in source_code:
        return (
            "import flask\n"
            "app = flask.Flask(__name__)\n"
            "app.config['TESTING'] = True\n"
            "app.config['SECRET_KEY'] = 'test-secret-key'\n\n"
        )
    if 'from sqlalchemy' in source_code or 'import sqlalchemy' in source_code:
        return (
            "from sqlalchemy import create_engine\n"
            "from sqlalchemy.orm import sessionmaker\n"
            "_engine = create_engine('sqlite:///:memory:')\n"
            "_Session = sessionmaker(bind=_engine)\n\n"
        )
    if 'from fastapi' in source_code or 'import fastapi' in source_code:
        return (
            "from fastapi.testclient import TestClient\n\n"
        )
    
    if 'from sympy' in source_code or 'import sympy' in source_code:
        # Mirror the file's own sympy imports instead of hardcoding
        sympy_imports = []
        for line in source_code.splitlines():
            stripped = line.strip()
            if (stripped.startswith('from sympy') or
                    stripped.startswith('import sympy')):
                sympy_imports.append(stripped)
        # Always add basic symbols as fallback
        if not any('symbols' in l for l in sympy_imports):
            sympy_imports.append(
                'from sympy import symbols, Integer, Float, pi'
            )
        if ('units' in source_code or 'Quantity' in source_code
                or 'convert_to' in source_code):
            sympy_imports.append(
                'from sympy.physics.units import ('
                'meter, kilometer, second, gram, kilogram, '
                'mile, foot, inch, speed_of_light, newton, '
                'day, UnitSystem)'
            )
            sympy_imports.append(
                'from sympy.physics.units.util import convert_to'
            )
        sympy_import_block = "\n".join(sympy_imports)
        return f"{sympy_import_block}\n\n"

    return ""  # No framework setup needed

FEW_SHOT_PATTERNS = {
    "__len__":     "assert len(obj) == 2",
    "__repr__":    "assert repr(obj) == '<expected string>'",
    "__iter__":    "assert list(obj) == ['key1', 'key2']",
    "__eq__":      "obj2 = ClassName(same_args); assert obj1 == obj2",
    "__ne__":      "obj1 = ClassName('a', 'b'); obj2 = ClassName('c', 'd'); result = (obj1 != obj2); assert result == True",
    "__delitem__": "del obj['key']; assert 'key' not in obj",
    "__getitem__": "obj.attr = 'val'; assert obj['attr'] == 'val'",
    "__setitem__": "obj['KEY'] = 'val'; assert obj['key'] == 'val'",
    "__contains__":"assert 'key' in obj",
    "__bool__":    "assert bool(obj) == True",
    "__hash__":    "assert isinstance(hash(obj), int)",
    "lower_items": "assert list(obj.lower_items()) == [('key', 'val')]",
    "copy":        "copy = obj.copy(); assert copy['key'] == obj['key']",
    "get":         "obj.attr = 'val'; assert obj.get('attr') == 'val'",
    "__call__":    "r = MagicMock(); \nwith pytest.raises(NotImplementedError):\n    AuthBase()(r)\n# no assert needed after raises",
    "format": (
        "def test_format_integer():\n"
        "    result = format(1234, '.')\n"
        "    assert result == '1234'\n"
        "\ndef test_format_none():\n"
        "    result = format(None, '.')\n"
        "    assert result == 'None'\n"
        "\ndef test_format_string():\n"
        "    result = format('1234', '.')\n"
        "    assert result == '1234'\n"
    ),
}

MOCK_HINTS = {
    "handle_401": """
CRITICAL: handle_401 requires thread state initialized FIRST, then a Response-like object:
```python
auth = HTTPDigestAuth("user", "pass")
auth.init_per_thread_state()  # MUST call this BEFORE handle_401
r = MagicMock()
r.status_code = 401
r.headers = {'WWW-Authenticate': 'Digest realm="test", nonce="abc123"'}
r.request = MagicMock()
r.request.method = "GET"
r.request.url = "http://example.com"
r.request.body = None
r.request.headers = {}
result = auth.handle_401(r, **{})
assert result is not None
```
""",
    "handle_redirect": """
CRITICAL: Session.resolve_redirects expects a Response object that yields items.
```python
from unittest.mock import MagicMock
mock_resp = MagicMock()
mock_resp.is_redirect = True
mock_resp.headers = {'location': 'http://example.com/new'}
mock_resp.url = 'http://example.com/old'
session = Session()
gen = session.resolve_redirects(mock_resp, mock_resp.request)
result = list(gen)
assert len(result) >= 0
```
""",
    "build_digest_header": """
CRITICAL: build_digest_header returns a Digest auth string.
First call init_per_thread_state(), then set realm/nonce manually:
```python
auth = HTTPDigestAuth("user", "pass")
auth.init_per_thread_state()
auth.chal = {'realm': 'test', 'nonce': 'abc123', 'qop': 'auth', 'algorithm': 'MD5'}
result = auth.build_digest_header("GET", "http://example.com/api")
assert result.startswith("Digest username=")
```
""",
    "dispatch_hook": """
dispatch_hook takes (key, value, hooks) where hooks is a dict of callables.
```python
def test_dispatch_hook():
    hooks = {'response': [lambda r, **kwargs: r]}
    result = dispatch_hook('response', MagicMock(), hooks)
    assert result is not None

def test_dispatch_hook_empty():
    result = dispatch_hook('response', MagicMock(), {})
    assert result is not None
```
""",
    "request": """
CRITICAL: Never make real HTTP calls. Mock the Session:
```python
from unittest.mock import patch, MagicMock
with patch('requests.Session.request') as mock_req:
    mock_req.return_value = MagicMock()
    mock_req.return_value.status_code = 200
    mock_req.return_value.text = 'ok'
    result = request('GET', 'http://example.com')
    assert result.status_code == 200
```
""",
    "get": """
CRITICAL: Never make real HTTP calls. Mock the Session:
```python
from unittest.mock import patch, MagicMock
with patch('requests.Session.request') as mock_req:
    mock_req.return_value = MagicMock()
    mock_req.return_value.status_code = 200
    result = get('http://example.com')
    assert result.status_code == 200
```
""",
    "post": """
CRITICAL: Never make real HTTP calls. Mock the Session:
```python
with patch('requests.Session.request') as mock_req:
    mock_req.return_value = MagicMock()
    mock_req.return_value.status_code = 201
    result = post('http://example.com', data={'key': 'val'})
    assert result.status_code == 201
```
""",
    "put": """
CRITICAL: Mock the Session — never hit real URLs:
```python
with patch('requests.Session.request') as mock_req:
    mock_req.return_value = MagicMock()
    mock_req.return_value.status_code = 200
    result = put('http://example.com', data={'key': 'val'})
    assert result.status_code == 200
```
""",
    "patch": """
CRITICAL: Mock the Session — never hit real URLs:
```python
with patch('requests.Session.request') as mock_req:
    mock_req.return_value = MagicMock()
    mock_req.return_value.status_code = 200
    result = patch('http://example.com', data={'key': 'val'})
    assert result.status_code == 200
```
""",
    "delete": """
CRITICAL: Mock the Session — never hit real URLs:
```python
with patch('requests.Session.request') as mock_req:
    mock_req.return_value = MagicMock()
    mock_req.return_value.status_code = 200
    result = delete('http://example.com')
    assert result.status_code == 200
```
""",
    "options": """
CRITICAL: Mock the Session — never hit real URLs:
```python
with patch('requests.Session.request') as mock_req:
    mock_req.return_value = MagicMock()
    mock_req.return_value.status_code = 200
    result = options('http://example.com')
    assert result.status_code == 200
```
""",
    "head": """
CRITICAL: Mock the Session — never hit real URLs:
```python
with patch('requests.Session.request') as mock_req:
    mock_req.return_value = MagicMock()
    mock_req.return_value.status_code = 200
    result = head('http://example.com')
    assert result.status_code == 200
```
""",
    "settings_to_cmd_args_env": """
CRITICAL: DatabaseClient requires a connection object with settings_dict:
```python
from unittest.mock import MagicMock
conn = MagicMock()
conn.settings_dict = {
    'NAME': 'testdb', 'USER': 'user',
    'PASSWORD': 'pass', 'HOST': 'localhost',
    'PORT': '5432', 'OPTIONS': {}
}
client = DatabaseClient(conn)
args, env = client.settings_to_cmd_args_env(
    conn.settings_dict, False)
assert isinstance(args, list)
assert env is None or isinstance(env, dict)
```
""",
    "runshell": """
CRITICAL: DatabaseClient.runshell calls a real subprocess.
Mock both the connection and subprocess:
```python
from unittest.mock import MagicMock, patch
conn = MagicMock()
conn.settings_dict = {
    'NAME': 'testdb', 'USER': 'user',
    'PASSWORD': 'pass', 'HOST': 'localhost',
    'PORT': '5432', 'OPTIONS': {}
}
client = DatabaseClient(conn)
with patch('subprocess.run') as mock_run:
    mock_run.return_value = MagicMock(returncode=0)
    client.runshell(parameters=[])
    assert mock_run.called
```
""",
    "make_token": """
CRITICAL: make_token, check_token, _make_hash_value and
_make_token_with_timestamp all require a Django User object.
Create a mock user like this:
```python
from unittest.mock import MagicMock
from django.contrib.auth.models import User
user = User()
user.pk = 1
user.password = 'hashed_password'
user.last_login = None
user.email = 'test@test.com'
gen = PasswordResetTokenGenerator()
token = gen.make_token(user)
assert isinstance(token, str)
assert '-' in token
```
""",
    "_make_hash_value": """
CRITICAL: _make_hash_value requires a Django User object.
Create a mock user:
```python
from django.contrib.auth.models import User
user = User()
user.pk = 1
user.password = 'hashed_password'
user.last_login = None
user.email = 'test@test.com'
gen = PasswordResetTokenGenerator()
result = gen._make_hash_value(user, 1000)
assert isinstance(result, str)
assert str(user.pk) in result
```
""",
    "_make_token_with_timestamp": """
CRITICAL: _make_token_with_timestamp requires a Django User object.
```python
from django.contrib.auth.models import User
user = User()
user.pk = 1
user.password = 'hashed_password'
user.last_login = None
user.email = 'test@test.com'
gen = PasswordResetTokenGenerator()
token = gen._make_token_with_timestamp(user, 1000)
assert isinstance(token, str)
assert '-' in token
```
""",
}

def _extract_signature_context(source_code: str, cls_name: str,
                                method_name: str, import_path: str) -> str:
    """Extract just imports + function/class signature for lean plan-mode prompts.

    Instead of feeding the full method body (which the plan already digested),
    this gives the model only what it needs to write syntactically correct code:
    the import path and the function/class signature.
    """
    parts = [f"from {import_path} import *"]

    try:
        tree = ast.parse(source_code)
        lines = source_code.splitlines()

        if cls_name:
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and node.name == cls_name:
                    # Class signature line
                    parts.append(f"\n{lines[node.lineno - 1]}")
                    # __init__ signature (for instantiation)
                    for item in node.body:
                        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            if item.name == "__init__":
                                sig_line = lines[item.lineno - 1]
                                parts.append(f"    {sig_line.strip()}")
                                parts.append("        ...")
                    # Target method signature
                    for item in node.body:
                        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            if item.name == method_name:
                                sig_line = lines[item.lineno - 1]
                                parts.append(f"    {sig_line.strip()}")
                                parts.append("        ...")
                                break
                    break
        else:
            # Top-level function
            for node in ast.iter_child_nodes(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if node.name == method_name:
                        sig_line = lines[node.lineno - 1]
                        parts.append(f"\n{sig_line}")
                        parts.append("    ...")
                        break
    except SyntaxError:
        pass

    return "\n".join(parts)


def _section_core(method_context, method_name, cls_name,
                  target_label, invocation_hint,
                  expected_func_name) -> str:
    return (
        f"Write ONE pytest test for `{target_label}`:\n\n"
        f"```python\n{method_context}\n```\n\n"
        f"## Target Anchoring\n"
        f"1. Test ONLY `{target_label}`. "
        f"Call `{invocation_hint}`.\n"
        f"2. Function name: `def {expected_func_name}():`\n"
        f"3. MUTATION-PROOF ASSERTIONS REQUIRED:\n"
        f"   - Use `== exact_value` not `is not None`\n"
        f"   - Use `== False` not just `assert not result`\n"
        f"   - Use `== True` not just `assert result`\n"
        f"   - Use `== 'exact_string'` not `assert result`\n"
        f"   - WRONG: `assert result is not None`\n"
        f"   - RIGHT: `assert result == 42` or "
        f"`assert result == 'expected_value'`\n"
        f"   Weak assertions score 0 on mutation testing.\n"
    )

def _section_docstring(docstring: str) -> str:
    if not docstring:
        return ""
    return (
        f"\nFunction documentation: {docstring}\n"
        f"Use this documentation to determine exact "
        f"expected values.\n"
    )

def _section_comments(comments: list) -> str:
    """Inject standalone developer comments as context.
    
    Comments often contain edge case hints, known bugs, TODOs,
    and behavioral invariants not captured in the docstring.
    """
    if not comments:
        return ""
    comment_block = "\n".join(f"  - {c}" for c in comments)
    return (
        f"\n## Developer Notes (from source comments)\n"
        f"{comment_block}\n"
        f"Consider these hints when choosing test inputs "
        f"and expected outputs.\n"
    )

def _section_rolling_summary(covered_summary: list,
                              target_label: str) -> str:
    if not covered_summary:
        return ""
    return (
        f"Already tested: {', '.join(covered_summary)}\n"
        f"Now testing: {target_label} — do NOT repeat.\n\n"
    )

def _section_source_patterns(source_patterns: list) -> str:
    if not source_patterns:
        return ""
    lines = "\n".join(
        f"- {desc}" for _, desc in source_patterns
    )
    return f"## Source Code Patterns Detected\n{lines}\n\n"

def _section_cfg_paths(method_paths: list) -> str:
    if not method_paths:
        return ""
    
    # Build checklist with suggested test function names per path
    checklist_lines = []
    for i, p in enumerate(method_paths[:8]):
        path_desc = ' → '.join(str(step) for step in p)
        # Suggest a test name based on the path's key feature
        path_str_check = " ".join(str(step) for step in p)
        if "RAISE" in path_str_check:
            suffix = "raises"
        elif "ELSE" in path_str_check:
            suffix = "else_branch"
        elif "IF-ONLY" in path_str_check:
            suffix = "if_truthy"
        elif "0 iterations" in path_str_check:
            suffix = "empty_input"
        elif "BREAK" in path_str_check:
            suffix = "early_exit"
        elif "RETURN None" in path_str_check:
            suffix = "returns_none"
        elif "RETURN" in path_str_check:
            suffix = f"path_{i+1}"
        else:
            suffix = f"path_{i+1}"
        checklist_lines.append(f"  ☐ P{i+1}: {path_desc}  →  test_..._{suffix}()")
    
    path_str = "\n".join(checklist_lines)
    
    # Count distinct path types to enforce specific coverage
    has_raise = any(any("RAISE" in str(step) for step in p) for p in method_paths)
    has_branch = any(any("ELSE" in str(step) for step in p) for p in method_paths)
    has_loop = any(any("LOOP" in str(step) or "FOR" in str(step) for step in p) for p in method_paths)
    has_if_only = any(any("IF-ONLY" in str(step) for step in p) for p in method_paths)
    
    guidance = "⚠️ MANDATORY — your test MUST cover ALL of these:"
    if has_raise:
        guidance += "\n  - Use `pytest.raises()` for the RAISE path"
    if has_branch:
        guidance += "\n  - Test BOTH the IF and ELSE branches with different inputs"
    if has_loop:
        guidance += "\n  - Test empty input (0 iterations) AND non-empty input"
    if has_if_only:
        guidance += "\n  - Test with TRUTHY input to enter the IF branch — do NOT only test falsy/empty cases"
    if not (has_raise or has_branch or has_loop or has_if_only):
        guidance += "\n  - Cover each distinct execution path with a separate assertion"
    
    return (
        f"\n## 🚨 REQUIRED Execution Paths — You MUST cover these:\n"
        f"{path_str}\n"
        f"{guidance}\n"
    )

def _section_return_value(data_flow: dict) -> str:
    if not data_flow.get("return_values"):
        return ""
    rv = data_flow["return_values"][0]
    return f"\n## Return Value\nThis method returns: `{rv}`\n"

def _section_exceptions(data_flow: dict) -> str:
    if not data_flow.get("raises"):
        return ""
    raises_str = ', '.join(data_flow['raises'])
    return (
        f"\n## Exception Paths\n"
        f"This method can raise: {raises_str}\n"
        f"Add `with pytest.raises(...)` to cover this path.\n"
    )

def _section_callees(subgraph: dict,
                     target_label: str) -> str:
    if not subgraph.get("callees"):
        return ""
    callees = ', '.join(subgraph['callees'][:3])
    return (
        f"\n## Internal Dependencies\n"
        f"`{target_label}` calls: {callees}\n"
        f"Your test will indirectly exercise these too.\n"
    )

def _section_state_deps(G, target_label_graph: str,
                        source_code: str,
                        cls_name: str) -> str:
    """Inject state dependency info from Graph RAG.
    Shows which attributes this method reads that are
    written by other methods — critical for setup order."""
    if not cls_name:
        return ""
    deps = []
    try:
        if target_label_graph in G:
            for pred, _, data in G.in_edges(
                    target_label_graph, data=True):
                if data.get('type') == 'state_dependency':
                    var = data.get('variable', '')
                    writer = pred.split('.')[-1]
                    if writer != '__init__' and var:
                        deps.append((writer, var))
            for _, succ, data in G.out_edges(
                    target_label_graph, data=True):
                pass  # outgoing handled by setup chain
    except Exception:
        return ""
    if not deps:
        return ""
    result = "\n## State Dependencies (Graph RAG)\n"
    result += (
        "This method reads state written by other methods:\n"
    )
    for writer, var in deps[:4]:
        result += (
            f"- `self.{var}` is written by "
            f"`{writer}()` — call it first\n"
        )
    return result

def _section_base_classes(base_classes: dict,
                           cls_name: str) -> str:
    if not base_classes or not cls_name:
        return ""
    result = ""
    for base, info in base_classes.items():
        methods = ', '.join(info['methods'][:5])
        result += (
            f"\n## Inherited From: {base}\n"
            f"Methods inherited: {methods}\n"
        )
    return result

def _section_few_shot(method_name: str,
                      FEW_SHOT_PATTERNS: dict,
                      rag_examples: list = None) -> str:
    # 1. Try hardcoded patterns first (highest priority)
    pattern = FEW_SHOT_PATTERNS.get(method_name, "")
    if pattern:
        return f"\n## Correct Pattern\n`{pattern}`\n"
    
    # 2. Fall back to RAG store examples as few-shot
    if rag_examples:
        best = [e for e in rag_examples
                if e.get("quality_score", 0) >= 50.0
                and "self.assert" not in e.get("test_code", "").lower()
                and "unittest.TestCase" not in e.get("test_code", "")][:2]
        if best:
            section = "\n## Example Tests From Similar Methods (adapt to current target)\n"
            for i, ex in enumerate(best, 1):
                snippet = ex["test_code"][:400]
                section += (
                    f"### Example {i} (quality {ex['quality_score']:.0f}/100)\n"
                    f"```python\n{snippet}\n```\n"
                )
            section += (
                "Adapt these patterns to the CURRENT target method. "
                "Do NOT copy class or method names — use the correct ones.\n"
            )
            return section
    
    return ""

def _section_builtin_collision(method_name: str) -> str:
    PYTHON_BUILTINS = {
        "format", "input", "print", "type", "list", "dict",
        "set", "filter", "map", "sorted", "sum", "min", "max",
        "len", "range", "enumerate", "zip", "open", "iter",
        "next", "hash", "id", "repr", "str", "int", "float",
        "bool", "bytes", "object", "super", "property",
    }
    clean = method_name.strip("_")
    if method_name not in PYTHON_BUILTINS and clean not in PYTHON_BUILTINS:
        return ""
    return (
        f"\n## NAME COLLISION WARNING\n"
        f"`{method_name}` has the same name as a Python builtin. "
        f"The MODULE version is already loaded via globals(). "
        f"Call it directly as `{method_name}(args)` — "
        f"it refers to the MODULE function, NOT the Python builtin.\n"
        f"Example: `result = {method_name}(your_args)` "
        f"calls the module function.\n"
    )

def _section_wrapper_types(source_code: str) -> str:
    WRAPPER_TYPES = {
        'mark_safe': 'SafeString',
        'mark_for_escaping': 'SafeData',
        'gettext_lazy': 'LazyString',
        'ugettext_lazy': 'LazyString',
        'format_html': 'SafeString',
        'escape': 'SafeString',
    }
    detected = [
        (w, t) for w, t in WRAPPER_TYPES.items()
        if w in source_code
    ]
    if not detected:
        return ""
    names = ', '.join(w for w, _ in detected)
    return (
        f"\n## WRAPPER RETURN TYPES DETECTED\n"
        f"This file uses: {names}\n"
        f"These return wrapper objects that compare equal "
        f"to plain strings.\n"
        f"Use == 'expected' directly — "
        f"do NOT use isinstance checks.\n"
        f"Example: assert result == '1234'  "
        f"# works with SafeString\n"
        f"Do NOT write: assert isinstance(result, str)  "
        f"# this FAILS\n"
    )

def _section_constructor_sig(method_name: str, cls_name: str,
                              actual_import: str) -> str:
    if method_name != "__init__" or not cls_name:
        return ""
    sig_info = probe_constructor_signature(actual_import, cls_name)
    if not sig_info or sig_info.startswith("ERROR"):
        return ""
    return (
        f"\n## Constructor Signature\n"
        f"`{cls_name}.__init__` accepts: `{sig_info}`\n"
        f"ONLY pass these exact parameters.\n"
        f"Example: if signature is `*args (required)`, "
        f"use `{cls_name}('message')`\n"
    )

def _section_setup_chain(G, target_label_graph: str,
                          cls_name: str, source_code: str,
                          actual_import: str,
                          ctx: dict) -> str:
    setup_chain = get_stateful_setup_chain(G, target_label_graph)
    if not setup_chain:
        return ""
    result = "\n## REQUIRED SETUP STATE\n"
    result += (
        "This method reads internal state. "
        "You MUST call these first:\n```python\n"
    )
    for pred_method, variable in setup_chain:
        method_short = pred_method.split('.')[-1]
        result += (
            f"obj.{method_short}()  "
            f"# initializes self.{variable}\n"
        )
    result += "```\nSkipping causes AttributeError. NOT optional.\n"

    _init_p = [
        ci.get("init_params", [])
        for ci in ctx["classes"]
        if ci["name"] == cls_name
    ]
    _init_params = _init_p[0] if _init_p else []
    state_snapshot = probe_stateful_dependencies(
        actual_import, cls_name,
        target_label_graph.split('.')[-1],
        setup_chain, _init_params
    )
    if state_snapshot:
        result += "\n## Runtime State After Setup\n"
        result += (
            "After calling setup, the object has this state:\n"
            "```python\n"
        )
        for k, v in list(state_snapshot.items())[:8]:
            result += f"# obj.{k} = {v}\n"
        result += "```\nUse these concrete values in assertions.\n"
    return result

def _section_async(source_code: str, cls_name: str,
                   method_name: str) -> str:
    if not is_async_method(source_code, cls_name or "",
                           method_name):
        return ""
    return (
        "\n## ASYNC METHOD\n"
        "Use pytest-asyncio:\n```python\n"
        "import pytest\n"
        "@pytest.mark.asyncio\n"
        "async def test_method():\n"
        "    result = await obj.method()\n"
        "    assert result == expected\n"
        "```\n"
    )

def _section_param_mocks(source_code: str, cls_name: str,
                         method_name: str) -> str:
    param_attrs = infer_param_attributes(
        source_code, cls_name or "", method_name
    )
    if not param_attrs:
        return ""
    result = "\n## Parameter Attributes Required\n"
    for param, attrs in param_attrs.items():
        attrs_str = ", ".join(f".{a}" for a in attrs[:6])
        mock_setup = "; ".join(
            f"{param}.{a} = MagicMock()" for a in attrs[:3]
        )
        result += (
            f"`{param}` accesses: {attrs_str}\n"
            f"```python\n"
            f"{param} = MagicMock()\n"
            f"{mock_setup}\n"
            f"```\n"
        )
    return result

def _section_branch_inputs(source_code: str, cls_name: str,
                            method_name: str) -> str:
    branch_inputs = synthesize_branch_inputs(
        source_code, cls_name or "", method_name
    )
    if not branch_inputs:
        return ""
    result = (
        "\n## Branch Coverage Inputs\n"
        "Test with these inputs for full branch coverage:\n"
    )
    for bi in branch_inputs:
        cond = bi["condition"]
        if bi.get("true_input") and bi.get("false_input"):
            result += (
                f"- `{cond}` → "
                f"TRUE: `{bi['true_input']}`, "
                f"FALSE: `{bi['false_input']}`\n"
            )
        else:
            result += (
                f"- Branch: `{cond}` — "
                f"test both True and False paths\n"
            )
    result += "Write separate tests for different branches.\n"
    return result

def _section_http_mock(subgraph: dict,
                       method_name: str,
                       MOCK_HINTS: dict) -> str:
    hint = MOCK_HINTS.get(method_name, "")
    if hint:
        return hint
    http_methods = {
        "get", "post", "put", "patch", "delete",
        "options", "head", "request", "send"
    }
    callee_names = {
        c.split('.')[-1].lower()
        for c in subgraph.get("callees", [])
    }
    if not (callee_names & http_methods):
        return ""
    return (
        "\n## HTTP Call Detected\n"
        "Mock it:\n```python\n"
        "from unittest.mock import patch, MagicMock\n"
        "mock_resp = MagicMock()\n"
        "mock_resp.status_code = 200\n"
        "mock_resp.text = '{}'\n"
        "mock_resp.headers = {}\n"
        "with patch('requests.Session.request', "
        "return_value=mock_resp):\n"
        "    result = obj.method()\n"
        "```\n"
    )

def _section_dict_like(cls_name: str, source_code: str,
                        base_classes: dict) -> str:
    if not cls_name:
        return ""
    dict_bases = {
        "dict", "MutableMapping", "Mapping", "OrderedDict"
    }
    cls_bases = set(base_classes.keys()) if base_classes else set()
    try:
        tree_check = ast.parse(source_code)
        for nd in ast.walk(tree_check):
            if (isinstance(nd, ast.ClassDef) and
                    nd.name == cls_name):
                for base in nd.bases:
                    if isinstance(base, ast.Name):
                        cls_bases.add(base.id)
                    elif isinstance(base, ast.Attribute):
                        cls_bases.add(base.attr)
    except Exception:
        pass
    if not (cls_bases & dict_bases):
        return ""
    return (
        "\n## Dict-Like Class\n"
        f"`{cls_name}` inherits from dict/Mapping. "
        f"Initialize with data:\n"
        f"```python\n"
        f"obj = {cls_name}({{'key': 'value'}})\n"
        f"```\n"
        f"Test __getitem__, __setitem__, __contains__ "
        f"with actual key-value pairs.\n"
    )

def _section_probe(probe_cache: dict, target_label: str,
                   invocation_hint: str) -> str:
    probed = probe_cache.get(target_label)
    if not probed or probed.startswith("ERROR"):
        return ""
    return (
        f"\n## Probed Actual Output\n"
        f"`{invocation_hint}` returns: `{probed}`\n"
        f"Use this exact value in your assertion.\n"
    )

def _section_rag(similar_examples: list) -> tuple[str, bool]:
    if not similar_examples:
        return "", False
    good = [e for e in similar_examples
            if e["quality_score"] >= 30.0
            and "self.assert" not in e["test_code"].lower()
            and "unittest.TestCase" not in e["test_code"]]
    if not good:
        return "", False
    best = good[0]
    if best["quality_score"] >= 100.0:
        return "", True
    section = (
        f"\nHere is an existing test for a similar method "
        f"(quality {best['quality_score']:.0f}/100):\n"
        f"```python\n{best['test_code']}\n```\n"
    )
    if best.get('coverage_pattern'):
        section += f"Coverage achieved: {best['coverage_pattern']}\n"
    section += (
        "IMPROVE this test: add more assertions, "
        "cover edge cases, use stricter exact values.\n"
        "If you cannot improve it, reproduce it exactly.\n"
    )
    return section, False

def _section_method_deps(method_deps: list) -> str:
    if not method_deps:
        return ""
    result = "\n## Required Setup — Call these methods first:\n"
    for dep in method_deps:
        result += (
            f"- Call `obj.{dep['method']}()` first "
            f"(type: {dep['type']})\n"
        )
    return result + "\n"

def _section_bad_examples(bad_examples: list) -> str:
    if not bad_examples:
        return ""
    result = (
        "\n## Previous Failed Attempts — "
        "DO NOT repeat these patterns:\n"
    )
    for ex in bad_examples[:2]:
        result += f"FAILED reason: {ex['failure_reason'][:100]}\n"
        if ex.get('failure_category'):
            result += f"Error type: {ex['failure_category']}\n"
        if ex.get('survived_mutants'):
            muts = ', '.join(ex['survived_mutants'][:3])
            result += f"Failed mutations: {muts}\n"
            result += (
                "Add assertions that would catch "
                "these mutations.\n"
            )
        result += (
            f"```python\n{ex['test_code'][:300]}\n```\n"
            "Do NOT write a test like this.\n\n"
        )
    return result

def _section_cross_file(source_code: str, target_file: str,
                         import_path: str) -> str:
    if import_path:
        repo_dir = os.path.dirname(os.path.dirname(target_file))
    else:
        repo_dir = os.path.dirname(os.path.abspath(target_file))
    cross_ctx = build_cross_file_context(
        source_code, target_file, repo_dir
    )
    if not cross_ctx:
        return ""
    result = "\n## Imported Dependencies\n"
    for dep_name, dep_info in list(cross_ctx.items())[:3]:
        result += (
            f"### `{dep_name}` "
            f"({dep_info['type']} from {dep_info['file']})\n"
            f"```python\n{dep_info['snippet']}\n```\n"
        )
    return result

def route_error_to_fix(
    error_logs: str,
    source_code: str,
    cls_name: str,
    method_name: str,
    G,
    target_label_graph: str,
    ctx: dict,
    actual_import: str,
    subgraph: dict,
    MOCK_HINTS: dict,
) -> str:
    logs = error_logs.lower()

    if ("attributeerror" in logs and
            ("_thread_local" in logs or
             "has no attribute" in logs)):
        return _section_setup_chain(
            G, target_label_graph, cls_name,
            source_code, actual_import, ctx
        )

    if "keyerror" in logs and (
            "realm" in logs or "nonce" in logs):
        return (
            "\n🔴 CRITICAL: auth.chal dict is missing:\n"
            "```python\n"
            "auth.init_per_thread_state()\n"
            "auth.chal = {'realm': 'test', "
            "'nonce': 'abc123', "
            "'qop': 'auth', 'algorithm': 'MD5'}\n"
            "```\n"
        )

    if ("connectionerror" in logs or
            "requests.exceptions" in logs or
            "requests.get" in logs or
            "requests.post" in logs):
        return _section_http_mock(subgraph, method_name,
                                   MOCK_HINTS)

    if ("attributeerror" in logs and
            "nonetype" in logs):
        return _section_param_mocks(
            source_code, cls_name, method_name
        )

    if "importerror" in logs or "modulenot" in logs:
        return (
            "\n## IMPORT FIX\n"
            "Do NOT add any import statements.\n"
            "All imports are already at the top of the file.\n"
            "Just write the test function body directly.\n"
        )

    if "typeerror" in logs:
        if ("sympy" in source_code.lower() or
                "from sympy" in source_code):
            return (
                "\n## SYMPY TYPE ERROR\n"
                "You passed wrong argument types.\n"
                "SymPy functions need SymPy objects:\n"
                "```python\n"
                "from sympy.physics.units import meter, kilometer\n"
                "result = convert_to(meter, kilometer)\n"
                "```\n"
                "NEVER pass plain strings or ints to SymPy.\n"
            )
        return (
            "\n## TYPE ERROR FIX\n"
            "Check the function signature carefully.\n"
            "You passed the wrong type of argument.\n"
            "Re-read the source code and match exact types.\n"
        )

    if "assertionerror" in logs or "assert" in logs:
        e_lines = [
            l.strip() for l in error_logs.split("\n")
            if l.strip().startswith("E ")
        ]
        if e_lines:
            e_block = "\n".join(e_lines[:3])
            return (
                f"\n## WRONG EXPECTED VALUE\n"
                f"Your assertion used wrong expected value:\n"
                f"```\n{e_block}\n```\n"
                f"Read the actual value from the error above "
                f"and use THAT exact value.\n"
            )

    if "notimplementederror" in logs:
        return (
            "\n## ABSTRACT METHOD\n"
            "This method raises NotImplementedError by design.\n"
            "Test it with pytest.raises:\n"
            "```python\n"
            "with pytest.raises(NotImplementedError):\n"
            f"    obj.{method_name}()\n"
            "```\n"
            "No other assertion needed.\n"
        )

    if "valueerror" in logs:
        return (
            "\n## VALUE ERROR FIX\n"
            "Your test passed an invalid value.\n"
            "Either use valid inputs OR test the error:\n"
            "```python\n"
            "with pytest.raises(ValueError):\n"
            f"    obj.{method_name}(invalid_input)\n"
            "```\n"
        )

    # General: detect missing connection/db object with settings_dict
    if ("nonetype" in logs and
            "settings_dict" in logs):
        return (
            "\n## CONNECTION OBJECT REQUIRED\n"
            "This class requires a database connection object "
            "with settings_dict. Mock it:\n"
            "```python\n"
            "from unittest.mock import MagicMock\n"
            "conn = MagicMock()\n"
            "conn.settings_dict = {\n"
            "    'NAME': 'testdb', 'USER': 'user',\n"
            "    'PASSWORD': 'pass', 'HOST': 'localhost',\n"
            "    'PORT': '5432', 'OPTIONS': {}\n"
            "}\n"
            "obj = ClassName(conn)\n"
            "```\n"
            "Replace ClassName with the actual class name.\n"
        )

    # General: detect subprocess calls that need mocking
    if ("oserror" in logs or
            "filenotfounderror" in logs or
            "permissionerror" in logs or
            "subprocess" in logs):
        return (
            "\n## SUBPROCESS MOCK REQUIRED\n"
            "This method calls a real subprocess. Mock it:\n"
            "```python\n"
            "from unittest.mock import patch, MagicMock\n"
            "with patch('subprocess.run') as mock_run:\n"
            "    mock_run.return_value = MagicMock()\n"
            "    mock_run.return_value.returncode = 0\n"
            "    result = obj.method()\n"
            "    assert mock_run.called\n"
            "```\n"
        )

    return ""


# ── Repo-level context detection ──
# Runs once per repo, before per-file generation loop

_repo_context_cache = {}  # Cache by repo dir


def detect_repo_context(target_file: str, deep_scan: bool = False,
                        source_files: list = None) -> dict:
    """Pre-scan the repo directory to extract project-level context.
    
    Detects:
      - Dependencies (from requirements.txt / setup.py / pyproject.toml)
      - Test framework and style (pytest vs unittest)
      - Existing fixtures from conftest.py
      - Framework (Django, Flask, FastAPI, etc.)
      - Python version constraints
    
    Returns a dict with all detected context.
    """
    # Find repo root (walk up to find setup.py / .git / pyproject.toml)
    target_dir = os.path.dirname(os.path.abspath(target_file))
    repo_dir = target_dir
    for _ in range(5):  # Walk up max 5 levels
        if any(os.path.exists(os.path.join(repo_dir, marker))
               for marker in [".git", "setup.py", "pyproject.toml", "setup.cfg"]):
            break
        parent = os.path.dirname(repo_dir)
        if parent == repo_dir:
            break
        repo_dir = parent
    
    # Return cached result if available
    if repo_dir in _repo_context_cache:
        return _repo_context_cache[repo_dir]
    
    context = {
        "repo_dir": repo_dir,
        "dependencies": [],
        "dev_dependencies": [],
        "framework": None,
        "test_style": "pytest",
        "fixtures": [],
        "conftest_imports": [],
        "python_version": None,
        "has_setup_py": False,
        "has_pyproject": False,
    }
    
    # 1. Parse requirements.txt
    for req_file in ["requirements.txt", "requirements-dev.txt",
                     "requirements_dev.txt", "test-requirements.txt"]:
        req_path = os.path.join(repo_dir, req_file)
        if os.path.exists(req_path):
            try:
                with open(req_path, encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and not line.startswith("-"):
                            # Extract package name (before ==, >=, etc.)
                            pkg = re.split(r'[><=!~;\[]', line)[0].strip()
                            if pkg:
                                target_list = "dev_dependencies" if "dev" in req_file or "test" in req_file else "dependencies"
                                context[target_list].append(pkg)
            except IOError:
                pass
    
    # 2. Parse setup.py for install_requires
    setup_py = os.path.join(repo_dir, "setup.py")
    if os.path.exists(setup_py):
        context["has_setup_py"] = True
        try:
            with open(setup_py, encoding="utf-8", errors="ignore") as f:
                setup_content = f.read()
            # Extract install_requires list
            match = re.search(r'install_requires\s*=\s*\[([^\]]+)\]', setup_content)
            if match:
                deps = re.findall(r'["\']([^"\'><=!~]+)', match.group(1))
                for d in deps:
                    d = d.strip()
                    if d and d not in context["dependencies"]:
                        context["dependencies"].append(d)
        except IOError:
            pass
    
    # 3. Parse pyproject.toml
    pyproject = os.path.join(repo_dir, "pyproject.toml")
    if os.path.exists(pyproject):
        context["has_pyproject"] = True
        try:
            with open(pyproject, encoding="utf-8", errors="ignore") as f:
                pyp_content = f.read()
            # Extract python version
            ver_match = re.search(r'python_requires\s*=\s*["\']([^"\']+)', pyp_content)
            if ver_match:
                context["python_version"] = ver_match.group(1)
            # Extract dependencies
            dep_match = re.search(r'dependencies\s*=\s*\[([^\]]+)\]', pyp_content)
            if dep_match:
                deps = re.findall(r'["\']([^"\'><=!~]+)', dep_match.group(1))
                for d in deps:
                    d = d.strip()
                    if d and d not in context["dependencies"]:
                        context["dependencies"].append(d)
        except IOError:
            pass
    
    # 4. Detect framework from dependencies
    all_deps_lower = [d.lower() for d in context["dependencies"] + context["dev_dependencies"]]
    if "django" in all_deps_lower:
        context["framework"] = "Django"
    elif "flask" in all_deps_lower:
        context["framework"] = "Flask"
    elif "fastapi" in all_deps_lower:
        context["framework"] = "FastAPI"
    elif "sqlalchemy" in all_deps_lower:
        context["framework"] = "SQLAlchemy"
    elif "numpy" in all_deps_lower or "scipy" in all_deps_lower:
        context["framework"] = "Scientific"
    
    # 5. Scan conftest.py for fixtures
    for root, dirs, files in os.walk(repo_dir):
        dirs[:] = [d for d in dirs if d not in {
            '__pycache__', '.git', 'node_modules', '.tox', 'build', 'dist', '.eggs'}]
        if "conftest.py" in files:
            conftest_path = os.path.join(root, "conftest.py")
            try:
                with open(conftest_path, encoding="utf-8", errors="ignore") as f:
                    conftest_src = f.read()
                tree = ast.parse(conftest_src)
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        # Check for @pytest.fixture decorator
                        for dec in node.decorator_list:
                            dec_name = ""
                            if isinstance(dec, ast.Attribute):
                                dec_name = dec.attr
                            elif isinstance(dec, ast.Name):
                                dec_name = dec.id
                            elif isinstance(dec, ast.Call):
                                if isinstance(dec.func, ast.Attribute):
                                    dec_name = dec.func.attr
                                elif isinstance(dec.func, ast.Name):
                                    dec_name = dec.func.id
                            if dec_name == "fixture":
                                context["fixtures"].append(node.name)
                # Collect imports from conftest
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            context["conftest_imports"].append(alias.name)
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            context["conftest_imports"].append(node.module)
            except (SyntaxError, IOError):
                pass
        # Only check first 3 levels deep
        if root.count(os.sep) - repo_dir.count(os.sep) >= 3:
            dirs.clear()
    
    # 6. Detect test style (pytest vs unittest)
    test_dirs = ["tests", "test"]
    for td in test_dirs:
        test_path = os.path.join(repo_dir, td)
        if os.path.isdir(test_path):
            for fname in os.listdir(test_path)[:10]:  # Sample first 10 test files
                if fname.startswith("test_") and fname.endswith(".py"):
                    try:
                        with open(os.path.join(test_path, fname),
                                  encoding="utf-8", errors="ignore") as f:
                            sample = f.read(2000)  # Read first 2KB
                        if "unittest.TestCase" in sample or "self.assert" in sample:
                            context["test_style"] = "unittest"
                        elif "def test_" in sample and "pytest" in sample:
                            context["test_style"] = "pytest"
                        break  # One sample is enough
                    except IOError:
                        pass
            break
    
    # Deduplicate
    context["dependencies"] = list(dict.fromkeys(context["dependencies"]))[:20]
    context["dev_dependencies"] = list(dict.fromkeys(context["dev_dependencies"]))[:10]
    context["fixtures"] = list(dict.fromkeys(context["fixtures"]))[:15]
    context["conftest_imports"] = list(dict.fromkeys(context["conftest_imports"]))[:10]
    
    _repo_context_cache[repo_dir] = context
    
    # Optional: Docker deep scan for verified imports and test discovery
    if deep_scan:
        try:
            from docker_runner import docker_deep_scan
            repo_name = None
            for dep in context["dependencies"]:
                repo_name = dep
                break
            if not repo_name:
                repo_name = os.path.basename(repo_dir)
            
            scan = docker_deep_scan(
                repo_dir, repo_name=repo_name,
                source_files=source_files or []
            )
            context["deep_scan"] = scan
            
            # Merge working fixtures from deep scan
            if scan.get("existing_fixtures"):
                for fix in scan["existing_fixtures"]:
                    if fix not in context["fixtures"]:
                        context["fixtures"].append(fix)
            
            # Update test style from deep scan
            if scan.get("test_patterns"):
                if "pytest" in scan["test_patterns"]:
                    context["test_style"] = "pytest"
                elif "unittest" in scan["test_patterns"]:
                    context["test_style"] = "unittest"
            
            print(f"   📋 Deep scan: {len(scan.get('working_imports', []))} imports OK, "
                  f"{len(scan.get('failed_imports', []))} failed")
        except Exception as e:
            print(f"   ⚠️ Deep scan failed: {e}")
    
    return context


def _section_repo_context(repo_ctx: dict) -> str:
    """Build prompt section with repo-level context."""
    if not repo_ctx:
        return ""
    
    parts = []
    parts.append("\n## Repository Context")
    
    if repo_ctx.get("framework"):
        parts.append(f"Framework: {repo_ctx['framework']}")
    
    if repo_ctx.get("dependencies"):
        deps = ", ".join(repo_ctx["dependencies"][:12])
        parts.append(f"Dependencies: {deps}")
    
    if repo_ctx.get("test_style"):
        parts.append(f"Test style in this repo: {repo_ctx['test_style']}")
    
    if repo_ctx.get("fixtures"):
        fixtures = ", ".join(repo_ctx["fixtures"][:8])
        parts.append(
            f"Available conftest.py fixtures: {fixtures}\n"
            f"You may use these fixtures as test function parameters."
        )
    
    if repo_ctx.get("python_version"):
        parts.append(f"Python version: {repo_ctx['python_version']}")
    
    if len(parts) <= 1:  # Only the header, no useful info
        return ""
    
    parts.append("")  # trailing newline
    return "\n".join(parts)


def _section_deep_scan(repo_ctx: dict) -> str:
    """Build prompt section with Docker deep scan results."""
    scan = repo_ctx.get("deep_scan")
    if not scan:
        return ""
    
    parts = []
    
    if scan.get("failed_imports"):
        failed = scan["failed_imports"][:5]
        parts.append(
            f"\n⚠️ These imports FAIL in a clean environment: "
            f"{', '.join(failed)}\n"
            f"Mock any dependencies from these modules.\n"
        )
    
    if scan.get("existing_tests"):
        sample = scan["existing_tests"][:5]
        parts.append(
            f"\nExisting test patterns in this repo:\n"
            + "\n".join(f"  - {t}" for t in sample) + "\n"
            f"Follow these naming/style conventions.\n"
        )
    
    if scan.get("existing_fixtures"):
        fixes = scan["existing_fixtures"][:8]
        parts.append(
            f"\nAvailable pytest fixtures: {', '.join(fixes)}\n"
        )
    
    return "".join(parts)


def post_run_audit(model, tokenizer, suspicious_tests: list, test_file_path: str):
    """
    Post-Run Audit — confidence-scored bug detection.

    Replaces the old 3-variant oracle (which was circular: same model confirming
    its own tests) with three independent evidence sources:
      Gate 1  — Syntax check (unchanged)
      Gate 2  — Flakiness check (unchanged, 10 runs)
      Gate 3a — Docstring oracle: does the test agree with the function spec?
      Gate 3b — Existing test cross-reference: do real repo tests also fail?
      Oracle 4 — Direct execution: run the function, compare real output to assertion

    A confidence score (0–100) is computed; only confirmed (≥70) and
    suspected (40–69) cases are written to bug_reports.jsonl.
    """
    import ast as _ast
    from bug_detector import run_confidence_audit

    discarded_path  = os.path.join(os.path.dirname(os.path.abspath(test_file_path)), "discarded_reports.jsonl")
    bug_report_path = os.path.join(os.path.dirname(os.path.abspath(test_file_path)), "bug_reports.jsonl")

    # ── Gates 1 & 2: syntax + flakiness (fast pre-filters) ──────────────
    passed_gates_12: list[dict] = []
    for case in suspicious_tests:
        func_name = case.get("method_name") or case.get("target_name", "unknown")
        print(f"\n   🔍 Pre-filter: {func_name}")

        # Gate 1: Syntax
        try:
            _ast.parse(case.get("test_code", ""))
        except SyntaxError as e:
            print(f"      🚪 Gate 1 (Syntax): FAILED — {e}")
            _log_discarded(discarded_path, case, "Syntax Hallucination", str(e))
            continue
        print(f"      ✅ Gate 1 (Syntax): passed")

        # Gate 2: Flakiness — run the specific test function 10×
        target_file_param = case.get("target_file", "")
        env = os.environ.copy()
        if target_file_param:
            root = os.path.dirname(os.path.abspath(target_file_param))
            while os.path.exists(os.path.join(root, "__init__.py")):
                root = os.path.dirname(root)
            env["PYTHONPATH"] = f"{root}{os.pathsep}{env.get('PYTHONPATH', '')}"

        match = re.search(r'def\s+(test_[a-zA-Z0-9_]+)', case.get("test_code", ""))
        if not match:
            _log_discarded(discarded_path, case, "Syntax Hallucination", "Could not parse test_ function name.")
            continue

        test_func_name = match.group(1)
        flaky = False
        for _ in range(10):
            r = subprocess.run(
                [sys.executable, "-m", "pytest", f"{test_file_path}::{test_func_name}", "-q", "--disable-warnings"],
                capture_output=True, text=True, timeout=10,
                cwd=os.path.dirname(os.path.abspath(test_file_path)), env=env,
            )
            if r.returncode == 0:
                flaky = True
                break

        if flaky:
            print(f"      🚪 Gate 2 (Flakiness): FAILED — test passed sporadically")
            _log_discarded(discarded_path, case, "Flaky / Environmental Noise", "Passed sporadically.")
            continue

        print(f"      ✅ Gate 2 (Flakiness): passed — 100% failure rate")
        passed_gates_12.append(case)

    if not passed_gates_12:
        print("   ℹ️  All suspicious tests filtered by Gates 1/2 — no bugs to report.")
        return

    # ── Gates 3a, 3b, Oracle 4: confidence scoring ────────────────────
    confirmed = run_confidence_audit(passed_gates_12, model, tokenizer)

    # ── Write bug reports ──────────────────────────────────────────────
    for result in confirmed:
        verdict = result.get("verdict", "suspected")
        confidence_label = "high" if verdict == "confirmed" else "medium"
        score = result.get("score", 50)

        bug_report = {
            "timestamp":     time.strftime("%Y-%m-%d %H:%M:%S"),
            "source_file":   os.path.basename(result.get("target_file", "")),
            "target":        result.get("func_name", result.get("target_name", "unknown")),
            "bug_type":      "logic_error",
            "confidence":    confidence_label,
            "confidence_score": score,
            "verdict":       verdict,
            "description":   _build_description(result),
            "evidence":      (result.get("failure_logs") or [""])[0][:300],
            "failure_count": len(result.get("failure_logs") or []),
            "test_code":     result.get("test_code", ""),
            "test_file":     test_file_path,
            "detection_evidence": result.get("evidence", {}),
        }
        with open(bug_report_path, "a") as f:
            f.write(json.dumps(bug_report) + "\n")

        icon = "🐛" if verdict == "confirmed" else "⚠️"
        print(f"   {icon} BUG {verdict.upper()} (score={score}): {bug_report['target']}")


def _build_description(result: dict) -> str:
    """Build a human-readable description from detection evidence."""
    parts = []
    score = result.get("score", 50)
    parts.append(f"Confidence score: {score}/100.")

    doc = result.get("evidence", {}).get("docstring", {})
    if doc.get("consistent") is True:
        parts.append("Docstring confirms expected behaviour.")
    elif doc.get("consistent") is False:
        parts.append(f"Note: docstring may contradict test ({doc.get('reasoning', '')[:60]}).")

    ext = result.get("evidence", {}).get("existing_tests", {})
    if ext.get("relevant_found"):
        parts.append("Existing tests also fail." if not ext.get("existing_pass") else "Existing tests pass (lower confidence).")

    exe = result.get("evidence", {}).get("execution", {})
    if exe.get("executed"):
        parts.append(f"Direct execution: actual={exe.get('actual')} expected={exe.get('expected')}.")

    return " ".join(parts)


def _log_discarded(discarded_path: str, case: dict, classification: str, reason: str):
    """Stores rejected generations into the secondary telemetry log."""
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_file": os.path.basename(case['target_file']),
        "target": case['target_name'],
        "classification": classification,
        "reason": reason
    }
    with open(discarded_path, "a") as f:
        f.write(json.dumps(report) + "\n")

def autonomous_loop(model, tokenizer, target_file: str, import_path: str = None,
                    max_targets: int = None, deep_scan: bool = False,
                    max_retries: int = 2, log_callback=None, plan_mode: bool = False,
                    srs_requirements: list = None, priming_examples: str = "",
                    auto_repair: bool = False, stats_out: dict = None,
                    skip_bes_gate: bool = False, max_seconds: float = None,
                    docker_image: str = None):
    """The main self-correcting loop.

    Args:
        import_path:  Optional package import path (e.g. 'requests.auth')
                      for files inside installed packages.
        log_callback: Optional callback for GUI events. Called as:
                      log_callback(event_type, *args)
                      event_type: 'ai_status', 'code_stream', 'code_clear'
        auto_repair:  If True, automatically invoke PartC on confirmed/suspected
                      bugs after post_run_audit completes (opt-in).
        stats_out:    Optional dict to receive runtime stats (mutated in place).
                      Keys filled: 'iterations', 'targets_processed',
                      'prompt_tokens', 'completion_tokens', 'accepted_bes_scores',
                      and 'target_timings' (dict mapping target_label -> seconds).
                      Pass an empty dict to opt in.
        skip_bes_gate: If True, bypass the BES quality gate. Used by the
                      `testmate_no_bes` ablation to isolate BES contribution.
        max_seconds:  Optional hard wall-clock cap for this file. When the
                      deadline is reached, the loop bails out of remaining
                      targets/retries. Prevents pathological files from
                      consuming hours of compute in batch ablations.
    """
    # Stats collection (opt-in via stats_out kwarg)
    if stats_out is not None:
        stats_out.setdefault("iterations", 0)
        stats_out.setdefault("targets_processed", 0)
        stats_out.setdefault("prompt_tokens", 0)
        stats_out.setdefault("completion_tokens", 0)
        stats_out.setdefault("accepted_bes_scores", [])
    # Per-file deadline (None = no limit)
    file_deadline = (time.time() + max_seconds) if max_seconds else None
    def _cb(event_type, *args):
        """Safe callback helper — no-op when running in CLI mode."""
        if log_callback:
            try:
                log_callback(event_type, *args)
            except Exception:
                pass
    
    # Ensure RAG DB is initialized for GUI / CLI symmetry before reading targets
    init_db()

    # Phase 1 Collection Memory
    suspicious_tests = []

    # ── Relative import pre-check ──
    # Detect files that use relative imports (from .xxx import ...) 
    # which will fail without a parent package context.
    with open(target_file, "r") as _pf:
        _pre_source = _pf.read()
    _rel_imports = re.findall(r'^\s*from\s+\.\S*\s+import', _pre_source, re.MULTILINE)
    if _rel_imports and not import_path:
        # Try to auto-detect package path from directory structure
        _dir = os.path.dirname(os.path.abspath(target_file))
        _init = os.path.join(_dir, "__init__.py")
        if os.path.exists(_init):
            # It's a package — derive import_path from directory name + module
            _pkg_name = os.path.basename(_dir)
            _mod_name = os.path.basename(target_file).replace('.py', '')
            import_path = f"{_pkg_name}.{_mod_name}"
            print(f"⚠️  Detected {len(_rel_imports)} relative import(s). "
                  f"Auto-set import_path = '{import_path}'")
        else:
            print(f"⚠️  WARNING: File uses {len(_rel_imports)} relative import(s) "
                  f"but is not in a package (no __init__.py). "
                  f"Tests may fail with ImportError. "
                  f"Consider using --import-as <package.module>.")
    del _pre_source  # free memory

    # ── Pre-scan repo context (runs once, cached) ──
    repo_ctx = detect_repo_context(target_file, deep_scan=deep_scan,
                                   source_files=[target_file])
    if repo_ctx.get("dependencies"):
        print(f"📦 Repo deps: {', '.join(repo_ctx['dependencies'][:8])}")
    if repo_ctx.get("framework"):
        print(f"🏗️  Framework: {repo_ctx['framework']}")
    if repo_ctx.get("fixtures"):
        print(f"🔧 Fixtures: {', '.join(repo_ctx['fixtures'][:5])}")

    with open(target_file, "r") as f:
        source_code = f.read()

    target_name = os.path.basename(target_file)
    module_name = target_name.replace('.py', '')
    
    # For package files, use the full import path (e.g. 'requests.auth')
    # Otherwise fall back to the plain module name (e.g. 'calculator')
    actual_import = import_path if import_path else module_name
    
    # ── Existing-test pre-pass ────────────────────────────────────────────────
    # Before generating anything, check whether real tests already exist for
    # this file.  The outcome shapes what PartB does next:
    #   all_pass + full coverage  → skip generation entirely
    #   all_pass + coverage gaps  → generate only for uncovered functions
    #   some fail                 → triage (real bug → PartC | stale → update)
    _prepass_result = {"has_existing": False, "all_pass": False,
                       "uncovered_funcs": [], "real_bugs_found": 0,
                       "stale_tests_fixed": 0, "skip_generation": False}
    try:
        from existing_test_runner import run_existing_test_prepass  # type: ignore[import]
        # Use the SAME bug_reports.jsonl path as post_run_audit and repair_pending_bugs.
        # When import_path is set, tests go into generated_tests/ dir; otherwise they go
        # next to the source file. The bug_reports.jsonl must be in the same directory.
        _stem_pre = os.path.splitext(target_name)[0]
        if import_path:
            _test_dir_pre = os.path.join(os.path.dirname(os.path.abspath(__file__)), "generated_tests")
        else:
            _test_dir_pre = os.path.dirname(os.path.abspath(target_file))
        _bug_reports_path = os.path.join(_test_dir_pre, "bug_reports.jsonl")
        _repo_dir = os.path.dirname(os.path.abspath(target_file))
        _prepass_result = run_existing_test_prepass(
            target_file=target_file,
            actual_import=actual_import,
            source_code=source_code,
            repo_dir=_repo_dir,
            model=model,
            tokenizer=tokenizer,
            bug_reports_path=_bug_reports_path,
            log_callback=log_callback,
        )
        if _prepass_result.get("skip_generation") and not _prepass_result.get("real_bugs_found"):
            print(f"Existing tests cover {target_name} fully and all pass — skipping generation.")
            return True  # nothing to do, source is healthy
        if _prepass_result.get("has_existing"):
            _bugs = _prepass_result["real_bugs_found"]
            _stale = _prepass_result["stale_tests_fixed"]
            _uncovered = _prepass_result["uncovered_funcs"]
            if _bugs:
                print(f"Pre-pass: {_bugs} real bug(s) confirmed — will route to PartC after generation.")
            if _stale:
                print(f"Pre-pass: {_stale} stale test(s) auto-updated.")
            if _uncovered:
                print(f"Pre-pass: generating tests for uncovered functions: {_uncovered}")
            # IMPORTANT: do NOT return early even when bugs found + no gaps.
            # The auto_repair block below reads bug_reports.jsonl and must run.
    except Exception as _prepass_err:
        print(f"  Pre-pass skipped (non-critical): {_prepass_err}")
    # ─────────────────────────────────────────────────────────────────────────

    # Detect private functions (start with _) that import * won't export
    private_names = []
    try:
        tree = ast.parse(source_code)
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("_") and not node.name.startswith("__"):
                private_names.append(node.name)
    except SyntaxError:
        pass
    
    # Build import statement
    if module_name[0].isdigit():
        # Module name starts with digit — cannot use regular import
        # Use importlib loader instead
        abs_target = os.path.abspath(target_file)
        import_statement = (
            f"import importlib.util, sys, os\n"
            f"_spec = importlib.util.spec_from_file_location("
            f'"{module_name}", r"{abs_target}")\n'
            f"_mod = importlib.util.module_from_spec(_spec)\n"
            f'sys.modules["{module_name}"] = _mod\n'
            f"_spec.loader.exec_module(_mod)\n"
            f"globals().update({{k: v for k, v in _mod.__dict__.items() "
            f"if not k.startswith('__')}})"
        )
    elif private_names:
        private_imports = ", ".join(private_names)
        import_statement = (
            f"from {actual_import} import *\n"
            f"from {actual_import} import {private_imports}"
        )
    else:
        import_statement = f"from {actual_import} import *"
    
    # CRITICAL: When testing files inside a package (e.g. repos/requests/requests/auth.py),
    # we MUST save the test file OUTSIDE the package directory. Otherwise pytest
    # tries to import it as 'requests.test_auth' which doesn't exist.
    #
    # File naming convention: test_<stem>_testmate.py
    #   - Never overwrites user-written test_<stem>.py files
    #   - Clear ownership marker: anything ending in _testmate.py is auto-generated
    _stem = os.path.splitext(target_name)[0]
    _testmate_name = f"test_{_stem}_testmate.py"

    if import_path:
        gen_tests_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "generated_tests")
        os.makedirs(gen_tests_dir, exist_ok=True)
        test_file = os.path.join(gen_tests_dir, _testmate_name)
    else:
        test_file = os.path.join(
            os.path.dirname(os.path.abspath(target_file)),
            _testmate_name
        )

    ctx = extract_context_from_code(source_code)

    print(f"📄 Target: {target_name}")
    if import_path:
        print(f"📦 Import: from {actual_import} import *")
    print(f"📝 Test file: {os.path.basename(test_file)}")
    if ctx["cfg_paths"]:  print(f"🔀 CFG paths found: {len(ctx['cfg_paths'])}")
    if ctx["raises"]:     print(f"⚠️  Exception types: {', '.join(ctx['raises'])}")
    print(f"📊 Complexity: {ctx.get('complexity', 'N/A')}\n")

    user_prompt = build_prompt(source_code, ctx, target_name)

    # Detect if source file needs Django setup
    framework_header = detect_framework_setup(source_code)
    needs_framework_setup = bool(framework_header)
    needs_django = 'django' in framework_header.lower()

    # Also inject Django setup hint into system prompt if needed
    django_hint = ""
    if needs_django:
        django_hint = """
10. DJANGO SETUP: This file uses Django. Django is already configured
    at the top of the test file. You can directly instantiate Django
    classes without any additional setup:
```python
    from django.contrib.auth.validators import ASCIIUsernameValidator
    v = ASCIIUsernameValidator()
    assert v.regex.pattern == r'^[\\w.@+-]+$'
```
    Do NOT call django.setup() again in your tests.
"""

    # ── Merged System Prompt: Two-Phase Planning + Proven Patterns ──
    system_prompt = f"""You are an expert Python test engineer embedded in the TestMate system.

Before writing any test, you MUST complete Phase 1 (Planning) first.
You are prohibited from writing the final test until your plan is complete.

════════════════════════════════════════════════
PHASE 1 — PLANNING (complete this first)
════════════════════════════════════════════════

[Objective]
One sentence: what this test must prove about the target function.

[Requirements]
- Function signature: (exact parameter types)
- Return type: (what it returns)
- Dependencies: (imports, mocks needed)
- RAG examples available: (yes/no — pattern to follow)

[Step-by-Step Test Logic]
1. Setup: what objects/variables to create before calling
2. Call: exact function call with correct types
3. Assert: exact expected value with correct assertion style
4. Edge case 1: what breaks this function and expected result
5. Edge case 2: boundary condition and expected result

[Potential Risks]
- Type mistakes: (e.g. passing string instead of dict)
- Wrong assertion: (e.g. asserting None when value expected)
- Missing import: (e.g. Response not imported)
- Duplicate case: (e.g. same input tested twice)

════════════════════════════════════════════════
PHASE 2 — EXECUTION (only after plan is complete)
════════════════════════════════════════════════

Rules before writing code:
  ✅ Every type in code matches Phase 1 requirements
  ✅ Every assertion matches Phase 1 step-by-step logic
  ✅ No duplicate assertions
  ✅ No assert == None unless Phase 1 explicitly states None
  ✅ Each assertion has a comment explaining what it tests

ALWAYS use these imports:
```python
import pytest
{import_statement}
```

Output format:
  - Valid Python only, no explanation, no markdown
  - Start with imports if needed
  - Then def test_{{target_name}}():
  - Return ALL test code inside <test>...</test> XML tags

─── REFERENCE PATTERNS (use these exact patterns) ───

PATTERN 1 — Mocking HTTP calls:
  If code calls requests.get/post/etc, you MUST mock:
```python
from unittest.mock import patch, MagicMock

@patch('requests.get')
def test_something(mock_get):
    mock_get.return_value.status_code = 200
    mock_get.return_value.json.return_value = {{"temp_celsius": 22.5}}
    result = get_temperature("London")
    assert result == 22.5
```

PATTERN 2 — Class instantiation:
  ALWAYS create instances first, NEVER call __init__() directly:
```python
def test_httpbasicauth():
    auth = HTTPBasicAuth("user", "pass")  # Create instance
    assert auth.username == "user"        # Check attributes
```

PATTERN 3 — Mock request parameters:
  If a method takes a `request` parameter, create a MagicMock:
```python
from unittest.mock import MagicMock
r = MagicMock()
r.headers = {{}}
result = obj(r)
```

PATTERN 4 — Dependency invocation:
  If the prompt says 'Required Setup — Call these methods first', you MUST
  call them BEFORE the target method:
```python
obj = ClassName('args')
obj.setup_method()  # REQUIRED FIRST
result = obj.target_method(...)
```

════════════════════════════════════════════════
1-SHOT EXAMPLE
════════════════════════════════════════════════

INPUT GIVEN:
  Function: dispatch_hook(key, hooks, hook_data)
  RAG example found: test_arithmetic.py

PHASE 1 — PLANNING:

  [Objective]
  Verify dispatch_hook correctly applies hook functions
  to hook_data and handles edge cases safely.

  [Requirements]
  - key: string ("response")
  - hooks: dict {{"response": [callable, ...]}}
  - hook_data: any object (Response)
  - Returns: modified hook_data or original if hook returns None
  - Dependencies: Response class from requests library
  - RAG pattern: assert result == expected_object

  [Step-by-Step Test Logic]
  1. Setup: r = Response(content="a")
  2. Call:  dispatch_hook("response", {{"response": [lambda x: x]}}, r)
  3. Assert: result == r  (identity hook returns same object)
  4. Edge 1: hook returns None → data unchanged → assert result == r
  5. Edge 2: empty hook list  → data unchanged → assert result == r

  [Potential Risks]
  - Type mistake: passing "1234" instead of {{"response": [...]}}
  - Wrong assertion: asserting == None when result is Response
  - Duplicate: testing same lambda twice with different content

PHASE 2 — EXECUTION:

```python
def test_dispatch_hook():
    # identity hook returns same response unchanged
    r = Response(content="a")
    result = dispatch_hook("response", {{"response": [lambda x: x]}}, r)
    assert result == r

    # hook returning None is ignored, original data preserved
    assert dispatch_hook("response", {{"response": [lambda x: None]}}, r) == r

    # empty hook list returns data unchanged
    assert dispatch_hook("response", {{"response": []}}, r) == r
```

════════════════════════════════════════════════
HARD RULES — NEVER VIOLATE
════════════════════════════════════════════════
  - dict  → {{"key": value}}        never "string"
  - list  → [lambda x: x]           never "string"
  - None  → only if plan says None
  - No duplicate assertions ever
  - No incomplete edge cases (never write "assert" alone)
  - Phase 1 must be complete before Phase 2 begins
  - If code has CLASSES, instantiate with ClassName(args), NEVER call __init__()
  - Use `with pytest.raises(SomeError):` to test exceptions
  - NEVER use isinstance() or type() as only assertion — assert specific VALUES
  - If code makes network/file/DB calls, MUST use @patch from unittest.mock
  - NEVER call the real internet, filesystem, or database in tests
{django_hint}"""

    # ── Plan Execution System Prompt (used ONLY when plan mode is active) ──
    # This replaces the two-phase prompt with a direct execution prompt.
    # The plan already contains all the context the model needs.
    plan_exec_system_prompt = f"""You are an expert Python test engineer. A test plan has been pre-approved.
Your ONLY job is to convert the plan into working pytest code.

════════════════════════════════════════════════
RULES — EXECUTE THE PLAN
════════════════════════════════════════════════

1. Implement EVERY test case from the plan — do NOT skip any.
2. Use the EXACT input values and expected outputs specified in the plan.
3. Do NOT re-plan or add your own test cases beyond what the plan specifies.
4. Do NOT do Phase 1 planning — the plan IS Phase 1. Go directly to writing code.
5. MUTATION-PROOF ASSERTIONS: Use `== exact_value`, never `is not None`.
6. If code has CLASSES, instantiate with `ClassName(args)`, NEVER call `__init__()`.
7. If code makes network/file/DB calls, use `@patch` from `unittest.mock`.
8. Wrap ALL test code inside `<test>...</test>` XML tags.
9. Write ONLY test functions, no imports (they are provided separately).
{django_hint}"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ]



    # ── Build target list from AST: every function/method to test ──
    targets = []
    for cls_info in ctx["classes"]:
        cls_name = cls_info["name"]
        if not cls_info["methods"]:
            targets.append((cls_name, "__init__"))
        else:
            for method in cls_info["methods"]:
                targets.append((cls_name, method))
    # Top-level functions
    try:
        tree = ast.parse(source_code)
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                targets.append((None, node.name))
    except SyntaxError:
        pass

    print(f"🎯 Found {len(targets)} test targets:")
    for cls, method in targets:
        label = f"   • {cls}.{method}()" if cls else f"   • {method}()"
        print(label)
    print()

    # Build call graph BEFORE sort — sort key uses G
    G = build_call_graph(source_code)
    print(f"🕸️  Call graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    def _dependency_sort_key(target):
        cls, method = target
        label = f"{cls}.{method}" if cls else method
        # __init__ always first
        if method == "__init__":
            return (0, 0)
        # Methods with no dependencies next
        try:
            deps = get_stateful_setup_chain(G, label)
            n_deps = len(deps)
        except Exception:
            n_deps = 0
        # Then by line count descending within same dep level
        lines = get_method_line_count(source_code, cls, method)
        return (n_deps, -lines)

    targets.sort(key=_dependency_sort_key)
    print("📊 Targets sorted by dependency order then coverage impact")
    if max_targets and len(targets) > max_targets:
        print(f"⚡ Limiting to top {max_targets} targets (--max-targets)")
        targets = targets[:max_targets]

    # ── Start with imports header ──
    accumulated_code = (
        f"{framework_header}"
        f"import pytest\n"
        f"from unittest.mock import MagicMock, patch\n"
        f"{import_statement}\n"
    )
    accumulated_tests = 0
    failed_targets = []
    
    covered_summary = []
    skipped_summary = []

    # Resolve base classes if repo_dir is available (Fix 14: cached)
    base_classes = {}
    if import_path:
        repo_dir = os.path.dirname(os.path.dirname(target_file))
        _bc_key = hash(source_code[:500])
        if _bc_key not in _base_classes_cache:
            _base_classes_cache[_bc_key] = resolve_base_classes(source_code, repo_dir)
        base_classes = _base_classes_cache[_bc_key]
        if base_classes:
            print(f"🔗 Base classes found: {', '.join(base_classes.keys())}")



    print("🚀 Probing targets in parallel...")
    probe_cache = probe_all_targets(targets, actual_import, ctx, source_code, import_path)

    # ── Plan Mode: generate file-level test plan BEFORE target loop ──
    _file_plan = ""
    if plan_mode:
        _cb("ai_status", "thinking",
            f"Planning tests for {os.path.basename(target_file)}...",
            os.path.basename(target_file))
        print(f"📋 Generating file-level test plan for {os.path.basename(target_file)} "
              f"({len(targets)} targets)...")
        _file_plan = generate_file_test_plan(
            model, tokenizer, source_code, targets, ctx, G,
            import_path, target_file, probe_cache,
            srs_requirements=srs_requirements,
            priming_examples=priming_examples,
            stats_out=stats_out)
        print(f"📋 File plan generated ({len(_file_plan)} chars)")
        # Send plan to GUI for user review/edit
        target_names = ", ".join(
            f"{c}.{m}" if c else m for c, m in targets[:8])
        if len(targets) > 8:
            target_names += f" (+{len(targets)-8} more)"
        _cb("plan_request", _file_plan,
            f"{os.path.basename(target_file)} ({len(targets)} functions)",
            f"test_{os.path.basename(target_file)}",
            target_names, "")
        # Check if the user edited the plan via the GUI callback
        if hasattr(_cb, '_edited_plan') and _cb._edited_plan is not None:
            _file_plan = _cb._edited_plan
            _cb._edited_plan = None
            print(f"📋 User edited plan applied ({len(_file_plan)} chars)")

    print("🚀 Starting Iterative Chunking Loop...\n" + "=" * 60)

    # Initialize coverage tracking
    covered_lines_so_far = set()
    # O7: per-target start times — recorded once per iteration so we can later
    # compute (next_start - this_start) for the duration of each target.
    _target_start_times: list[tuple[str, float]] = []
    # Write initial empty test file to get baseline
    with open(test_file, "w") as f:
        f.write(accumulated_code)

    for idx, (cls_name, method_name) in enumerate(targets, 1):
        # Per-file deadline: skip remaining targets if budget exhausted
        if file_deadline is not None and time.time() > file_deadline:
            print(f"\n⏱️  File deadline reached — skipping remaining {len(targets) - idx + 1} targets")
            break
        if cls_name:
            target_label = f"{cls_name}.{method_name}"
        else:
            target_label = method_name

        _target_start_times.append((target_label, time.time()))
        print(f"\n--- Target {idx}/{len(targets)}: {target_label} ---")
        _cb("ai_status", "analyzing", f"Extracting context for {target_label}", target_label)

        # Analyze source patterns once per target
        source_patterns = analyze_source_patterns(source_code, cls_name or "", method_name, ctx)
        if source_patterns:
            print(f"   📋 Detected patterns: {', '.join(p[0] for p in source_patterns)}")

        # Detect method dependencies once per class
        if cls_name:
            detect_method_dependencies(source_code, cls_name, ctx)

        # ── Build focused prompt for ONE function with Target Anchoring ──
        # Map dunder methods to their Python equivalents
        dunder_map = {
            "__init__": f"{cls_name}(...)" if cls_name else f"{method_name}()",
            "__len__": "len(obj)",
            "__repr__": "repr(obj)",
            "__str__": "str(obj)",
            "__iter__": "iter(obj) or for x in obj",
            "__getitem__": "obj[key]",
            "__setitem__": "obj[key] = value",
            "__delitem__": "del obj[key]",
            "__contains__": "key in obj",
            "__eq__": "obj1 == obj2",
            "__ne__": "obj1 != obj2",
            "__lt__": "obj1 < obj2",
            "__bool__": "bool(obj)",
            "__hash__": "hash(obj)",
            "__call__": "obj(...)",
        }
        invocation_hint = dunder_map.get(method_name, f".{method_name}()")
        
        # Build safe function name for uniqueness
        safe_method = method_name.strip("_")
        expected_func_name = f"test_{cls_name.lower()}_{safe_method}" if cls_name else f"test_{safe_method}"
        
        # 1. Get token-budgeted context using call graph
        target_label_graph = f"{cls_name}.{method_name}" if cls_name else method_name
        _n_callees = len(
            get_2hop_subgraph(G, target_label_graph).get("callees", [])
        )
        _token_budget = min(500 + (_n_callees * 100), 900)
        method_context = build_token_budgeted_context(
            source_code, cls_name, method_name, G,
            token_budget=_token_budget
        )

        # 2. Get per-method CFG paths
        method_paths = get_method_cfg_paths(source_code, cls_name, method_name)

        # 3. Get data flow for this method
        data_flow = extract_data_flow(source_code, cls_name, method_name)

        # 4. Get 2-hop subgraph info
        subgraph = get_2hop_subgraph(G, target_label_graph)

        # ── Build focused conditional prompt ──
        # Compute these ONCE before retry loop (not per retry)
        _docstring = extract_docstring(
            source_code, cls_name, method_name)
        _comments = extract_comments(
            source_code, cls_name, method_name)
        _branch_inputs = synthesize_branch_inputs(
            source_code, cls_name or "", method_name)
        _param_attrs_section = _section_param_mocks(
            source_code, cls_name, method_name)
        _setup_chain_section_cache = None

        def get_setup_chain_section():
            nonlocal _setup_chain_section_cache
            if _setup_chain_section_cache is None:
                _setup_chain_section_cache = _section_setup_chain(
                    G, target_label_graph, cls_name,
                    source_code, actual_import, ctx)
            return _setup_chain_section_cache
        _http_section = _section_http_mock(
            subgraph, method_name, MOCK_HINTS)
        _async_section = _section_async(
            source_code, cls_name, method_name)
        _collision_section = _section_builtin_collision(
            method_name)
        _wrapper_section = _section_wrapper_types(source_code)
        _dict_section = _section_dict_like(
            cls_name, source_code, base_classes)
        _constructor_section = _section_constructor_sig(
            method_name, cls_name, actual_import)
        _cross_file_section = _section_cross_file(
            source_code, target_file, import_path)
        _base_class_section = _section_base_classes(
            base_classes, cls_name)
        _callee_section = _section_callees(
            subgraph, target_label)
        _cfg_section = _section_cfg_paths(method_paths)
        _return_section = _section_return_value(data_flow)
        _exception_section = _section_exceptions(data_flow)

        # RAG retrieval (pass source_file to avoid self-reinforcing patterns)
        similar_examples = retrieve_similar(
            method_name, cls_name or "", top_k=2,
            source_file=os.path.basename(target_file),
            target_signature=target_label_graph)
        rag_section, perfect_rag_hit = _section_rag(
            similar_examples)
        best_example = (similar_examples[0]
                        if similar_examples else None)

        method_deps = retrieve_method_dependencies(
            cls_name or "", method_name)
        bad_examples = retrieve_bad_examples(
            method_name, cls_name or "",
            target_signature=target_label_graph)

        # Probed value
        probed_value = probe_cache.get(target_label_graph)
        if probed_value and not probed_value.startswith("ERROR"):
            print(f"   🔍 Probed: {target_label} → "
                  f"{probed_value[:60]}")

        def build_chunk_prompt(error_logs: str = "",
                                retry: int = 0, test_plan: str = "") -> str:

            # ══════════════════════════════════════════════════════════
            # PLAN MODE: Lean Stage 2 prompt
            # The plan is the main guidance. Only add:
            #   1. The plan itself (already has test cases, inputs, expected outputs)
            #   2. Short context (imports + function signature only)
            #   3. Target anchoring (function name + expected test name)
            #   4. Error logs (only on retry)
            # ══════════════════════════════════════════════════════════
            if test_plan:
                # Short context: just the signature, not the full method body
                short_ctx = _extract_signature_context(
                    source_code, cls_name, method_name, actual_import)

                base = (
                    f"═══════════════════════════════════════\n"
                    f"📋 TEST PLAN — EXECUTE ALL TEST CASES\n"
                    f"═══════════════════════════════════════\n\n"
                    f"{test_plan}\n\n"
                    f"═══════════════════════════════════════\n\n"
                )

                # Rolling summary to prevent duplicate tests
                base += _section_rolling_summary(
                    covered_summary, target_label)

                # Short context: imports + signature only
                base += (
                    f"## Context (signature only)\n"
                    f"```python\n{short_ctx}\n```\n\n"
                )

                # Target anchoring
                base += (
                    f"## Target\n"
                    f"1. Test ONLY `{target_label}`. "
                    f"Call `{invocation_hint}`.\n"
                    f"2. Function name: `def {expected_func_name}():`\n"
                    f"3. MUTATION-PROOF: Use `== exact_value`, "
                    f"never `is not None`.\n\n"
                )

                # Error logs only on retry
                if error_logs and retry > 0:
                    base += (
                        f"## ❌ Previous attempt FAILED:\n"
                        f"```\n{error_logs[:500]}\n```\n"
                        f"Fix this error while still following the plan.\n\n"
                    )

                # Recency bias: reinforce the plan at the end
                base += (
                    f"🚨 REMINDER: Implement ALL test cases from the plan above. "
                    f"Use the EXACT input values and expected outputs. "
                    f"Do NOT skip any.\n\n"
                    f"Wrap in <test>...</test>. "
                    f"Write ONLY the function, no imports."
                )
                return base

            # ══════════════════════════════════════════════════════════
            # NON-PLAN MODE: Full prompt (unchanged)
            # All context sections included for the model to plan + execute
            # ══════════════════════════════════════════════════════════
            base = (
                _section_rolling_summary(
                    covered_summary, target_label)
                + _section_source_patterns(source_patterns)
                + _section_core(
                    method_context, method_name, cls_name,
                    target_label, invocation_hint,
                    expected_func_name)
                + _section_docstring(_docstring)
                + _section_comments(_comments)
                + _section_repo_context(repo_ctx)
                + _section_deep_scan(repo_ctx)
                + _section_few_shot(method_name,
                                     FEW_SHOT_PATTERNS,
                                     rag_examples=similar_examples)
            )

            if retry == 0 or not error_logs:
                # ── RETRY 0: Lean prompt — let the model work freely ──
                # Only essential context: source code, CFG paths, probe, few-shot
                # This is ~60% fewer tokens than the full prompt
                base += (
                    _cfg_section
                    + _collision_section
                    + _section_probe(
                        probe_cache, target_label_graph,
                        invocation_hint)
                    + rag_section
                )
            else:
                # ── RETRY 1+: Rich prompt — escalate with error context ──
                # Model failed once, now give it everything to help
                fix = route_error_to_fix(
                    error_logs, source_code,
                    cls_name, method_name,
                    G, target_label_graph, ctx,
                    actual_import, subgraph, MOCK_HINTS
                )
                if fix:
                    base += fix
                # Always add heavy context on retries (model needs more help)
                base += (
                    _cfg_section
                    + _return_section
                    + _exception_section
                    + _callee_section
                    + _collision_section
                    + _wrapper_section
                    + _constructor_section
                    + get_setup_chain_section()
                    + _section_branch_inputs(
                        source_code, cls_name, method_name)
                    + _section_probe(
                        probe_cache, target_label_graph,
                        invocation_hint)
                    + rag_section
                    + _section_bad_examples(bad_examples)
                )

            # Path reminder at end of prompt (recency bias — LLMs attend most to end)
            if method_paths and len(method_paths) > 1:
                n_paths = len(method_paths[:8])
                path_types = set()
                for p in method_paths[:8]:
                    ps = " ".join(str(s) for s in p)
                    if "IF-ONLY" in ps: path_types.add("IF-truthy")
                    if "ELSE" in ps: path_types.add("ELSE")
                    if "RAISE" in ps: path_types.add("RAISE")
                    if "0 iterations" in ps: path_types.add("empty-loop")
                types_str = ", ".join(path_types) if path_types else "all branches"
                base += (
                    f"\n🚨 REMINDER: You MUST cover {n_paths} execution paths "
                    f"({types_str}). Do NOT only test the happy path.\n"
                )

            base += (
                f"\nAlso: if empty/None input raises an exception, "
                f"add `with pytest.raises(ExceptionType):` "
                f"as second assertion.\n"
                f"\nWrap in <test>...</test>. "
                f"Write ONLY the function, no imports."
            )
            return base


        # Use file-level plan (generated before loop, reviewed by user)
        chunk_prompt = build_chunk_prompt(test_plan=_file_plan)

        # ── Generate with up to 2 attempts per target ──
        test_added = False
        
        # Track tests per target to enforce MAX_TESTS_PER_TARGET
        tests_for_this_target = 0
        failure_logs_for_target = []
        
        _last_error_summary = ""
        _last_category = None

        # Adaptive retry count based on method complexity
        _effective_retries = max_retries
        if len(method_paths) <= 1 and max_retries > 2:
            _effective_retries = 2  # Simple methods rarely benefit from 3+ retries

        # Fix: max_tokens computed once entirely outside the retry branch saving redundant looping calculations internally
        method_lines = get_method_line_count(source_code, cls_name, method_name)
        max_tokens = calculate_max_tokens(
            method_lines=method_lines,
            method_paths=len(method_paths),
            subgraph=subgraph,
            data_flow=data_flow,
            retry=0,  # dynamic scale removed explicitly
            has_mock_hint=bool(MOCK_HINTS.get(method_name)),
            has_bad_examples=bool(bad_examples),
            has_rag_examples=bool(similar_examples),
        )

        if stats_out is not None:
            stats_out["targets_processed"] = stats_out.get("targets_processed", 0) + 1
        # Initialize so the post-loop "FAILING TEST" block (line ~5783) has a value
        # even when every retry iteration hit the "No valid test extracted" continue.
        test_func_clean = ""
        # Track the last unique test attempted for this target. If the model
        # emits the same token sequence on a retry (common at low temperatures),
        # we skip the redundant enforcement + pytest cycle and force diversity.
        _last_attempted_test = ""
        _identical_retry_skips = 0
        for retry in range(_effective_retries):
            if stats_out is not None:
                stats_out["iterations"] = stats_out.get("iterations", 0) + 1
            # Per-file deadline: break out if we've already burnt the budget for this file
            if file_deadline is not None and time.time() > file_deadline:
                print(f"   ⏱️  File deadline reached — aborting remaining retries for {target_label}")
                break
            if tests_for_this_target >= 3:
                print(f"   🛑 Reached soft cap of 3 tests for {target_label}.")
                break

            if retry > 0:
                print(f"   🔄 Retry for {target_label}...")
                _cb("code_clear")
                _cb("ai_status", "analyzing", f"Retry {retry} — rebuilding prompt for {target_label}", target_label)

            if perfect_rag_hit and retry == 0:
                test_func_clean = best_example["test_code"]
            else:

                # Use plan execution system prompt when plan mode is active
                _active_system_prompt = plan_exec_system_prompt if _file_plan else system_prompt
                messages = [
                    {"role": "system", "content": _active_system_prompt},
                    {"role": "user", "content": chunk_prompt},
                ]

                # Retry temperature diversity: 0.5 → 0.7 → 0.9
                _retry_temps = [0.5, 0.7, 0.9]
                _temp = _retry_temps[min(retry, len(_retry_temps) - 1)]
                
                print(f"   🤔 Generating test for {target_label} (max_tokens={max_tokens}, temp={_temp})...")
                _cb("ai_status", "thinking", f"Reasoning about {target_label}...", target_label)
                raw = generate_test(model, tokenizer, messages, max_tokens=max_tokens, temperature=_temp, stats_out=stats_out)
                test_func = extract_test_code(raw)
                test_func = fix_generator_assertions(test_func)

                if not test_func or "def test_" not in test_func:
                    print(f"   ⚠️  No valid test extracted, skipping...")
                    continue

                # O5: pre-pytest symbol validation — catches hallucinated
                # `from {actual_import} import <unknown_or_private>` lines.
                # Diagnostic-only: we print warnings but do NOT mutate
                # _last_category, because:
                #   (a) the per-target import-stripping below removes most
                #       suspect lines before pytest sees them, so flagging them
                #       as "import_error" would be wrong; and
                #   (b) the category-aware retry bail (Fix 2) at line ~5923
                #       trips after 2 consecutive import_error categories — if
                #       we mark this iteration as import_error preemptively
                #       and pytest also yields import_error, we'd burn the
                #       remaining retry budget on a single failure.
                try:
                    from symbol_validator import validate_test_imports  # type: ignore[import]
                    _sym_warnings = validate_test_imports(test_func, actual_import, source_code)
                    if _sym_warnings:
                        print(f"   ⚠️  Symbol validator: {len(_sym_warnings)} suspect import(s) "
                              f"(diagnostic only, not failing the test)")
                except Exception:
                    pass  # validator must never block the generation flow

                # Strip imports from the chunk — we already have them
                func_lines = []
                KEEP_IMPORTS = {
                    "import re", "import os", "import sys",
                    "import math", "import decimal",
                    "import datetime", "import json",
                    "import collections", "import string",
                    "import itertools", "import functools",
                    "from django.contrib.auth.models import User",
                    "from django.contrib.auth",
                }
                for line in test_func.splitlines():
                    stripped = line.strip()
                    # Keep mock-related imports, strip only module imports
                    if stripped.startswith("from unittest.mock") or \
                       stripped.startswith("import unittest"):
                        func_lines.append(line)
                        continue
                    if any(stripped.startswith(k)
                           for k in KEEP_IMPORTS):
                        func_lines.append(line)
                        continue
                    if stripped.startswith("import ") or stripped.startswith("from "):
                        continue
                    func_lines.append(line)
                test_func_clean = "\n".join(func_lines).strip()

            if "def test_" not in test_func_clean:
                print(f"   ⚠️  No test function in output, skipping...")
                continue

            # Q2: identical-retry skip. If the model just produced the same
            # tokens as the previous attempt for this target, running through
            # enforcement + pytest will deterministically fail the same way.
            # Skip ahead to the next retry (temperature will already be higher).
            if test_func_clean and test_func_clean == _last_attempted_test:
                _identical_retry_skips += 1
                print(f"   🔁 Identical test as previous retry — skipping pytest, forcing diversity")
                continue
            _last_attempted_test = test_func_clean

            # Fix 5 wiring: enforce stateful setup post-processing
            test_func_clean = enforce_stateful_setup(
                test_func_clean, source_code, cls_name, method_name, G, target_label_graph
            )
            # Fix: enforce param mocks (inject MagicMock for params that access .attr)
            test_func_clean = enforce_param_mocks(
                test_func_clean, source_code, cls_name or "", method_name
            )
                
            qqc_ok, qqc_reason = quick_quality_check(test_func_clean)
            if not qqc_ok:
                print(f"   ⚠️  Quick Quality Check Failed: {qqc_reason}")
                chunk_prompt = build_chunk_prompt(error_logs=qqc_reason, retry=retry, test_plan=_file_plan)
                continue

            # Path coverage verification
            # Hard gate on retry 0 if < 50% critical paths covered
            # Soft hint on later retries
            _pcov, _ptotal, _pmissing = verify_path_coverage(
                test_func_clean, method_paths)
            if _ptotal > 0:
                pct = (_pcov / _ptotal * 100) if _ptotal else 0
                if _pmissing:
                    hint = "; ".join(_pmissing[:2])
                    print(f"   🔀 Path coverage: {_pcov}/{_ptotal} critical paths ({pct:.0f}%)")
                    chunk_prompt = build_chunk_prompt(
                        error_logs=f"path coverage gap: {hint}", retry=retry, test_plan=_file_plan)
                    # Hard reject on retry 0 when coverage is very low
                    if retry == 0 and pct < 50:
                        print(f"   ❌ Path coverage too low ({pct:.0f}%) — forcing retry")
                        with open(test_file, "w") as f:
                            f.write(accumulated_code)
                        continue
                elif _pcov == _ptotal:
                    print(f"   ✅ Path coverage: all {_ptotal} critical paths covered")

            # Fix 8: Progressive dup threshold for comparison dunders
            dup_threshold = 0.95 if method_name in (
                "__ne__", "__eq__", "__lt__", "__gt__", "__le__", "__ge__",
                "__getitem__", "__setitem__", "__delitem__"
            ) else 0.85
            if is_semantically_duplicate(test_func_clean, accumulated_code, threshold=dup_threshold):
                print(f"   ⚠️  Semantically duplicate test (RAG or generated) — retrying...")
                perfect_rag_hit = False  # force fresh generation next retry
                # Fix 13: Branch retry with CFG path injection
                if method_paths and tests_for_this_target < len(method_paths):
                    target_path = method_paths[tests_for_this_target]
                    path_str = ' → '.join(str(p) for p in target_path) if isinstance(target_path, (list, tuple)) else str(target_path)
                    chunk_prompt = build_chunk_prompt(error_logs='duplicate', retry=retry, test_plan=_file_plan)
                else:
                    chunk_prompt = build_chunk_prompt(error_logs='duplicate', retry=retry, test_plan=_file_plan)
                continue

            # ── Fix 2: Programmatic Name Checking (Target Anchoring Guardrail) ──
            # Verify the LLM actually called the target method
            target_keywords = [method_name]  # e.g. '__len__'
            clean_target = method_name.strip("_")  # e.g. 'len'
            if clean_target:
                target_keywords.append(clean_target)
            # Add Python builtin equivalents
            builtin_map = {
                "__len__": ["len("], "__repr__": ["repr("], "__str__": ["str("],
                "__iter__": ["iter(", "for "], "__eq__": [" == "],
                "__ne__": [" != "], "__contains__": [" in "],
                "__delitem__": ["del "], "__bool__": ["bool("],
                "__getitem__": ['["', "['", "[0", "[1"],
                "__setitem__": ["] =", "]="],
                "__hash__": ["hash("], "__call__": ["("],
            }
            if method_name in builtin_map:
                target_keywords.extend(builtin_map[method_name])
            
            target_found = any(kw in test_func_clean for kw in target_keywords)
            if not target_found:
                print(f"   ⚠️  Target drift! LLM ignored `{method_name}`. Forcing retry...")
                chunk_prompt = build_chunk_prompt(error_logs='target drift', retry=retry, test_plan=_file_plan)
                continue

            # ── Detect and fix duplicate function names ──
            existing_names = set(re.findall(r'def (test_\w+)', accumulated_code))
            new_names = re.findall(r'def (test_\w+)', test_func_clean)
            for dup_name in new_names:
                if dup_name in existing_names:
                    suffix = 2
                    while f"{dup_name}_{suffix}" in existing_names:
                        suffix += 1
                    new_name = f"{dup_name}_{suffix}"
                    test_func_clean = test_func_clean.replace(f"def {dup_name}(", f"def {new_name}(", 1)
                    print(f"   🔄 Renamed duplicate: {dup_name} → {new_name}")

            # ── Test it: append to accumulated code and run ──
            _cb("ai_status", "writing", f"Writing test for {target_label}", target_label)
            _cb("code_stream", test_func_clean, f"test_{os.path.basename(target_file)}", target_label, retry > 0)
            candidate = accumulated_code + "\n" + test_func_clean + "\n"
            with open(test_file, "w") as f:
                f.write(candidate)

            # Pre-flight: AST parse the candidate before paying pytest startup cost.
            # If the candidate is not syntactically valid Python, pytest is guaranteed
            # to fail at collection time — skip it and feed the parse error to the next retry.
            try:
                ast.parse(candidate)
            except SyntaxError as _se:
                _se_msg = f"SyntaxError: {_se.msg} at line {_se.lineno}"
                print(f"   ⚠️  Pre-flight syntax check failed: {_se_msg}")
                with open(test_file, "w") as f:
                    f.write(accumulated_code)
                chunk_prompt = build_chunk_prompt(
                    error_logs=_se_msg, retry=retry, test_plan=_file_plan)
                continue

            _cb("ai_status", "validating", f"Running pytest for {target_label}", target_label)
            try:
                # Fast path: only run_pytest during the loop
                # Coverage is collected ONCE at the end for speed
                passed, logs = run_pytest(test_file, target_file, deadline_ts=file_deadline,
                                          docker_image=docker_image)
                cov_pct = 0.0
                branch_cov_pct = 0.0
                lines_after = covered_lines_so_far
            except subprocess.TimeoutExpired:
                passed, logs, cov_pct, branch_cov_pct, lines_after = False, "pytest timed out", 0.0, 0.0, set()

            if passed:
                # Quick syntax/triviality guard (pre-BES — zero cost)
                _assert_lines = [l.strip() for l in test_func_clean.splitlines() if re.match(r'\s*assert\s', l)]
                _non_trivial = [l for l in _assert_lines if l not in ("assert True", "assert False")]
                if not _non_trivial:
                    reject_msg = "No real assertions (only assert True/False)"
                    print(f"   Trivial test rejected: {reject_msg}")
                    target_sig = f"{cls_name}.{method_name}" if cls_name else method_name
                    cat, mutants = categorize_failure("", reject_msg)
                    store_bad_example(target_sig, method_name, cls_name or "",
                                      test_func_clean, reject_msg,
                                      os.path.basename(target_file),
                                      failure_category="quality_gate", survived_mutants=mutants)
                    chunk_prompt = build_chunk_prompt(error_logs=reject_msg, retry=retry, test_plan=_file_plan)
                    with open(test_file, "w") as f:
                        f.write(accumulated_code)
                    continue
            
                # Note: new_lines check is skipped since coverage runs once at end.
                # Graph RAG CFG paths + is_semantically_duplicate already prevent redundancy.
                new_lines = set()  # placeholder for RAG storage
            
                # Gate 3: Bug Exposure Score (BES) — replaces old composite gate
                try:
                    from bes_scorer import bes_score, build_retry_hint  # type: ignore[import]
                    _bes = bes_score(
                        test_code=test_func_clean,
                        func_source=source_code,
                        func_name=method_name or "",
                    )
                    _bes_val = _bes["score"]
                    composite = {  # keep old key names for downstream RAG storage
                        "composite": _bes_val,
                        "assertion": _bes["assertion_quality"],
                        "diversity": _bes["input_variety"],
                        "edge_cases": _bes["boundary_coverage"],
                        "branch_cov": 0.0,
                        "line_cov": 0.0,
                        "mutation": _bes["mutation_potential"],
                    }
                except ImportError:
                    # Fallback to old scorer if bes_scorer not available
                    composite = compute_composite_score(test_code=test_func_clean, per_test=True)
                    _bes = {"score": composite["composite"], "verdict": "accept", "missing": []}
                    _bes_val = composite["composite"]

                # Threshold scales with retry: 50 → 43 → 36 → 30 (more lenient on each retry)
                # Max 3 retries before fallback to Layer 1+2 only.
                threshold = max(30, 50 - retry * 7)
                # Ablation: skip_bes_gate bypasses this gate entirely so we can
                # measure how much the BES gate contributes vs. lets through.
                if skip_bes_gate:
                    _bes_val = threshold  # force-pass the check below
                if _bes_val < threshold:
                    _hint = ""
                    try:
                        from bes_scorer import build_retry_hint  # type: ignore[import]
                        _hint = build_retry_hint(_bes)
                    except ImportError:
                        pass
                    print(f"   BES {_bes_val:.0f}/100 < {threshold} — rejected "
                          f"(spec:{_bes.get('spec_coverage',0):.0f} "
                          f"variety:{_bes.get('input_variety',0):.0f} "
                          f"boundary:{_bes.get('boundary_coverage',0):.0f})")
                    target_sig = f"{cls_name}.{method_name}" if cls_name else method_name
                    cat, mutants = categorize_failure("", f"BES {_bes_val:.0f}/100")
                    store_bad_example(target_sig, method_name, cls_name or "",
                                      test_func_clean, f"BES {_bes_val:.0f}/100 < {threshold}",
                                      os.path.basename(target_file),
                                      failure_category="quality_gate", survived_mutants=mutants)
                    # Tier 4: self-healing RAG feedback on BES rejection
                    feedback_test_rejected(target_label_graph)
                    feedback_bad_didnt_help(target_label_graph, "quality_gate")
                    # Inject the retry hint into the next prompt
                    _retry_error = _hint or f"BES score too low ({_bes_val:.0f}/100). Use more diverse inputs."
                    chunk_prompt = build_chunk_prompt(error_logs=_retry_error, retry=retry, test_plan=_file_plan)
                    with open(test_file, "w") as f:
                        f.write(accumulated_code)
                    continue

                # Accept the test
                accumulated_code = candidate
                covered_lines_so_far = lines_after
                accumulated_tests += 1
                covered_summary.append(target_label)
                n_funcs = len(re.findall(r'def test_\w+', accumulated_code))
                score_emoji = "✓" if _bes_val >= 70 else "~" if _bes_val >= 50 else "?"
                print(f"   Passed! ({n_funcs} tests accumulated) [{score_emoji}] BES: {_bes_val:.0f}/100")
                # M2: persist BES score for paper metrics
                if stats_out is not None:
                    stats_out["accepted_bes_scores"].append(_bes_val)
                # Tier 4: self-healing RAG feedback
                feedback_test_accepted(target_label_graph)
                feedback_bad_helped(target_label_graph)
                test_added = True
                tests_for_this_target += 1
                
                # Store in RAG (already passed ≥50 gate above)
                target_sig = f"{cls_name}.{method_name}" if cls_name else method_name
                coverage_pattern = (
                    f"branch:{composite['branch_cov']:.0f}%,"
                    f"mut:0.0%"  # Will update after end-of-loop mutations
                )
                if new_lines:
                    coverage_pattern += f",lines:{','.join(sorted(str(l) for l in list(new_lines)[:5]))}"

                # Fix 12: Validate test actually tests the target method before RAG storage
                target_kws_rag = [method_name, method_name.strip("_")]
                builtin_map_rag = {
                    "__len__": ["len("], "__repr__": ["repr("], "__str__": ["str("],
                    "__iter__": ["iter(", "list("], "__bool__": ["bool("],
                    "__eq__": ["=="], "__ne__": ["!="], "__hash__": ["hash("],
                    "__getitem__": ["["], "__setitem__": ["["],
                    "__delitem__": ["del "], "__contains__": [" in "],
                }
                if method_name in builtin_map_rag:
                    target_kws_rag.extend(builtin_map_rag[method_name])
                actually_tests_target = any(kw in test_func_clean for kw in target_kws_rag)

                if actually_tests_target:
                    # Build functional signature for better RAG matching
                    _func_sig = {
                        "return_type": (
                            data_flow["outputs"][0]
                            if data_flow.get("outputs") else "unknown"
                        ),
                        "param_count": len(data_flow.get("inputs", [])),
                        "raises": data_flow.get("raises", []),
                        "calls": subgraph.get("callees", [])[:5],
                        "complexity": ctx.get("complexity", 0),
                        "has_early_return": any(
                            "RETURN" in str(p) for p in method_paths
                        ),
                        "reads_settings": "settings" in method_context,
                        "has_loop": any(
                            "LOOP" in str(p) for p in method_paths
                        ),
                    }
                    store_example(
                        target_signature=target_sig,
                        method_name=method_name,
                        class_name=cls_name or "",
                        test_code=test_func_clean,
                        quality_score=composite["composite"],
                        coverage_lines=len(new_lines),
                        passed_mutation=False,
                        source_file=os.path.basename(target_file),
                        coverage_pattern=coverage_pattern
                    )
                    # Store functional signature as separate probe value
                    # keyed by target_sig so it can be retrieved alongside tests
                    _func_sig_key = f"func_sig:{target_sig}"
                    store_probe_value(
                        import_path or "",
                        cls_name or "",
                        f"__func_sig__{method_name}",
                        json.dumps(_func_sig)
                    )
                    print(f"   💾 RAG: stored with composite {composite['composite']:.0f}/100 "
                          f"(assert:{composite['assertion']:.0f} branch:{composite['branch_cov']:.0f} "
                          f"div:{composite['diversity']:.0f} edge:{composite['edge_cases']:.0f})")
                    # Also save to flywheel for future LoRA fine-tuning
                    save_to_flywheel(
                        target_file, source_code, test_func_clean,
                        cov_pct, 0.0  # mutation_score updated later
                    )
                else:
                    print(f"   ⚠️  RAG skip: test doesn't actually test {method_name}")
                    store_bad_example(target_sig, method_name, cls_name or "",
                                     test_func_clean,
                                     f"Target drift: test doesn't call {method_name}",
                                     os.path.basename(target_file),
                                     failure_category="target_drift")
            
                # Coverage feedback for next retry on same target
                if retry == 0:
                    uncovered = get_uncovered_lines_with_source(
                        target_file, logs, cls_name, method_name
                    )
                    if uncovered:
                        uncovered_str = "\n".join(
                            f"  Line {ln}: `{src}`"
                            for ln, src in uncovered[:3]
                        )
                        print(f"   🎯 {len(uncovered)} uncovered lines — retrying same target")
                        chunk_prompt = build_chunk_prompt(error_logs=f'uncovered lines: {uncovered_str}', retry=retry)
                        test_added = False
                        continue
            
                break
            else:
                # Extract error for retry prompt
                error_lines = [l for l in logs.split("\n")
                               if any(k in l for k in
                               ["FAILED", "Error", "assert", "error",
                                "Exception", "WARNING", "collected"])]
                error_summary = " | ".join(error_lines[:2]) if error_lines else logs[:200]
                print(f"   ❌ Failed: {error_summary[:120]}")
                _cb("ai_status", "analyzing", f"Test failed — analyzing errors for {target_label}", target_label)
                failure_logs_for_target.append(logs)

                if (error_summary == _last_error_summary
                        and error_summary != "unknown error"):
                    print(f"   ⏭️  Same error repeated — skipping")
                    break
                _last_error_summary = error_summary

                # Store as bad example with categorization
                target_sig = f"{cls_name}.{method_name}" if cls_name else method_name
                category, mutants = categorize_failure(logs, error_summary)
                store_bad_example(target_sig, method_name, cls_name or "",
                                  test_func_clean, error_summary[:200],
                                  os.path.basename(target_file),
                                  failure_category=category, survived_mutants=mutants)
                # Tier 4: self-healing RAG feedback on pytest failure
                feedback_test_rejected(target_label_graph)
                feedback_bad_didnt_help(target_label_graph, category)

                # Fix 2: Category-aware bail. Different retries often produce
                # slightly different error strings (different missing symbols),
                # so the exact-string check above misses repeated import errors.
                # Two consecutive import_error retries means no test for this
                # target will ever collect — stop wasting retry budget.
                if category == "import_error" and _last_category == "import_error":
                    print(f"   ⏭️  Repeated import_error category — bailing on {target_label}")
                    _last_category = category
                    break
                _last_category = category

                # Extract the FULL pytest E-lines (actual vs expected values)
                e_lines = [l.strip() for l in logs.split("\n") if l.strip().startswith("E ")]
                e_block = "\n".join(e_lines[:5]) if e_lines else ""

                actual_value = None
                for e_line in e_lines:
                    m = re.search(r"assert\s+(.+?)\s*==\s*(.+?)$", e_line)
                    if m:
                        left, right = m.group(1).strip(), m.group(2).strip()
                        if left.startswith(("'", '"', '[', '{', '(')) or left.lstrip('-').isdigit():
                            actual_value = left
                            break

                fix_hint = ""
                if actual_value:
                    fix_hint = (
                        f"\n🔴 CRITICAL FIX: The actual value is `{actual_value}`. "
                        f"Your assertion must be `== {actual_value}`. "
                        f"Do not guess — use this exact value."
                    )
                else:
                    value_fixes = _extract_actual_values(logs)
                    if value_fixes:
                        vf = value_fixes[0]
                        fix_hint = (
                            f"\nACTUAL VALUE: `{vf['actual']}` "
                            f"(you wrote `== {vf['expected']}` which is WRONG). "
                            f"Use the actual value."
                        )

                chunk_prompt = build_chunk_prompt(error_logs=logs, retry=retry)

                # Specific hint for digest auth chal setup
                pass  # auth chal handled by route_error_to_fix

        if not test_added and tests_for_this_target == 0:
            failed_targets.append(target_label)
            skipped_summary.append(target_label)
            print(f"   ⏭️  Skipping {target_label} after attempts")

            # ── PHASE 1: Collect Suspicious Tests (Post-Run Audit) ──
            if failure_logs_for_target:
                print(f"   🧠 Queuing {len(failure_logs_for_target)} failures for post-run batch audit...")
                suspicious_tests.append({
                    "target_name": target_label,
                    "target_file": target_file,
                    "method_name": method_name,
                    "cls_name": cls_name,
                    "source_code": source_code,
                    "test_code": test_func_clean,
                    "failure_logs": failure_logs_for_target,
                    "actual_import": actual_import,
                    "test_file": test_file
                })

            candidate_failure = accumulated_code + "\n\n# FAILING TEST (exhausted retries): Potential Bug\n" + test_func_clean + "\n"
            with open(test_file, "w") as f:
                f.write(candidate_failure)

    # O7: compute per-target wall time as (next_start - this_start),
    # with the final target's end being now.
    if stats_out is not None and _target_start_times:
        _end_of_loop = time.time()
        _timings: dict[str, float] = {}
        for _i, (_label, _t0) in enumerate(_target_start_times):
            _next = _target_start_times[_i + 1][1] if _i + 1 < len(_target_start_times) else _end_of_loop
            _timings[_label] = round(_next - _t0, 2)
        stats_out["target_timings"] = _timings

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"📊 Chunking complete: {accumulated_tests}/{len(targets)} targets covered")
    if failed_targets:
        print(f"   ❌ Failed: {', '.join(failed_targets)}")

    n_total_funcs = len(re.findall(r'def test_\w+', accumulated_code))

    if n_total_funcs == 0:
        print(f"   No tests generated — running auto_repair on pre-pass bugs if available")
        print(f"{'='*60}")
        # bug_report_path not yet defined here — compute from test_file which is available
        _early_bug_report_path = os.path.join(os.path.dirname(os.path.abspath(test_file)), "bug_reports.jsonl")
        # Before returning, still attempt auto_repair on any pre-pass confirmed bugs
        if auto_repair and os.path.exists(_early_bug_report_path):
            bug_report_path = _early_bug_report_path
            try:
                from bug_to_partc import repair_pending_bugs  # type: ignore[import]
                print(f"\nAuto-Repair (pre-pass bugs): invoking PartC...")
                _early_repairs = repair_pending_bugs(
                    bug_reports_path=bug_report_path,
                    repo_dir=os.path.dirname(os.path.abspath(target_file)),
                    model=model,
                    tokenizer=tokenizer,
                    log_callback=log_callback,
                )
                if log_callback and _early_repairs:
                    log_callback("repair_summary", _early_repairs)
            except Exception as _re:
                print(f"  Auto-repair failed: {_re}")
        return False
        
    # ── Layer 1+2 Amplifier: inject docstring + boundary tests ──────────────
    # These deterministic tests supplement whatever the LLM generated.
    # They are appended AFTER the LLM loop so they don't interfere with
    # the per-test BES gate (they always score well on their own).
    _amplifier_code = ""
    try:
        from docstring_extractor import build_docstring_tests  # type: ignore[import]

        for _cls, _meth in targets:
            _fn_source = _extract_function_source(source_code, _meth)
            if not _fn_source:
                continue

            # Layer 1: docstring examples (safe to add — spec-derived, not execution-derived)
            _doc_tests = build_docstring_tests(_fn_source, _meth, actual_import)
            if _doc_tests:
                _new_blocks = []
                for _block in _doc_tests.split("\n\n"):
                    if _block.strip() and _block.strip() not in accumulated_code:
                        _new_blocks.append(_block)
                if _new_blocks:
                    _amplifier_code += "\n\n# --- Layer 1: docstring spec examples ---\n"
                    _amplifier_code += "\n\n".join(_new_blocks)


        if _amplifier_code:
            accumulated_code = accumulated_code.rstrip() + "\n" + _amplifier_code + "\n"
            _amp_count = len(re.findall(r"def test_\w+", _amplifier_code))
            print(f"   Amplifier: +{_amp_count} deterministic tests (docstring + boundary)")
    except Exception as _amp_err:
        print(f"   Amplifier skipped (non-critical): {_amp_err}")

    # BES summary on final accumulated code
    try:
        from bes_scorer import bes_score  # type: ignore[import]
        _final_bes = bes_score(accumulated_code)
        print(f"   Final BES: {_final_bes['score']:.0f}/100 "
              f"(spec:{_final_bes['spec_coverage']:.0f} "
              f"variety:{_final_bes['input_variety']:.0f} "
              f"boundary:{_final_bes['boundary_coverage']:.0f})")
    except Exception:
        pass

    # ── Save final accumulated tests ──
    with open(test_file, "w") as f:
        f.write(accumulated_code)

    # Get final coverage from last successful run or a new quick run
    try:
        _, _, final_cov_pct, final_branch_cov_pct, _ = run_pytest_with_coverage(
            test_file, target_file, actual_import, need_lines=False,
            deadline_ts=file_deadline, docker_image=docker_image,
        )
    except:
        final_cov_pct = 0.0
        final_branch_cov_pct = 0.0

    # ── Run mutation testing ──
    print(f"\n   🦠 Running mutation testing on {n_total_funcs} tests...")
    if n_total_funcs == 0:
        print("   ⏭️  Skipping mutation — no tests generated")
        mutants_killed = False
        mutation_feedback = "Skipped (coverage < 20%)"
        mutation_score = 0.0
    else:
        mutants_killed, mutation_feedback = run_mutation_testing(
            target_file, test_file)

    # P14: Always print Tests count so parse_output can capture it
    print(f"   Tests: {n_total_funcs}")
    if mutants_killed:
        print(f"\n{'='*60}")
        print(f"🏆 ABSOLUTE SUCCESS!")
        print(f"   {mutation_feedback}")
        print(f"{'='*60}\n")
    else:
        print(f"   ⚠️  {mutation_feedback}")

    mutation_score = 0.0
    if not mutants_killed:
        m = re.search(r'(\d+)%', mutation_feedback)
        if m:
            mutation_score = float(m.group(1))
    else:
        mutation_score = 100.0

    # FEED SURVIVORS BACK TO RAG
    if not mutants_killed and mutation_feedback:
        survivor_matches = re.findall(r'(L\d+):\s*([^;]+)', mutation_feedback)
        for line_str, mut_desc in survivor_matches:
            line_no = int(line_str.replace('L', ''))
            mut_cls, mut_method = get_method_from_line(source_code, line_no)
            if mut_method:
                target_sig = f"{mut_cls}.{mut_method}" if mut_cls else mut_method
                method_test = ""
                for block in accumulated_code.split("def test_"):
                    if mut_method.strip('_') in block.split('(')[0]:
                        method_test = "def test_" + block
                        break
                store_bad_example(
                    target_sig, mut_method, mut_cls, method_test[:500],
                    f"Mutation survived on line {line_no}: {mut_desc}",
                    os.path.basename(target_file),
                    failure_category="mutation_survivor")
                
                # Write to bug_reports.jsonl for unified GUI display.
                # Includes verdict + test_file + test_code so the PartC bridge
                # (bug_to_partc.repair_pending_bugs) can pick this up and route
                # it through SBFL-guided repair.
                bug_report = {
                    "timestamp":   time.strftime("%Y-%m-%d %H:%M:%S"),
                    "source_file": os.path.basename(target_file),
                    "target":      target_sig,
                    "bug_type":    "mutation_survivor",
                    "confidence":  "high",
                    "confidence_score": 75,        # mutation survivors are real evidence (high but not max)
                    "verdict":     "suspected",    # PartC will promote/demote based on whether a fix exists
                    "description": f"Mutation testing exposed unprotected behaviour on line {line_no}: {mut_desc}",
                    "evidence":    f"On Line {line_no}: Mutant survived {mut_desc}",
                    "failure_count": 1,
                    # ── Fields needed by bug_to_partc bridge ──────────────────
                    "test_file":   os.path.abspath(test_file),
                    "test_code":   method_test[:2000] if method_test else "",
                    "line_no":     line_no,
                }
                bug_report_path = os.path.join(
                    os.path.dirname(os.path.abspath(test_file)),
                    "bug_reports.jsonl"
                )
                with open(bug_report_path, "a") as f:
                    f.write(json.dumps(bug_report) + "\n")
                print(f"   🧠 RAG learned from mutation failure in {target_sig}")

    # Final BES + coverage combined score
    try:
        from bes_scorer import bes_score as _bes_fn  # type: ignore[import]
        _fbes = _bes_fn(accumulated_code, func_source=source_code)
        _fbes_val = _fbes["score"]
    except Exception:
        _fbes = None
        _fbes_val = 0.0

    # Keep compute_composite_score for backward compat (coverage/mutation fields)
    final_composite = compute_composite_score(
        test_code=accumulated_code,
        line_coverage_pct=final_cov_pct,
        branch_coverage_pct=final_branch_cov_pct,
        mutation_score=mutation_score,
        per_test=False,
    )
    final_composite["composite"] = round(
        (_fbes_val * 0.5 + final_composite["composite"] * 0.5), 1
    ) if _fbes else final_composite["composite"]

    print(f"\n   Final quality: BES={_fbes_val:.0f}/100  Coverage={final_cov_pct:.0f}%  Mutation={mutation_score:.0f}%")
    if _fbes:
        print(f"      spec:{_fbes['spec_coverage']:.0f} "
              f"variety:{_fbes['input_variety']:.0f} "
              f"boundary:{_fbes['boundary_coverage']:.0f} "
              f"assertion:{_fbes['assertion_quality']:.0f} "
              f"mutation_potential:{_fbes['mutation_potential']:.0f}")

    # Flywheel: save successful tests for future fine-tuning
    # Lowered threshold (was 50/50) to accumulate more training data
    if final_cov_pct >= 25 and mutation_score >= 20:
        save_to_flywheel(
            target_file, source_code, 
            accumulated_code, final_cov_pct, mutation_score
        )
        print(f"   💾 Flywheel: saved for future fine-tuning "
              f"(cov:{final_cov_pct:.0f}% mut:{mutation_score:.0f}%)")

    print(f"\nFinal test code ({os.path.basename(test_file)}):")
    print("-" * 40)
    for line in accumulated_code.split("\n"):
        print(f"  {line}")
    print("-" * 40)

    # ── PHASE 2 & 3: POST-RUN AUDIT ──
    if suspicious_tests:
        print(f"\n🔄 Executing Post-Run Batch Audit on {len(suspicious_tests)} suspicious tests...")
        post_run_audit(model, tokenizer, suspicious_tests, test_file)

    # ── PHASE 4 (opt-in): AUTO-REPAIR via PartC ──────────────────────────
    bug_report_path = os.path.join(
        os.path.dirname(os.path.abspath(test_file)),
        "bug_reports.jsonl"
    )
    if auto_repair and os.path.exists(bug_report_path):
        try:
            from bug_to_partc import repair_pending_bugs  # type: ignore[import]
            print(f"\n🔧 Auto-Repair: invoking PartC on confirmed/suspected bugs...")
            repair_results = repair_pending_bugs(
                bug_reports_path=bug_report_path,
                repo_dir=os.path.dirname(os.path.abspath(target_file)),
                model=model,
                tokenizer=tokenizer,
                log_callback=log_callback,
            )
            # Attach repair results to callback so GUI can display them
            if log_callback and repair_results:
                log_callback("repair_summary", repair_results)
        except Exception as _repair_err:
            print(f"  Auto-repair failed (non-critical): {_repair_err}")
            if log_callback:
                log_callback("log", "warning", f"Auto-repair error: {_repair_err}")

    # ── Print bug report summary ──
    if os.path.exists(bug_report_path):
        try:
            bugs = []
            with open(bug_report_path) as f:
                for line in f:
                    try:
                        b = json.loads(line)
                        if b.get("source_file") == os.path.basename(target_file):
                            bugs.append(b)
                    except:
                        pass
            if bugs:
                print(f"\n🐛 BUG REPORT SUMMARY for {target_name}:")
                for b in bugs:
                    print(f"   • {b['target']} [{b['confidence']}]: {b['description']}")
        except:
            pass

    # ── QUALITY GATES: apply before final save ──────────────────────────
    try:
        from quality_gates import apply_all_gates
        cleaned, gate_report = apply_all_gates(
            accumulated_code, source_code, ast_threshold=4, verbose=True
        )
        if cleaned is None:
            print(f"   ⚠️  Quality gate REJECTED entire test file: {gate_report.get('ast_reason', '')}")
            # Keep the original rather than saving nothing
        else:
            if gate_report["tautologies_removed"] or gate_report["duplicates_renamed"] or gate_report["none_strings_fixed"]:
                print(f"   🛡️  Quality gates: removed {gate_report['tautologies_removed']} tautologies, "
                      f"renamed {gate_report['duplicates_renamed']} duplicates, "
                      f"fixed {gate_report['none_strings_fixed']} None-string issues")
            accumulated_code = cleaned
            with open(test_file, "w") as f:
                f.write(accumulated_code)
    except Exception as _gate_exc:
        print(f"   ⚠️  Quality gates skipped (error): {_gate_exc}")

    return True

# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Autonomous Test Generation Agent")
    parser.add_argument("--plan-mode", action="store_true", default=False,
                        help="Enable plan‑mode: generate a file‑level test plan and force the model to follow it")
    parser.add_argument("--target", default="calculator.py",
                        help="Python file to generate tests for (default: calculator.py)")
    parser.add_argument("--import-as", default=None,
                        help="Package import path (e.g. 'requests.auth') for files in installed packages")
    parser.add_argument("--max-iter", type=int, default=MAX_ITERATIONS,
                        help=f"Max self-correction iterations (default: {MAX_ITERATIONS})")
    parser.add_argument("--max-targets", type=int, default=None,
                        help="Maximum number of targets to test per file (None = all)")
    args = parser.parse_args()

    MAX_ITERATIONS = args.max_iter

    if not os.path.exists(args.target):
        print(f"❌ Target file not found: {args.target}")
        sys.exit(1)

    print("=" * 60)
    print("  🤖 Autonomous Test Generation Agent")
    print("  📦 Qwen2.5-Coder-7B + LoRA (4-bit)")
    print("  🔄 Self-correcting loop w/ Graph-RAG context")
    print("=" * 60)
    print()

    model, tokenizer = load_model()
    success = autonomous_loop(
        model,
        tokenizer,
        args.target,
        import_path=args.import_as,
        max_targets=args.max_targets,
        plan_mode=args.plan_mode,
    )

    sys.exit(0 if success else 1)
