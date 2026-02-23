"""
Advanced control_loop.py - End-to-End APR Pipeline
Features: AST-based safe patching & JSON Results Filtering
"""

import subprocess
import json
import ast
import os
import re
from pathlib import Path

from sbfl_localiser import rank_suspicious_lines
from run_and_collect import collect_spectrum
from model_runner import run_testmate


# -------------------- Paths --------------------
PROJECT_ROOT = Path(__file__).parent
BUGGY_FILE = PROJECT_ROOT / "buggy_code.py"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
RESULTS_FILE = PROJECT_ROOT / "local_run_results.json"

ARTIFACTS_DIR.mkdir(exist_ok=True)


# -------------------- Phase 1: Filtering --------------------
def load_and_filter_candidates(json_path):
    """Filter test results to find valid repair candidates (Logic Errors Only)."""
    path = Path(json_path)
    if not path.exists():
        print(f"⚠️ Warning: Pipeline results file not found at {json_path}")
        return []

    with open(path, 'r', encoding='utf-8') as f:
        try:
            results = json.load(f)
        except json.JSONDecodeError:
            print("❌ Error: Invalid JSON format in results file.")
            return []

    valid_candidates = []
    print(f"\n🔍 Analyzing pipeline test results...")

    for entry in results:
        instance_id = entry.get('instance_id', 'unknown')
        metrics = entry.get('metrics', {})
        error_msg = metrics.get('execution_error', '')
        
        # 1. Skip if test passed
        if metrics.get('execution_success', False):
            continue

        # 2. Skip if the test generation itself is bad (Syntax, Missing args, etc.)
        is_bad_test = False
        bad_test_indicators = ["SyntaxError", "IndentationError", "missing", "NameError", "ImportError"]
        
        for indicator in bad_test_indicators:
            if indicator in error_msg:
                print(f"⏭️  Skipping {instance_id}: Bad Test Generation ({indicator})")
                is_bad_test = True
                break
        
        if is_bad_test:
            continue

        # 3. Accept if it's a logic failure (AssertionError)
        if "AssertionError" in error_msg or "Error" in error_msg:
            print(f"✅ Queued {instance_id}: Logic failure detected.")
            valid_candidates.append({
                "id": instance_id,
                "problem": entry.get('problem', ''),
                "failed_test": entry.get('generated_test', ''),
                "error": error_msg
            })

    print(f"📊 Found {len(valid_candidates)} candidates ready for APR.\n")
    return valid_candidates


# -------------------- Phase 2: Execution & Extraction --------------------
def run_tests():
    """Run tests and return status + output"""
    result = subprocess.run(
        ["pytest", "-v"], 
        capture_output=True,
        text=True
    )
    return result.returncode == 0, result.stdout + result.stderr

def extract_snippet(suspicious_lines, window=5):
    """
    Extract clean code snippet WITHOUT line numbers or markers.
    The model needs pure Python code to understand and fix it correctly.
    """
    code = BUGGY_FILE.read_text(encoding="utf-8").splitlines()
    lines_to_include = set()
    for ln in suspicious_lines:
        for offset in range(-window, window + 1):
            idx = ln - 1 + offset
            if 0 <= idx < len(code):
                lines_to_include.add(idx)
    
    snippet_lines = [code[idx] for idx in sorted(lines_to_include)]
    return '\n'.join(snippet_lines)

def extract_full_code():
    """Get full code with line numbers"""
    code = BUGGY_FILE.read_text(encoding="utf-8").splitlines()
    return '\n'.join(f"{i+1:3d}: {line}" for i, line in enumerate(code))


# -------------------- Phase 3: AST Smart Patching --------------------
def apply_patch(new_code: str) -> bool:
    """
    Safely apply the patch using Abstract Syntax Trees (AST).
    Includes robust extraction to ignore LLM conversational text.
    """
    try:
        # 1. Robustly extract ONLY the Python code from the LLM output
        match = re.search(r'```(?:python)?(.*?)```', new_code, re.DOTALL | re.IGNORECASE)
        if match:
            clean_code = match.group(1)
        else:
            lines = new_code.split('\n')
            start_idx = 0
            for i, line in enumerate(lines):
                if line.strip().startswith('def ') or line.strip().startswith('class ') or line.strip().startswith('@'):
                    start_idx = i
                    break
            clean_code = '\n'.join(lines[start_idx:])
            
        clean_code = clean_code.replace("⚠️", "").replace("⚠", "").strip()
        
        # 2. Parse the LLM's generated code
        new_ast = ast.parse(clean_code)
        new_functions = {node.name: node for node in new_ast.body if isinstance(node, ast.FunctionDef)}
        
        if not new_functions:
            print("⚠️ Patch rejected: No function definitions found in model output.")
            return False

        # 3. Parse the original buggy file
        original_code = BUGGY_FILE.read_text(encoding="utf-8")
        original_ast = ast.parse(original_code)
        
        # 4. Replace ONLY the modified functions
        modified = False
        for i, node in enumerate(original_ast.body):
            if isinstance(node, ast.FunctionDef) and node.name in new_functions:
                original_ast.body[i] = new_functions[node.name]
                modified = True
                print(f"🔧 AST Engine: Successfully replaced function '{node.name}'")

        if not modified:
            print("⚠️ Patch rejected: Function names from model didn't match the original file.")
            return False

        # 5. Write the safely modified AST back to the file
        fixed_source = ast.unparse(original_ast)
        BUGGY_FILE.write_text(fixed_source, encoding="utf-8")
        print("✅ AST Patch applied without breaking syntax.")
        return True

    except SyntaxError as e:
        print(f"❌ Patch rejected: Model generated invalid Python syntax -> {e}")
        print(f"--- Model Output was ---\n{new_code}\n------------------------")
        return False
    except Exception as e:
        print(f"❌ AST Patching failed: {e}")
        return False


# -------------------- Phase 4: Prompting & Saving --------------------
def build_repair_prompt(fail_output: str, suspicious_lines: list, snippet: str, full_code: str) -> str:
    failure_summary = [line.strip() for line in fail_output.split('\n') if 'E   ' in line or 'FAILED' in line or 'Error' in line]
    trace_text = '\n'.join(failure_summary[:4])
    
    if not trace_text:
        trace_text = "AssertionError: Logic verification failed."
        
    clean_code = BUGGY_FILE.read_text(encoding="utf-8")
    
    prompt = f"ISSUE: Fix the logic bug causing the test failure.\n\nTRACE:\n{trace_text}\n\nBUGGY:\n{clean_code}"
    
    return prompt
def save_model_output(code: str, attempt: int):
    path = ARTIFACTS_DIR / f"model_output_attempt_{attempt}.py"
    path.write_text(code, encoding="utf-8")


# -------------------- Phase 5: The Main Repair Loop --------------------
def repair_loop(max_attempts=3, top_k=5):
    print("="*70)
    print("🚀 TestMate + SBFL + AST Repair Loop")
    print("="*70)
    
    for attempt in range(1, max_attempts + 1):
        print(f"\n🔄 Repair Attempt {attempt}/{max_attempts}")
        
        # Step 1: Run tests
        passed, test_output = run_tests()
        if passed:
            print("\n" + "="*70)
            print("🎉 SUCCESS! All tests passed! Bug is fixed.")
            print("="*70)
            return True
        
        print(f"❌ Tests failed. Proceeding with SBFL analysis...")
        
        # Step 2: SBFL Analysis
        spectrum, fail_output = collect_spectrum()
        ranked = rank_suspicious_lines(spectrum)
        
        if not ranked:
            print("⚠️ No suspicious lines found by SBFL. Exiting.")
            return False
        
        suspicious_lines = [ln for ln, _ in ranked[:top_k]]
        print(f"🔍 Top suspicious lines: {suspicious_lines}")
        
        # Step 3: Extract & Build Prompt
        snippet = extract_snippet(suspicious_lines, window=1)
        full_code = extract_full_code()
        prompt = build_repair_prompt(fail_output, suspicious_lines, snippet, full_code)
        
        # Step 4: Call Model
        print("🤖 Calling TestMate model...")
        try:
            new_code = run_testmate(prompt)
        except Exception as e:
            print(f"❌ Error calling model: {e}")
            return False
        
        save_model_output(new_code, attempt)
        
        # Step 5: AST Smart Patching
        print("🔧 Applying smart patch...")
        if not apply_patch(new_code):
            print("❌ Patch application failed. Trying next attempt if available.")
            continue
            
    print("\n" + "="*70)
    print(f"❌ FAILED: Max attempts ({max_attempts}) reached without fixing the bug.")
    print("="*70)
    return False


# -------------------- Entry Point --------------------
if __name__ == "__main__":
    # Optional: Read from pipeline results first
    # candidates = load_and_filter_candidates(RESULTS_FILE)
    
    # Run the core repair loop
    success = repair_loop(max_attempts=3, top_k=5)
    
    if success:
        exit(0)
    else:
        exit(1)