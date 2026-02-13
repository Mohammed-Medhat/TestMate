"""
Improved control_loop.py - Better integration with TestMate
"""

import subprocess
import json
from pathlib import Path

from sbfl_localiser import rank_suspicious_lines
from run_and_collect import collect_spectrum
from model_runner import run_testmate  # Updated to use TestMate


# -------------------- Paths --------------------
PROJECT_ROOT = Path(__file__).parent
BUGGY_FILE = PROJECT_ROOT / "buggy_code.py"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

ARTIFACTS_DIR.mkdir(exist_ok=True)


# -------------------- Helpers --------------------
def run_tests():
    """Run all tests and return status + output"""
    result = subprocess.run(
        ["pytest", "-v"],  # Verbose for better output
        capture_output=True,
        text=True
    )
    return result.returncode == 0, result.stdout + result.stderr


def extract_snippet(suspicious_lines, window=3):
    """Extract code around suspicious lines with line numbers"""
    code = BUGGY_FILE.read_text(encoding="utf-8").splitlines()
    
    # Collect all lines to include
    lines_to_include = set()
    for ln in suspicious_lines:
        for offset in range(-window, window + 1):
            idx = ln - 1 + offset  # Convert to 0-indexed
            if 0 <= idx < len(code):
                lines_to_include.add(idx)
    
    # Build snippet with line numbers
    snippet_lines = []
    for idx in sorted(lines_to_include):
        marker = " ⚠️" if (idx + 1) in suspicious_lines else ""
        snippet_lines.append(f"{idx + 1:3d}: {code[idx]}{marker}")
    
    return '\n'.join(snippet_lines)


def extract_full_code():
    """Get full code with line numbers"""
    code = BUGGY_FILE.read_text(encoding="utf-8").splitlines()
    return '\n'.join(f"{i+1:3d}: {line}" for i, line in enumerate(code))


def save_ranking(ranked):
    """Save SBFL ranking to JSON"""
    data = [
        {"line": ln, "score": float(score)}  # Ensure float for JSON
        for ln, score in ranked
    ]
    
    path = ARTIFACTS_DIR / "sbfl_ranking.json"
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"📊 SBFL ranking saved to {path}")


def save_model_output(code: str, attempt: int):
    """Save model output"""
    path = ARTIFACTS_DIR / f"model_output_attempt_{attempt}.py"
    path.write_text(code, encoding="utf-8")
    print(f"📝 Model output saved to {path}")


def apply_patch(new_code: str) -> bool:
    """
    Apply the fixed code
    
    Accepts:
    1. Full file code
    2. Just the fixed function (extracts and replaces)
    """
    
    # Check if it looks like valid Python
    if "def " not in new_code:
        print("⚠️ Patch rejected: does not contain function definition")
        return False
    
    # Remove common markdown artifacts
    new_code = new_code.replace("```python", "").replace("```", "")
    new_code = new_code.strip()
    
    # If it's a complete file, use it directly
    if new_code.count("def ") >= 2:
        BUGGY_FILE.write_text(new_code, encoding="utf-8")
        print("✅ Applied full file patch")
        return True
    
    # If it's a single function, try to replace it in the file
    # This is more complex - for now, just accept full files
    print("⚠️ Partial patch - need full file. Trying to use as-is...")
    BUGGY_FILE.write_text(new_code, encoding="utf-8")
    return True


def build_repair_prompt(fail_output: str, suspicious_lines: list, snippet: str, full_code: str) -> str:
    """
    Build a well-formatted prompt for TestMate
    
    Format matches training data:
    ISSUE: [description]
    CODE: [suspicious code]
    """
    
    # Extract key failure info
    failure_summary = []
    for line in fail_output.split('\n'):
        if 'FAILED' in line or 'assert' in line or 'Error' in line:
            failure_summary.append(line.strip())
    
    failure_text = '\n'.join(failure_summary[:5])  # Top 5 lines
    
    # Format suspicious lines
    susp_lines_text = ', '.join([f"Line {ln}" for ln in suspicious_lines[:3]])
    
    # Build prompt
    prompt = f"""ISSUE: Fix the bug in the code
    
Test Failures:
{failure_text}

SBFL Analysis identified suspicious lines: {susp_lines_text}

Suspicious Code Section:
{snippet}

Full Code for Context:
{full_code}

Provide the complete fixed code for the entire file.
"""
    
    return prompt


# -------------------- Main Loop --------------------
def repair_loop(max_attempts=3, top_k=5):
    """
    Main repair loop with SBFL-guided TestMate
    
    Args:
        max_attempts: Maximum number of repair iterations
        top_k: Number of top suspicious lines to focus on
    """
    
    print("="*70)
    print("🚀 TestMate + SBFL Repair Loop")
    print("="*70)
    
    for attempt in range(1, max_attempts + 1):
        print(f"\n{'='*70}")
        print(f"🔄 Repair Attempt {attempt}/{max_attempts}")
        print(f"{'='*70}")
        
        # Step 1: Run tests
        print("\n📋 Step 1: Running tests...")
        passed, test_output = run_tests()
        
        if passed:
            print("\n" + "="*70)
            print("✅ SUCCESS! All tests passed!")
            print("="*70)
            return True
        
        print(f"❌ Tests failed. Proceeding with repair...\n")
        
        # Step 2: SBFL Analysis
        print("📊 Step 2: Running SBFL analysis...")
        spectrum, fail_output = collect_spectrum()
        ranked = rank_suspicious_lines(spectrum)
        
        if not ranked:
            print("⚠️ No suspicious lines found. Exiting.")
            return False
        
        save_ranking(ranked)
        
        # Step 3: Display suspicious lines
        print(f"\n🔍 Step 3: Top {top_k} suspicious lines:")
        for i, (ln, score) in enumerate(ranked[:top_k], 1):
            print(f"   {i}. Line {ln:3d}: suspiciousness = {score:.4f}")
        
        suspicious_lines = [ln for ln, _ in ranked[:top_k]]
        
        # Step 4: Extract code
        print("\n📝 Step 4: Extracting code snippets...")
        snippet = extract_snippet(suspicious_lines, window=3)
        full_code = extract_full_code()
        
        print(f"Suspicious code snippet ({len(snippet.split(chr(10)))} lines):")
        print("-" * 70)
        print(snippet)
        print("-" * 70)
        
        # Step 5: Build prompt
        print("\n🧠 Step 5: Building repair prompt...")
        prompt = build_repair_prompt(fail_output, suspicious_lines, snippet, full_code)
        
        # Step 6: Call TestMate
        print("🤖 Step 6: Calling TestMate model...")
        try:
            new_code = run_testmate(prompt)
            print(f"Generated {len(new_code)} characters of code")
        except Exception as e:
            print(f"❌ Error calling model: {e}")
            return False
        
        save_model_output(new_code, attempt)
        
        # Step 7: Apply patch
        print("\n🔧 Step 7: Applying patch...")
        if not apply_patch(new_code):
            print("❌ Patch application failed. Stopping.")
            return False
        
        print("✅ Patch applied successfully")
    
    # Failed after max attempts
    print("\n" + "="*70)
    print(f"❌ FAILED: Max attempts ({max_attempts}) reached")
    print("="*70)
    return False


# -------------------- Entry Point --------------------
if __name__ == "__main__":
    success = repair_loop(max_attempts=3, top_k=5)
    
    if success:
        print("\n🎉 Repair completed successfully!")
        exit(0)
    else:
        print("\n⚠️ Repair unsuccessful")
        exit(1)