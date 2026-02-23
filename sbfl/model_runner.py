"""
model_runner.py - FINAL WORKING VERSION
Uses the already-working inference.py as backend
"""

import sys
import os

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import the WORKING inference module
from inference import load_model, repair_code

# Global model (load once)
_model = None
_tokenizer = None


def load_testmate_model():
    """
    Load model using the WORKING inference.py method
    """
    global _model, _tokenizer
    
    if _model is not None:
        return _model, _tokenizer
    
    print("📦 Loading TestMate using inference.py...")
    _model, _tokenizer = load_model()
    print("✅ Model loaded successfully")
    
    return _model, _tokenizer


def run_testmate(prompt: str) -> str:
    """
    Run TestMate model
    
    Converts SBFL-style prompt to inference.py format
    """
    try:
        model, tokenizer = load_testmate_model()
        
        # Extract issue and code from prompt
        # The prompt from control_loop looks like:
        # "ISSUE: Fix the bug in the code
        #  Test Failures: ...
        #  SBFL Analysis: ...
        #  Suspicious Code Section: ...
        #  Full Code for Context: ..."
        
        # Simple extraction
        if "ISSUE:" in prompt:
            lines = prompt.split('\n')
            issue_line = [l for l in lines if l.startswith("ISSUE:")][0]
            issue = issue_line.replace("ISSUE:", "").strip()
        else:
            issue = "Fix the bug in the code"
        
        # Extract code section
        if "Full Code for Context:" in prompt:
            code = prompt.split("Full Code for Context:")[1].strip()
        elif "Suspicious Code Section:" in prompt:
            code = prompt.split("Suspicious Code Section:")[1].split("Full Code")[0].strip()
        else:
            # Fallback: use the whole prompt as code
            code = prompt
        
        print(f"📝 Extracted issue: {issue[:50]}...")
        print(f"📝 Code length: {len(code)} chars")
        
        # Call the WORKING repair_code function
        fixed_code = repair_code(issue, code, model, tokenizer)
        
        print(f"✅ Generated fix: {len(fixed_code)} chars")
        
        return fixed_code
        
    except Exception as e:
        import traceback
        error_msg = f"Error: {e}"
        print(f"❌ {error_msg}")
        traceback.print_exc()
        return f"# {error_msg}"


def run_llamacpp(prompt: str) -> str:
    """Backward compatibility alias"""
    return run_testmate(prompt)


# ═══════════════════════════════════════════════════════════
# Test
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("="*70)
    print("🧪 Testing model_runner.py (using inference.py)")
    print("="*70)
    
    # Test 1: Simple format
    print("\n📝 Test 1: Simple prompt")
    test_prompt_1 = """
ISSUE: Fix operator bug

Suspicious Code:
def divide(a, b):
    if b == 0:
        return None
    return a * b
"""
    
    result_1 = run_testmate(test_prompt_1)
    print(f"\n🔧 Result 1:")
    print(result_1)
    
    # Test 2: Full format (as from control_loop)
    print("\n" + "="*70)
    print("\n📝 Test 2: Full SBFL format")
    test_prompt_2 = """ISSUE: Fix the bug in the code
    
Test Failures:
FAILED test_divide - AssertionError: assert 50 == 5

SBFL Analysis identified suspicious lines: Line 8

Suspicious Code Section:
  6: def divide(a, b):
  7:     if b == 0:
  8:         return a * b  # ⚠️

Full Code for Context:
  1: def add(a, b):
  2:     return a + b
  3: 
  4: def divide(a, b):
  5:     if b == 0:
  6:         return None
  7:     return a * b
  8: 
  9: def max_in_list(numbers):
 10:     if not numbers:
 11:         return None
 12:     max_val = numbers[0]
 13:     for num in numbers:
 14:         if num > max_val:
 15:             max_val = num
 16:     return max_val

Provide the complete fixed code for the entire file.
"""
    
    result_2 = run_testmate(test_prompt_2)
    print(f"\n🔧 Result 2:")
    print(result_2[:200] + "..." if len(result_2) > 200 else result_2)
    
    print("\n" + "="*70)
    if "Error" not in result_1 and "Error" not in result_2:
        print("✅ All tests passed!")
    else:
        print("⚠️ Some tests failed - check output above")