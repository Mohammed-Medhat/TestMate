import json
import os
import re
from datasets import load_dataset

def build_humaneval_dataset():
    out_file = "humaneval_eval_dataset.json"
    
    # مسح أي نسخة قديمة
    if os.path.exists(out_file):
        os.remove(out_file)
        print(f"🗑️  Deleted old {out_file}")

    print("⏳ Downloading HumanEvalPack (Python)...")
    dataset = load_dataset("bigcode/humanevalpack", "python", split="test")
    
    formatted_dataset = []
    
    for i, item in enumerate(dataset):
        entry_point = item['entry_point']
        test_setup = item.get('test_setup', "")
        test_code = item['test']
        
        # ── FIX 1: بناء ملف التست بشكل صحيح ──────────────────────
        # الـ check function بتكون موجودة في test_code من HumanEval
        # نحتاج نحطها جوه test file ونخلي pytest يلاقيها
        
        # إزالة أي استدعاء قديم لـ check في نهاية الكود
        clean_test_body = re.sub(rf"\s*check\s*\(\s*{entry_point}\s*\)\s*$", "", test_code, flags=re.MULTILINE).strip()
        
        # ── FIX 2: بناء الـ test file الصحيح ─────────────────────
        # نحط كل حاجة جوه دالة واحدة test_* عشان pytest يشوفها
        full_test_file = f"""from solution import {entry_point}

{test_setup}

# ═══════════════════════════════════════════════════════════════
# Test function from HumanEval (contains check function + assertions)
# ═══════════════════════════════════════════════════════════════
{clean_test_body}

# ═══════════════════════════════════════════════════════════════
# Pytest-compatible test wrapper
# ═══════════════════════════════════════════════════════════════
def test_{entry_point}_correctness():
    \"\"\"
    Pytest will recognize this function and run the check() assertions.
    If check() raises an assertion error, pytest will mark this as FAILED.
    If check() completes without errors, pytest will mark this as PASSED.
    \"\"\"
    check({entry_point})
"""
        
        sample = {
            "id": f"HumanEvalFix_{i:03d}",
            "bug_type": "HumanEval",
            "description": item.get('docstring', 'Fix the python code').strip(),
            "buggy_code": item['prompt'] + item['declaration'] + item['buggy_solution'],
            "tests": full_test_file,
            "entry_point": entry_point  # مفيد للـ debugging
        }
        formatted_dataset.append(sample)
    
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(formatted_dataset, f, indent=4, ensure_ascii=False)
        
    print(f"✅ Successfully formatted {len(formatted_dataset)} samples to {out_file}.")
    
    # عرض مثال للتحقق
    print("\n" + "═" * 70)
    print("🔍 Sample test structure (first entry):")
    print("═" * 70)
    sample_lines = formatted_dataset[0]['tests'].splitlines()
    for i, line in enumerate(sample_lines[:25], 1):  # أول 25 سطر
        print(f"{i:3}: {line}")
    if len(sample_lines) > 25:
        print(f"     ... ({len(sample_lines) - 25} more lines)")
    print("═" * 70 + "\n")

if __name__ == "__main__":
    build_humaneval_dataset()