"""
Seed the TestMate RAG database with Django's own test suite.
Provides high-quality human-written examples for Django auth,
admin, validators, and model patterns.

Usage:
    cd D:\TestMate\TestMate\PartB\testgen
    python seed_rag_django.py --django-path D:\django

To get Django source:
    git clone https://github.com/django/django.git D:\django
"""

import os, sys, ast, re, json, argparse

def convert_to_pytest_style(test_code: str) -> str:
    """Convert unittest-style assertions to pytest style."""
    import re
    test_code = re.sub(
        r'self\.assertEqual\((.+?),\s*(.+?)\)',
        r'assert \1 == \2', test_code)
    test_code = re.sub(
        r'self\.assertNotEqual\((.+?),\s*(.+?)\)',
        r'assert \1 != \2', test_code)
    test_code = re.sub(
        r'self\.assertTrue\((.+?)\)',
        r'assert \1', test_code)
    test_code = re.sub(
        r'self\.assertFalse\((.+?)\)',
        r'assert not \1', test_code)
    test_code = re.sub(
        r'self\.assertIsNone\((.+?)\)',
        r'assert \1 is None', test_code)
    test_code = re.sub(
        r'self\.assertIsNotNone\((.+?)\)',
        r'assert \1 is not None', test_code)
    test_code = re.sub(
        r'self\.assertIn\((.+?),\s*(.+?)\)',
        r'assert \1 in \2', test_code)
    test_code = re.sub(
        r'self\.assertNotIn\((.+?),\s*(.+?)\)',
        r'assert \1 not in \2', test_code)
    test_code = re.sub(
        r'self\.assertRaises\((.+?)\)',
        r'pytest.raises(\1)', test_code)
    # Remove class wrapper — extract just the function body
    # Strip leading self parameter from test methods
    test_code = re.sub(
        r'def (test_\w+)\(self\)',
        r'def \1()', test_code)
    return test_code
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rag_store import init_db, store_example

DJANGO_TEST_FILES = [
    "tests/auth_tests/test_tokens.py",
    "tests/auth_tests/test_validators.py",
    "tests/auth_tests/test_models.py",
    "tests/admin_checks/tests.py",
    "tests/model_fields/tests.py",
    "tests/utils_tests/test_numberformat.py",
]

def extract_expected_outputs(test_code: str) -> dict:
    outputs = {}
    for line in test_code.splitlines():
        line = line.strip()
        m = re.search(r'assert\s+(.+?)\s*==\s*(.+)$', line)
        if m:
            call = m.group(1).strip()
            val = m.group(2).strip()
            if len(call) < 100 and len(val) < 100:
                outputs[call] = val
    return outputs

def parse_test_file(filepath: str) -> list:
    try:
        with open(filepath, encoding="utf-8", errors="ignore") as f:
            source = f.read()
        tree = ast.parse(source)
    except (SyntaxError, OSError):
        return []

    lines = source.splitlines()
    results = []

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for item in node.body:
                if (isinstance(item, (ast.FunctionDef,
                        ast.AsyncFunctionDef))
                        and item.name.startswith('test_')):
                    test_src = "\n".join(
                        lines[item.lineno - 1:item.end_lineno]
                    )
                    if len(test_src) < 50:
                        continue
                    if ('assert' not in test_src
                            and 'raises' not in test_src
                            and 'self.assert' not in test_src):
                        continue
                    raw = item.name.replace('test_', '', 1)
                    parts = raw.split('_')
                    method_guess = raw
                    for i in range(len(parts), 0, -1):
                        candidate = '_'.join(parts[:i])
                        if len(candidate) > 3:
                            method_guess = candidate
                            break
                    expected = extract_expected_outputs(test_src)
                    results.append({
                        "class_name": node.name,
                        "test_name": item.name,
                        "method_guess": method_guess,
                        "test_code": test_src,
                        "expected_outputs": expected,
                        "source_file": os.path.basename(filepath),
                    })

        elif (isinstance(node, (ast.FunctionDef,
                ast.AsyncFunctionDef))
                and node.name.startswith('test_')):
            test_src = "\n".join(
                lines[node.lineno - 1:node.end_lineno]
            )
            if len(test_src) < 50:
                continue
            if ('assert' not in test_src
                    and 'raises' not in test_src
                    and 'self.assert' not in test_src):
                continue
            raw = node.name.replace('test_', '', 1)
            parts = raw.split('_')
            method_guess = raw
            for i in range(len(parts), 0, -1):
                candidate = '_'.join(parts[:i])
                if len(candidate) > 3:
                    method_guess = candidate
                    break
            expected = extract_expected_outputs(test_src)
            results.append({
                "class_name": "",
                "test_name": node.name,
                "method_guess": method_guess,
                "test_code": test_src,
                "expected_outputs": expected,
                "source_file": os.path.basename(filepath),
            })

    return results

def seed_from_django(django_path: str, verbose: bool = True):
    init_db()
    total = 0
    skipped = 0

    for rel_path in DJANGO_TEST_FILES:
        full_path = os.path.join(django_path, rel_path)
        if not os.path.exists(full_path):
            if verbose:
                print(f"⚠️  Not found: {full_path}")
            continue
        if verbose:
            print(f"\n📂 Processing: {rel_path}")

        tests = parse_test_file(full_path)

        for t in tests:
            assert_count = len(
                re.findall(
                    r'\bassert\b|self\.assert\w+|self\.fail',
                    t["test_code"])
            )
            if assert_count == 0:
                skipped += 1
                continue

            # Convert unittest style to pytest style
            t["test_code"] = convert_to_pytest_style(
                t["test_code"]
            )

            quality = 85.0
            if len(t["expected_outputs"]) >= 2:
                quality = 92.0
            if len(t["expected_outputs"]) >= 4:
                quality = 95.0

            coverage_pattern = (
                f"human_written,django_suite,"
                f"expected:{json.dumps(t['expected_outputs'])[:200]}"
            )

            store_example(
                target_signature=t["method_guess"],
                method_name=t["method_guess"],
                class_name=t["class_name"],
                test_code=t["test_code"],
                quality_score=quality,
                coverage_lines=10,
                passed_mutation=False,
                source_file=f"django_seed:{t['source_file']}",
                coverage_pattern=coverage_pattern
            )
            total += 1

            if verbose:
                print(
                    f"   ✅ {t['class_name']}.{t['method_guess']}"
                    f" (quality={quality:.0f}, "
                    f"expected={len(t['expected_outputs'])})"
                )

    print(f"\n{'='*50}")
    print(f"✅ Seeded {total} tests into RAG")
    print(f"⏭️  Skipped {skipped} trivial tests")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--django-path", required=True)
    parser.add_argument("--verbose", action="store_true",
                        default=True)
    args = parser.parse_args()

    if not os.path.exists(args.django_path):
        print(f"❌ Not found: {args.django_path}")
        sys.exit(1)

    seed_from_django(args.django_path, args.verbose)
