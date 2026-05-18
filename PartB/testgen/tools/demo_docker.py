# ============================================================
# DOCKER DEMO: Run Generated Test on Real SWE-bench Instance
# ============================================================
# For Seminar: Shows test generation + actual execution
# ============================================================

import subprocess
import os
import json
import tempfile
from pathlib import Path

# Choose a SIMPLE SWE-bench instance for demo
# django__django-11001 is a good choice - simple bug, quick to set up

DEMO_INSTANCE = {
    "instance_id": "django__django-11001",
    "repo": "django/django",
    "base_commit": "e7fd69d051eaa67cb17f172a39b57253e9cb831a",
    "problem_statement": """
    Incorrect removal of order_by clause when using multiline RawSQL.
    
    When using RawSQL with a multiline string, Django incorrectly removes
    the ORDER BY clause. This happens because the regex doesn't account
    for newlines in the SQL string.
    """
}

# Your generated test (paste your model's output here)
GENERATED_TEST = '''
import pytest
from django.db.models.expressions import RawSQL
from django.test import TestCase

class TestRawSQLOrderBy(TestCase):
    """Test multiline RawSQL ORDER BY handling."""
    
    def test_multiline_rawsql_order_by(self):
        """Test that multiline RawSQL preserves ORDER BY clause."""
        # Arrange
        multiline_sql = """
            CASE
                WHEN status = 'active' THEN 1
                ELSE 2
            END
        """
        raw_sql = RawSQL(multiline_sql, [])
        
        # Act - This should not raise an error
        # Assert - The ORDER BY should be preserved
        assert raw_sql is not None
'''

def run_docker_demo():
    """Run the demo using Docker."""
    
    print("="*60)
    print("🐳 DOCKER DEMO: Test Generation + Execution")
    print("="*60)
    
    # Check Docker
    try:
        subprocess.run(["docker", "--version"], check=True, capture_output=True)
        print("✅ Docker is available")
    except:
        print("❌ Docker not found. Please install Docker Desktop.")
        return
    
    print(f"\n📋 Instance: {DEMO_INSTANCE['instance_id']}")
    print(f"📦 Repo: {DEMO_INSTANCE['repo']}")
    
    # Create temp directory for test
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save the test
        test_file = Path(tmpdir) / "test_generated.py"
        test_file.write_text(GENERATED_TEST)
        print(f"\n📝 Test saved to: {test_file}")
        
        # Run in Docker with Django
        print("\n🚀 Running in Docker...")
        
        docker_cmd = f"""
docker run --rm -v {tmpdir}:/tests python:3.10 bash -c "
    pip install django pytest -q &&
    cd /tests &&
    python -c 'import django; print(f\"Django version: {{django.__version__}}\")' &&
    pytest test_generated.py -v --tb=short
"
"""
        
        result = subprocess.run(
            docker_cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=300
        )
        
        print("\n📤 OUTPUT:")
        print(result.stdout)
        
        if result.returncode == 0:
            print("\n✅ TEST PASSED!")
        else:
            print("\n❌ TEST FAILED (expected for bug reproduction)")
            print(result.stderr[-500:] if result.stderr else "")

def simple_demo_without_docker():
    """Simpler demo without Docker - just show the workflow."""
    
    print("="*60)
    print("🎯 SEMINAR DEMO: Test Generation Workflow")
    print("="*60)
    
    print("\n📋 STEP 1: Bug Report (Input)")
    print("-"*40)
    print(DEMO_INSTANCE['problem_statement'])
    
    print("\n🤖 STEP 2: Model Generates Test")
    print("-"*40)
    print(GENERATED_TEST)
    
    print("\n✅ STEP 3: Test Analysis")
    print("-"*40)
    
    # Quick analysis
    import ast
    try:
        ast.parse(GENERATED_TEST)
        print("   Syntax: ✅ Valid")
    except:
        print("   Syntax: ❌ Invalid")
    
    if "assert" in GENERATED_TEST.lower():
        print("   Assertions: ✅ Present")
    
    if "def test_" in GENERATED_TEST:
        print("   Test Function: ✅ Present")
    
    if "django" in GENERATED_TEST.lower():
        print("   Relevance: ✅ References Django")
    
    print("\n🐳 STEP 4: Docker Execution (Live Demo)")
    print("-"*40)
    print("   Would run: docker run python:3.10 pytest test_generated.py")
    
    print("\n" + "="*60)
    print("🎉 Demo Complete!")
    print("="*60)

if __name__ == "__main__":
    import sys
    
    if "--docker" in sys.argv:
        run_docker_demo()
    else:
        simple_demo_without_docker()
        print("\n💡 Run with --docker to execute in Docker container")
