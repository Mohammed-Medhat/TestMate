import subprocess, sys, json, os, tempfile
from pathlib import Path

src  = Path(r'D:\TestMate\TestMate\samples\humaneval_demo\HumanEvalFix_053.py')
work = src.parent

test_code = f"""
import sys
sys.path.insert(0, r'{work}')
from HumanEvalFix_053 import add

def test_add_catches_bug():
    # Bug: returns x+y+y+x instead of x+y; 2+3+3+2=10, not 5
    assert add(2, 3) == 5
"""

with tempfile.NamedTemporaryFile(suffix='.py', mode='w', dir=str(work), delete=False, prefix='test_sanity_') as f:
    f.write(test_code)
    tpath = f.name

cov_out = work / '_sanity_cov.json'
env = os.environ.copy()
env['PYTHONPATH'] = str(work) + os.pathsep + env.get('PYTHONPATH', '')

r = subprocess.run(
    [sys.executable, '-m', 'pytest', tpath,
     f'--cov={src.stem}',
     f'--cov-report=json:{cov_out}',
     '-q', '--no-header', '--tb=short'],
    capture_output=True, text=True, cwd=str(work), env=env, timeout=30
)
print('Return code:', r.returncode)
print(r.stdout[-600:])

if cov_out.exists():
    d = json.loads(cov_out.read_text())
    for fname, finfo in d.get('files', {}).items():
        if 'HumanEvalFix_053' in fname:
            s = finfo.get('summary', {})
            pct = s.get('percent_covered', 0)
            stmts = s.get('num_statements', 0)
            print(f'Coverage: {pct:.1f}%  ({stmts} statements)')
    cov_out.unlink()
else:
    print('No coverage JSON generated')
    print('stderr:', r.stderr[-400:])

os.unlink(tpath)
