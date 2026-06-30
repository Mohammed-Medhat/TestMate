import json, os, re

data = json.load(open('PartC/humaneval_eval_dataset.json', encoding='utf-8'))
os.makedirs('samples/humaneval_demo', exist_ok=True)

# Create __init__.py so it registers as a Python package
open('samples/humaneval_demo/__init__.py', 'w').close()

picks = [
    (0,  'HumanEvalFix_000'),
    (2,  'HumanEvalFix_002'),
    (3,  'HumanEvalFix_003'),
    (30, 'HumanEvalFix_030'),
    (53, 'HumanEvalFix_053'),
]

for i, hid in picks:
    item = data[i]
    raw  = item['buggy_code']
    desc = item['description'].strip()

    # Take only the last function block (the actual buggy impl)
    parts = re.split(r'(?=\ndef )', raw)
    body = parts[-1].strip()

    lines = [
        '"""',
        f'HumanEvalFix Benchmark — {hid}',
        '=' * 60,
        'Problem Description:',
    ]
    for d in desc.split('\n'):
        lines.append(d)
    lines += [
        '=' * 60,
        'NOTE: This file contains an intentional bug for TestMate evaluation.',
        '"""',
        '',
        'from typing import List',
        '',
    ]
    lines.append(body)
    lines.append('')

    full = '\n'.join(lines)
    path = f'samples/humaneval_demo/{hid}.py'
    with open(path, 'w', encoding='utf-8') as f:
        f.write(full)
    print(f'{hid}: {len(full.splitlines())} lines')
