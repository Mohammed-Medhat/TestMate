import sys

with open(r'D:\TestMate\TestMate\PartB\testgen\main.py', encoding='utf-8') as f:
    lines = f.readlines()

def p(q, c=5):
    for i, l in enumerate(lines):
        if q in l:
            out.write(f'--- MATCH {q.strip()} ---\n')
            for j in range(max(0, i-1), min(len(lines), i+c)):
                out.write(f'{j+1}: {lines[j]}')
            break

with open('temp_fix.txt', 'w', encoding='utf-8') as out:
    p('max_length=3072')
    p('test_mutants = mutants[:10]')
    p('mutants_killed, mutation_feedback = run_mutation_testing(', 8)
    p('test_added = False', 8)
    p('for retry in range(3):', 5)
    p('print(f\"   ❌ Failed: {error_summary[:120]}\")', 5)
    p('if depth > 8 or len(paths) > 20:', 5)
    p('if depth > 6 or len(paths) > 10:', 5)
