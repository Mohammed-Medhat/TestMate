
import json
import os
from pathlib import Path

def analyze_file(filename):
    print('='*60)
    print(f'ANALYZING DOCUMENT: {filename}')
    print('='*60)
    
    file_path = os.path.join(r"E:\Omar uni\Semester 7\GP\TestMate\srs2testcases\results", filename)
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        total = len(data)
        reqs = sum(1 for x in data if x['label'] == 1)
        non_reqs = sum(1 for x in data if x['label'] == 0)
        unlabeled = sum(1 for x in data if x['label'] == -1)
        
        matches = {
            'exact': sum(1 for x in data if x.get('match_type') == 'exact'),
            'fuzzy': sum(1 for x in data if x.get('match_type') == 'fuzzy'),
            'unmatched': sum(1 for x in data if x.get('match_type') == 'unmatched')
        }
        
        print(f"Total Sentences Extracted: {total}")
        print(f"Requirements Identified:   {reqs} ({reqs/total*100:.1f}%)")
        print(f"Non-Requirements:          {non_reqs} ({non_reqs/total*100:.1f}%)")
        print(f"Unlabeled Sentences:       {unlabeled}")
        
        print("\nMatch Quality Breakdown:")
        print(f"  Exact Matches: {matches['exact']} (100% confidence)")
        print(f"  Fuzzy Matches: {matches['fuzzy']} (>85% confidence)")
        print(f"  Unmatched:     {matches['unmatched']}")
        
        high_conf = [x for x in data if x['label'] == 1 and x.get('match_type') in ['exact', 'fuzzy']]
        if high_conf:
            print("\nSample Extracted Requirement:")
            sample = high_conf[0]
            print(f"  Text:  \"{sample['text'][:100]}{'...' if len(sample['text']) > 100 else ''}\"")
            print(f"  Type:  {sample.get('match_type').upper()} (Score: {sample.get('score', 0):.2f})")
            
        print("\n")
            
    except Exception as e:
        print(f"Error processing {filename}: {e}\n")

files = [
    '0000 - cctns_requirements.json',
    '0000 - gamma j_requirements.json',
    '0000 - inventory_requirements.json'
]

for f in files:
    analyze_file(f)
