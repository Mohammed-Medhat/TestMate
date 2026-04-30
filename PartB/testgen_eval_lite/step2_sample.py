import random
import json
from datasets import load_dataset

random.seed(42)

dataset = load_dataset("kjain14/testgenevallite", split="test")

small = []    # files under 50 lines
medium = []   # files 50-150 lines  
large = []    # files over 150 lines

for i, sample in enumerate(dataset):
    source = sample.get("code_src") or ""
    lines = len(source.splitlines())
    
    entry = {"index": i, "lines": lines, "sample": sample}
    
    if lines < 50:
        small.append(entry)
    elif lines < 150:
        medium.append(entry)
    else:
        large.append(entry)

print(f"Small: {len(small)}, Medium: {len(medium)}, Large: {len(large)}")

# Take ALL small and medium, fill remaining from large to reach 50
TARGET = 50
selected_small  = small[:]  # take all
selected_medium = medium[:]  # take all
remaining_needed = TARGET - len(selected_small) - len(selected_medium)
n_large = min(remaining_needed, len(large))
selected_large = random.sample(large, n_large)

selected = selected_small + selected_medium + selected_large

print(f"\nSelected {len(selected)} files:")
for s in selected:
    print(f"  Index {s['index']}: {s['lines']} lines")

with open("selected_samples.json", "w") as f:
    json.dump([s["index"] for s in selected], f)

print(f"\nSaved {len(selected)} indices to selected_samples.json")
