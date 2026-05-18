import os
import json
from datasets import load_dataset

dataset = load_dataset("kjain14/testgenevallite", split="test")

with open("selected_samples.json") as f:
    indices = json.load(f)

os.makedirs("testgen_eval_files", exist_ok=True)

for idx in indices:
    sample = dataset[idx]
    
    source_code = sample.get("code_src") or ""
    filename    = sample.get("code_file") or f"file_{idx}.py"
    
    basename = os.path.basename(filename)
    if not basename.endswith(".py"):
        basename = f"sample_{idx}.py"
    else:
        basename = f"{idx}_{basename}"
    
    filepath = os.path.join("testgen_eval_files", basename)
    
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(source_code)
    
    print(f"Saved: {filepath} ({len(source_code.splitlines())} lines)")
