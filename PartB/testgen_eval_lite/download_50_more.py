"""
Download 50 additional files from TestGenEval-Lite dataset.
Skips files already present in testgen_eval_files/.
"""

import os
import json

EVAL_DIR = "testgen_eval_files"

# Files already downloaded (dataset indices from existing filenames)
existing_indices = set()
for fname in os.listdir(EVAL_DIR):
    if fname.endswith(".py") and not fname.startswith("test_") and "_" in fname:
        try:
            idx = int(fname.split("_")[0])
            existing_indices.add(idx)
        except ValueError:
            pass

print(f"Already have {len(existing_indices)} files: {sorted(existing_indices)}")

# Load dataset
print("Loading TestGenEval-Lite from HuggingFace...")
from datasets import load_dataset
dataset = load_dataset("kjain14/testgenevallite", split="test")
print(f"Dataset has {len(dataset)} total samples")

# Pick 50 new files, prioritizing Django files (better success rate)
# then other repos, skipping very large files
candidates = []
for i, sample in enumerate(dataset):
    if i in existing_indices:
        continue
    lines = len(sample.get("code_src", "").splitlines())
    repo = sample.get("repo", "")
    # Priority: 0=Django, 1=other supported, 2=unsupported
    if "django" in repo:
        priority = 0
    elif repo in ("sympy/sympy", "mwaskom/seaborn", "sphinx-doc/sphinx",
                   "psf/requests", "matplotlib/matplotlib"):
        priority = 1
    else:
        priority = 2
    candidates.append((priority, lines, i, sample))

# Sort by priority then by line count (smaller first = faster to process)
candidates.sort(key=lambda x: (x[0], x[1]))

# Take first 50
selected = candidates[:50]

print(f"\nDownloading {len(selected)} new files:")
for priority, lines, idx, sample in selected:
    code_src = sample.get("code_src", "")
    code_file = sample.get("code_file", "")
    repo = sample.get("repo", "")
    version = sample.get("version", "")
    basename = os.path.basename(code_file) if code_file else f"sample_{idx}.py"
    safe_name = f"{idx}_{basename}"
    filepath = os.path.join(EVAL_DIR, safe_name)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(code_src)

    print(f"  ✅ {safe_name} ({lines} lines) — {repo} v{version}")

# Save metadata for step4_runner to use
metadata = []
for priority, lines, idx, sample in selected:
    metadata.append({
        "dataset_id": idx,
        "instance_id": sample.get("instance_id", ""),
        "repo": sample.get("repo", ""),
        "version": sample.get("version", ""),
        "code_file": sample.get("code_file", ""),
        "lines": lines,
    })

meta_file = os.path.join(EVAL_DIR, "new_50_metadata.json")
with open(meta_file, "w") as f:
    json.dump(metadata, f, indent=2)

print(f"\n✅ Done! {len(selected)} new files saved to {EVAL_DIR}/")
print(f"   Metadata saved to {meta_file}")
print(f"   Total files now: {len(existing_indices) + len(selected)}")
