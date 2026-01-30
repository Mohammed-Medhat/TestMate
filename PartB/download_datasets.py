# from datasets import load_dataset

# # Load the dataset
# ds = load_dataset("ise-uiuc/Magicoder-OSS-Instruct-75K", split='train')

# # Filter for Python only (Assuming 'lang' field exists, otherwise we check the code block)
# def is_python(example):
#     # Some datasets have a 'lang' column; others require checking the 'code' column
#     return example.get('lang') == 'python' or 'def ' in example.get('solution', '')

# python_ds = ds.filter(is_python)
# print(f"Downloaded {len(python_ds)} Python-specific samples.")

# # Save locally to avoid re-downloading
# python_ds.save_to_disk("data/magicoder_python")
# from datasets import load_dataset

# # Load the full training set
# swe_train = load_dataset("princeton-nlp/SWE-bench", split="train")

# # Save locally
# swe_train.save_to_disk("data/swe_bench_train")
# print(f"Downloaded {len(swe_train)} SWE-bench training instances.")
# from datasets import load_dataset

# # Load the Lite version for cost-effective, high-quality testing
# swe_lite = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")

# swe_lite.save_to_disk("data/swe_bench_lite")
# print(f"Ready to test on {len(swe_lite)} Lite instances.")

# from datasets import load_from_disk

# # Replace 'data/magicoder_python' with your actual folder name
# dataset = load_from_disk("data/magicoder_python")

# # See the structure
# print(dataset)

# # See the first 3 rows (Instructions and Solutions)
# for i in range(3):
#     print(f"--- Sample {i} ---")
#     print(f"Instruction: {dataset[i]['problem']}")
#     print(f"Code Solution:\n{dataset[i]['solution']}\n")

from datasets import load_from_disk

# Load your local SWE-bench Lite data
swe_lite = load_from_disk("data/swe_bench_lite")

# Print the keys to confirm (they are different from Magicoder!)
print(f"Available Keys: {swe_lite.features.keys()}")

# View the first sample
for i in range(1):
    print(f"\n{'='*20} SAMPLE {i} {'='*20}")
    print(f"INSTANCE ID: {swe_lite[i]['instance_id']}")
    print(f"REPO: {swe_lite[i]['repo']}")
    
    # The 'Requirement' - What the user reported on GitHub
    print(f"\nISSUE (Requirement):\n{swe_lite[i]['problem_statement'][:500]}...") 
    
    # The 'Golden Patch' - The human solution you are trying to match
    print(f"\nHUMAN FIX (Patch):\n{swe_lite[i]['patch']}")
    
    # The 'Tests' - The code used to verify if the fix works
    print(f"\nTESTS:\n{swe_lite[i]['test_patch']}")