from datasets import load_dataset

def load_swebench():
    dataset = load_dataset("SWE-bench/SWE-bench", split="test") #
    return dataset

def get_repos(dataset):
    # Using a set for unique names and faster O(1) lookup
    repos = {item["repo"] for item in dataset}
    return list(repos)