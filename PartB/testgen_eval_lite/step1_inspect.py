import json
from datasets import load_dataset

def main():
    dataset = load_dataset("kjain14/testgenevallite", split="test")
    keys = list(dataset[0].keys())
    with open("schema.json", "w") as f:
        json.dump({"keys": keys}, f)

if __name__ == "__main__":
    main()
