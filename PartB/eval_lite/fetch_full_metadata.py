"""
Fetch original path metadata for ALL eval files.

Merges:
  - new_50_metadata.json  (the "new 50" files, already have metadata)
  - selected_samples.json (the original 50 IDs, fetch from HF)

Run locally (internet required):
    cd PartB/eval_lite
    python fetch_full_metadata.py

Output: testgen_eval_files/full_metadata.json
Then commit the output and push — Kaggle reads it from the repo.
"""
import json
from pathlib import Path

HERE = Path(__file__).parent
EVAL_DIR = HERE / "testgen_eval_files"
SELECTED_JSON = HERE / "selected_samples.json"
NEW_META_JSON = EVAL_DIR / "new_50_metadata.json"
OUT_JSON = EVAL_DIR / "full_metadata.json"


def main() -> None:
    # Load existing new_50_metadata.json (already covers the "new 50")
    existing_meta: dict[int, dict] = {}
    if NEW_META_JSON.exists():
        for entry in json.loads(NEW_META_JSON.read_text(encoding="utf-8")):
            ds_id = entry.get("dataset_id")
            if ds_id is not None:
                existing_meta[int(ds_id)] = entry
        print(f"Loaded {len(existing_meta)} entries from new_50_metadata.json")

    # Load original 50 IDs
    original_ids: list[int] = json.loads(SELECTED_JSON.read_text(encoding="utf-8"))
    print(f"Found {len(original_ids)} original IDs")

    # Which original IDs are NOT already covered by existing metadata?
    missing_ids = [i for i in original_ids if i not in existing_meta]
    print(f"Need to fetch HF metadata for {len(missing_ids)} files")

    fetched_meta: dict[int, dict] = {}
    if missing_ids:
        print("Loading kjain14/testgenevallite from HuggingFace...")
        from datasets import load_dataset  # noqa: PLC0415
        dataset = load_dataset("kjain14/testgenevallite", split="test")
        print(f"Dataset has {len(dataset)} total samples")

        for idx in sorted(missing_ids):
            sample = dataset[idx]
            code_src = sample.get("code_src", "")
            code_file = sample.get("code_file", "")
            fetched_meta[idx] = {
                "dataset_id": idx,
                "instance_id": sample.get("instance_id", ""),
                "repo": sample.get("repo", ""),
                "version": sample.get("version", ""),
                "code_file": code_file,
                "lines": len(code_src.splitlines()),
            }
            print(f"  [{idx:>3}] {code_file}")

    # Merge: existing (new 50) + freshly fetched (original 50)
    all_meta = {**existing_meta, **fetched_meta}
    out = sorted(all_meta.values(), key=lambda x: x["dataset_id"])

    OUT_JSON.write_text(json.dumps(out, indent=2), encoding="utf-8")

    with_path = sum(1 for e in out if e.get("code_file"))
    print(f"\nDone: {len(out)} entries, {with_path} with code_file -> {OUT_JSON}")
    print("Next: git add testgen_eval_files/full_metadata.json && git push")


if __name__ == "__main__":
    main()
