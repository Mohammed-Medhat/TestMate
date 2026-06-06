"""
Decontaminate MBPP against HumanEval.

The clean retrain trains on MBPP and tests on HumanEval (+ realistic sets).
MBPP and HumanEval are distinct benchmarks, but a handful of problems are
near-duplicates. This script flags and removes those so the training set has
zero overlap with the HumanEval test set.

Method: token-set (Jaccard) similarity between each MBPP problem
(text + code) and every HumanEval problem (prompt + canonical_solution).
Any MBPP problem whose best match exceeds the threshold is dropped.

Output: mbpp_clean_ids.json — the list of MBPP task_ids safe to train on,
plus a small report of what was removed and why.

All stdout is ASCII-only (Windows consoles default to cp1252 and crash on
non-Latin-1 characters).

Usage:
    python decontaminate_mbpp.py                 # default threshold 0.6
    python decontaminate_mbpp.py --threshold 0.5 # stricter
"""
import argparse
import json
import re
from pathlib import Path

from datasets import load_dataset

_OUT = Path(__file__).parent / "mbpp_clean_ids.json"

_TOKEN_RE = re.compile(r"[a-zA-Z_][a-zA-Z_0-9]*")
# Generic tokens that appear in almost every problem — ignoring them keeps the
# similarity signal on the meaningful identifiers, not boilerplate.
_STOP = {
    "def", "return", "for", "in", "if", "else", "elif", "while", "the", "a",
    "an", "of", "to", "and", "or", "is", "are", "be", "function", "write",
    "python", "given", "list", "number", "numbers", "string", "value", "true",
    "false", "none", "self", "import", "from", "assert", "print", "range",
    "len", "int", "str", "float", "bool",
}


def _tokens(text: str) -> set:
    toks = {t.lower() for t in _TOKEN_RE.findall(text or "")}
    return toks - _STOP


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def main() -> int:
    ap = argparse.ArgumentParser(description="Decontaminate MBPP vs HumanEval.")
    # 0.35 catches genuine concept overlaps (gcd, roman numerals, min/max of
    # two) that have different wording but are the same task as a HumanEval
    # problem. Cheap insurance: ~4/974 dropped for a clean contamination claim.
    ap.add_argument("--threshold", type=float, default=0.35,
                    help="Jaccard above which an MBPP problem is dropped.")
    args = ap.parse_args()

    print("Loading HumanEval (test, 164)...")
    he = load_dataset("openai_humaneval")["test"]
    he_tok = [
        _tokens(f"{r['prompt']}\n{r['canonical_solution']}")
        for r in he
    ]
    print(f"  HumanEval problems: {len(he_tok)}")

    print("Loading MBPP (all splits)...")
    mb = load_dataset("mbpp", "full")
    # Train on train + test + validation + prompt (the whole thing is fair game
    # for training since we never EVAL on MBPP). task_id is unique across splits.
    rows = []
    for split in ("train", "test", "validation", "prompt"):
        if split in mb:
            rows.extend(list(mb[split]))
    print(f"  MBPP problems: {len(rows)}")

    kept, dropped = [], []
    for r in rows:
        tid = r["task_id"]
        tok = _tokens(f"{r['text']}\n{r['code']}")
        best = 0.0
        best_he = None
        for i, ht in enumerate(he_tok):
            s = _jaccard(tok, ht)
            if s > best:
                best, best_he = s, he[i]["task_id"]
        if best >= args.threshold:
            dropped.append({"task_id": tid, "jaccard": round(best, 3),
                            "matched_humaneval": best_he,
                            "text": (r["text"] or "")[:90]})
        else:
            kept.append(tid)

    report = {
        "threshold": args.threshold,
        "mbpp_total": len(rows),
        "kept": len(kept),
        "dropped": len(dropped),
        "clean_task_ids": sorted(kept),
        "dropped_detail": dropped,
    }
    _OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("")
    print(f"Threshold:    {args.threshold}")
    print(f"MBPP total:   {len(rows)}")
    print(f"Kept (clean): {len(kept)}")
    print(f"Dropped:      {len(dropped)}")
    if dropped:
        print("Removed (MBPP task_id -> HumanEval, jaccard):")
        for d in dropped[:20]:
            print(f"  {d['task_id']:>5} -> {d['matched_humaneval']:>12}  "
                  f"j={d['jaccard']}  | {d['text']}")
    print("")
    print(f"Wrote clean id list -> {_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
