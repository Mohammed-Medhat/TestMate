# ============================================================
# TestMate — Local Hidden-Test Evaluator
# ============================================================
# Run AFTER downloading Kaggle artifacts.
# Computes TRUE repair rate by running HumanEval's gold tests
# on the repaired code produced by pipeline_eval_kaggle.py.
#
# Usage:
#   python pipeline_eval_local.py \
#       --dataset   PartC/humaneval_eval_dataset.json \
#       --artifacts /kaggle/working/pipeline_results/E4/artifacts \
#       --results   /kaggle/working/pipeline_results/E4/results.json \
#       --output    pipeline_results_E4_final.json
#
# Also handles E0/E1 (from apr_evaluator.py) by running the same
# hidden-test pass on their repaired_code artifacts.
# ============================================================

from __future__ import annotations

import sys
import re
import json
import shutil
import argparse
import tempfile
import subprocess
from pathlib import Path


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",   required=True, help="humaneval_eval_dataset.json path")
    p.add_argument("--artifacts", required=True, help="Dir with {id}_repaired.py files")
    p.add_argument("--results",   required=True, help="results.json from Kaggle run")
    p.add_argument("--output",    required=True, help="Output JSON with true repair metrics")
    return p.parse_args()


# ── Sandbox ───────────────────────────────────────────────────────────────────

def _run_gold_tests(repaired_code: str, gold_test_code: str) -> dict:
    """Run HumanEval gold tests against repaired code. Returns pass/fail dict."""
    tmpdir = tempfile.mkdtemp(prefix="tm_hidden_")
    try:
        (Path(tmpdir) / "solution.py").write_text(repaired_code, encoding="utf-8")
        (Path(tmpdir) / "test_solution.py").write_text(gold_test_code, encoding="utf-8")
        r = subprocess.run(
            [sys.executable, "-m", "pytest", "test_solution.py",
             "--tb=short", "--no-header", "-q"],
            capture_output=True, text=True, cwd=tmpdir, timeout=30,
        )
        out = r.stdout + r.stderr
        passed = int(m.group(1)) if (m := re.search(r"(\d+) passed", out)) else 0
        failed = int(m.group(1)) if (m := re.search(r"(\d+) failed", out)) else 0
        errors = int(m.group(1)) if (m := re.search(r"(\d+) error",  out)) else 0
        total  = passed + failed + errors
        return {
            "true_repaired": r.returncode == 0 and total > 0,
            "gold_passed":   passed,
            "gold_failed":   failed + errors,
            "gold_total":    total,
            "output":        out[-1000:],
        }
    except subprocess.TimeoutExpired:
        return {"true_repaired": False, "gold_passed": 0,
                "gold_failed": 0, "gold_total": 0, "output": "TIMEOUT"}
    except Exception as exc:
        return {"true_repaired": False, "gold_passed": 0,
                "gold_failed": 0, "gold_total": 0, "output": str(exc)}
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = _parse_args()

    with open(args.dataset, encoding="utf-8") as f:
        dataset = {s["id"]: s for s in json.load(f)}

    with open(args.results, encoding="utf-8") as f:
        kaggle_results = json.load(f)

    artifact_dir = Path(args.artifacts)

    final_results = []
    true_repaired_count = 0
    apparent_repaired_count = 0
    plausible_not_correct = 0
    no_artifact = 0

    for r in kaggle_results:
        sid = r["id"]
        sample = dataset.get(sid)
        if not sample:
            print(f"[warn] {sid}: not found in dataset, skipping")
            continue

        repaired_file = artifact_dir / f"{sid}_repaired.py"
        if not repaired_file.exists():
            print(f"[warn] {sid}: no repaired artifact found")
            no_artifact += 1
            r["true_repaired"] = False
            r["gold_passed"]   = 0
            r["gold_failed"]   = 0
            r["gold_total"]    = 0
            r["plausible_not_correct"] = False
            final_results.append(r)
            continue

        repaired_code = repaired_file.read_text(encoding="utf-8")
        gold          = sample["tests"]

        print(f"  [test] {sid} ...", end=" ")
        hidden = _run_gold_tests(repaired_code, gold)

        r["true_repaired"]          = hidden["true_repaired"]
        r["gold_passed"]            = hidden["gold_passed"]
        r["gold_failed"]            = hidden["gold_failed"]
        r["gold_total"]             = hidden["gold_total"]
        r["plausible_not_correct"]  = (r.get("apparent_repaired", False)
                                       and not hidden["true_repaired"])

        if r.get("apparent_repaired"):
            apparent_repaired_count += 1
        if hidden["true_repaired"]:
            true_repaired_count += 1
            print("[PASS] true repair")
        elif r.get("apparent_repaired"):
            plausible_not_correct += 1
            print("[WARN] plausible-not-correct")
        else:
            print("[FAIL]")

        final_results.append(r)

    n = len(final_results)
    summary = {
        "n_samples":              n,
        "apparent_repair_rate":   round(apparent_repaired_count / n * 100, 2) if n else 0,
        "true_repair_rate":       round(true_repaired_count     / n * 100, 2) if n else 0,
        "plausible_not_correct":  plausible_not_correct,
        "plausible_not_correct_pct": round(plausible_not_correct / n * 100, 2) if n else 0,
        "no_artifact_count":      no_artifact,
    }

    out = {"summary": summary, "per_sample": final_results}
    Path(args.output).write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("\n=== SUMMARY ===")
    for k, v in summary.items():
        print(f"  {k:<35} {v}")
    print(f"\nSaved -> {args.output}")


if __name__ == "__main__":
    main()
