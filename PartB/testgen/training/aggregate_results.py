# ============================================================
# TestMate — Results Aggregator
# ============================================================
# Reads every eval JSON produced by the runbook and builds the
# paper tables. Handles THREE separate schemas:
#
#   1. Baseline/TestGen (C0, C7)
#      Dataset:  SWE-bench Lite (problem statements, static metrics only)
#      File:     baseline_detailed.json  OR  detailed_metrics.json
#      Format:   flat list of dicts with `syntax_valid`,
#                `compile_success`, `has_assertions`, `assertion_count`,
#                `follows_aaa_pattern`, `has_edge_cases`,
#                `relevance_score`, `cyclomatic_complexity`,
#                `generation_time_ms`, ...
#
#   2. Ablation (C1–C6)
#      Dataset:  PartB/eval_lite/testgen_eval_files/ (real source files)
#      File:     ablation_<variant>.json (+ _with_mut.json)
#      Format:   {"per_file": [...], "summary": {...}} with keys
#                `syntax_valid`, `tests_runnable`, `tests_collected`,
#                `tests_passed`, `pass_rate`, `line_coverage`,
#                `assertions_per_test`, `test_diversity`,
#                `wall_time_sec`, `mutation_score`
#
#   3. APR (E0, E1)
#      Dataset:  HumanEval-164
#      File:     eval_<timestamp>.json
#      Format:   {mode_name: {"results": [...], "metrics": {"overall": {...}}}}
#
#   4. Pipeline (E2–E5)
#      Dataset:  HumanEval-164
#      Files:    summary.json + pipeline_results_<COND>_final.json
#
# Important: C0/C7 and C1–C6 are NOT on the same dataset, so we
# emit them as TWO tables (2A static / 2B dynamic) — never one row.
#
# Usage:
#   python aggregate_results.py \
#       --partb_dir   paper_eval/partb_results \
#       --pipeline_dir paper_eval/pipeline_results \
#       --apr_dir     paper_eval/apr_results \
#       --output_dir  paper_eval/paper_tables
# ============================================================

from __future__ import annotations

import csv
import json
import argparse
from pathlib import Path
from typing import Optional


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--partb_dir",     default="paper_eval/partb_results",
                   help="Dir containing baseline/testgen/ablation JSONs")
    p.add_argument("--pipeline_dir",  default="paper_eval/pipeline_results",
                   help="Dir containing E2/, E3/, E4/, E5/ subdirs from Kaggle eval")
    p.add_argument("--apr_dir",       default="paper_eval/apr_results",
                   help="Dir containing E0_eval_*.json + E1_eval_*.json from PartC")
    p.add_argument("--output_dir",    default="paper_eval/paper_tables")
    return p.parse_args()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_json(path: Path) -> dict | list | None:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _mean(vals) -> float | None:
    vals = [v for v in vals if v is not None]
    return round(sum(vals) / len(vals), 2) if vals else None


def _pct_true(records: list, key: str) -> float | None:
    vals = [r.get(key) for r in records if r.get(key) is not None]
    if not vals:
        return None
    return round(sum(1 for v in vals if v) / len(vals) * 100, 1)


def _pretty(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.2f}"
    return str(v)


# ── Schema 1 (LEGACY): static-quality on SWE-bench Lite ──────────────────────
# Kept for backwards compatibility with old kaggle_eval_baseline.py / kaggle_eval_testgen.py.
# The current runbook DOES NOT use these — everything runs on eval_lite.

def _row_static(records: list, label: str, desc: str) -> dict:
    n = len(records)
    if n == 0:
        return {"#": label, "Condition": desc, "N": 0}
    row: dict = {
        "#":               label,
        "Condition":       desc,
        "N":               n,
        "Syntax %":        _pct_true(records, "syntax_valid"),
        "Compile %":       _pct_true(records, "compile_success"),
        "Has-assert %":    _pct_true(records, "has_assertions"),
        "AAA pattern %":   _pct_true(records, "follows_aaa_pattern"),
        "Edge cases %":    _pct_true(records, "has_edge_cases"),
        "Err-handling %":  _pct_true(records, "has_error_handling"),
        "Asserts / test":  _mean(r.get("assertion_count")     for r in records),
        "Cyclomatic":      _mean(r.get("cyclomatic_complexity") for r in records),
        "Relevance":       _mean(r.get("relevance_score")     for r in records),
        "Gen latency (ms)":_mean(r.get("generation_time_ms")  for r in records),
    }
    return row


def build_partb_static_table(partb_dir: Path) -> list[dict]:
    """Table 2A — C0 vs C7 on SWE-bench Lite (static quality metrics)."""
    rows = []

    # C0: baseline_evaluation/baseline_detailed.json
    c0 = _load_json(partb_dir / "baseline_evaluation" / "baseline_detailed.json")
    if c0 is None:
        c0 = _load_json(partb_dir / "baseline_detailed.json")
    if c0 is None:
        print("  [warn]C0 baseline_detailed.json not found")
    else:
        rows.append(_row_static(c0, "C0", "Qwen-7B base (no RAG, no LoRA)"))

    # C7: testgen_evaluation/detailed_metrics.json
    c7 = _load_json(partb_dir / "testgen_evaluation" / "detailed_metrics.json")
    if c7 is None:
        c7 = _load_json(partb_dir / "detailed_metrics.json")
    if c7 is None:
        print("  [warn]C7 detailed_metrics.json not found")
    else:
        rows.append(_row_static(c7, "C7", "TestMate (full RAG + LoRA + loop)"))

    return rows


# ── Schema 2: Ablation (C1–C6 on eval_lite) ───────────────────────────────────

def _row_ablation(per_file: list, label: str, desc: str) -> dict:
    valid = [e for e in per_file if "error" not in e]
    n = len(valid)
    if n == 0:
        return {"#": label, "Condition": desc, "N": 0}

    syn_ok    = sum(1 for e in valid if e.get("syntax_valid"))
    runnable  = sum(1 for e in valid if e.get("tests_runnable"))
    any1      = sum(1 for e in valid if e.get("any_pass_1"))
    all1      = sum(1 for e in valid if e.get("all_pass_1"))

    pass_rates  = [e.get("pass_rate", 0)        for e in valid]
    line_covs   = [e.get("line_coverage", 0)    for e in valid]
    api_covs    = [e.get("api_coverage", 0)     for e in valid]
    asserts     = [e.get("assertions_per_test", 0) for e in valid]
    walls       = [e.get("wall_time_sec", 0)    for e in valid]
    mut_scores  = [e.get("mutation_score")      for e in valid
                   if e.get("mutation_score") is not None]
    iterations  = [e.get("iterations", 1)       for e in valid]

    return {
        "#":                label,
        "Condition":        desc,
        "N":                n,
        "Syntax %":         round(syn_ok   / n * 100, 1),
        "Runnable %":       round(runnable / n * 100, 1),
        "Any-pass@1 %":     round(any1     / n * 100, 1),
        "All-pass@1 %":     round(all1     / n * 100, 1),
        "Mean pass-rate":   _mean(pass_rates),
        "Line coverage %":  _mean(line_covs),
        "API coverage":     _mean(api_covs),
        "Asserts / test":   _mean(asserts),
        "Mutation kill %":  (_mean(mut_scores) if mut_scores else None),
        "Mean iterations":  _mean(iterations),
        "Mean wall (s)":    _mean(walls),
    }


_ABLATION_FILES = [
    ("ablation_no_rag",          "Baseline",          "Qwen-7B (no RAG, no LoRA)"),
    ("ablation_graph_only",      "Graph Only",        "Knowledge graph RAG only"),
    ("ablation_no_graph",        "Vector Only",       "Semantic vector RAG only"),
    ("ablation_full",            "Full RAG",          "All 3 RAG layers"),
    ("ablation_full_with_loop",  "Full RAG + Loop",   "All RAG + self-correction"),
    ("ablation_full_with_plan",  "Full RAG + Plan",   "All RAG + plan-first"),
    ("ablation_testmate",        "TestMate",          "Full RAG + LoRA + Loop (PRODUCTION)"),
]


def build_partb_dynamic_table(partb_dir: Path) -> list[dict]:
    """Table 2B — C1–C6 ablation on eval_lite (pytest pass + coverage)."""
    rows = []
    for stem, label, desc in _ABLATION_FILES:
        # Try in order, preferring `_with_mut.json` (Phase 2 — has mutation_score).
        candidates = [
            partb_dir / "results" / f"{stem}_with_mut.json",
            partb_dir / "results" / f"{stem}.json",
            partb_dir / f"{stem}_with_mut.json",
            partb_dir / f"{stem}.json",
        ]
        data = None
        for cand in candidates:
            data = _load_json(cand)
            if data is not None:
                break

        if data is None:
            print(f"  [warn] {stem}.json not found in any expected location")
            continue
        per_file = data.get("per_file", [])
        rows.append(_row_ablation(per_file, label, desc))
    return rows


# ── Schema 3: APR (E0, E1) ────────────────────────────────────────────────────

def build_apr_table(apr_dir: Path) -> list[dict]:
    """Rows for E0 (base_paper) and E1 (finetuned_full) from PartC's APR output."""
    rows = []
    if not apr_dir.exists():
        print(f"  [warn]APR dir not found: {apr_dir}")
        return rows

    # Find the most recent eval_*.json files
    eval_files = sorted(apr_dir.glob("eval_*.json"), key=lambda p: p.stat().st_mtime)
    if not eval_files:
        # Try subdirectories (e.g. apr_results_E0/eval_*.json)
        eval_files = sorted(apr_dir.glob("**/eval_*.json"),
                            key=lambda p: p.stat().st_mtime)

    seen_modes: dict[str, dict] = {}
    for ef in eval_files:
        data = _load_json(ef)
        if not isinstance(data, dict):
            continue
        for mode_name, mode_data in data.items():
            if isinstance(mode_data, dict) and "metrics" in mode_data:
                # latest wins (eval_files is sorted by mtime ascending)
                seen_modes[mode_name] = mode_data["metrics"].get("overall", {})

    _MODE_INFO = {
        "base_paper":     ("APR Baseline",     "Base Qwen, gold tests"),
        "base_simple":    ("APR Base (simple)","Base Qwen, simple prompt"),
        "base_pipeline":  ("APR Base (rich)",  "Base Qwen, rich prompt"),
        "finetuned_zero": ("APR LoRA (zero-shot)","TestMate APR, pass@1"),
        "finetuned_full": ("APR LoRA (TestMate)", "TestMate APR, iterative repair"),
    }

    for mode, (label, desc) in _MODE_INFO.items():
        if mode not in seen_modes:
            continue
        m = seen_modes[mode]
        rate = m.get("repair_success_rate", 0)
        rows.append({
            "#":                       label,
            "Condition":               desc,
            "N":                       m.get("total_samples"),
            "Test source":             "HumanEval gold",
            "APR model":               "Base Qwen" if mode.startswith("base") else "TestMate LoRA",
            "Apparent repair %":       round(rate * 100, 2) if rate else None,
            "True repair %":           round(rate * 100, 2) if rate else None,
            "Plausible-not-correct %": 0.0,
            "Avg attempts":            m.get("avg_attempts"),
        })
    return rows


# ── Schema 4: Pipeline (E2–E5) ────────────────────────────────────────────────

_PIPELINE_DESC = {
    "E2": "Generated tests (no spec) + base APR",
    "E3": "Generated tests (no spec) + TestMate APR",
    "E4": "FULL PIPELINE: spec -> tests -> TestMate APR",
    "E5": "FULL PIPELINE + gap-fill (best mode)",
}

_PIPELINE_LABEL = {
    "E2": "B+C Naive",
    "E3": "B+C with LoRA",
    "E4": "Full Pipeline",
    "E5": "Pipeline + Gap-fill",
}


def build_pipeline_table(pipeline_dir: Path) -> list[dict]:
    rows = []
    for cond in ("E2", "E3", "E4", "E5"):
        summary_path = pipeline_dir / cond / "summary.json"
        final_paths  = [
            pipeline_dir / f"pipeline_results_{cond}_final.json",
            pipeline_dir / cond / "final.json",
            pipeline_dir / cond / f"pipeline_results_{cond}_final.json",
        ]

        summary = _load_json(summary_path)
        final   = None
        for fp in final_paths:
            final = _load_json(fp)
            if final:
                break

        if summary is None:
            print(f"  [warn]pipeline {cond}/summary.json not found")
            continue

        row: dict = {
            "#":                     _PIPELINE_LABEL[cond],
            "Condition":             _PIPELINE_DESC[cond],
            "N":                     summary.get("n_samples"),
            "Test source":           "TestMate PartB",
            "APR model":             "Base Qwen" if cond == "E2" else "TestMate LoRA",
            "Test detection %":      summary.get("test_detection_rate"),
            "Apparent repair %":     summary.get("apparent_repair_rate"),
            "Avg req coverage %":    summary.get("avg_req_coverage_pct"),
            "Avg gen time (s)":      summary.get("avg_gen_time_s"),
        }
        if isinstance(final, dict):
            fs = final.get("summary", {})
            row["True repair %"]            = fs.get("true_repair_rate")
            row["Plausible-not-correct %"]  = fs.get("plausible_not_correct_pct")
        else:
            row["True repair %"]            = "— (run pipeline_eval_local.py)"
            row["Plausible-not-correct %"]  = "—"
        rows.append(row)
    return rows


# ── Output ────────────────────────────────────────────────────────────────────

def _print_table(title: str, rows: list[dict], note: Optional[str] = None):
    if not rows:
        print(f"\n{title}\n  (no data)\n")
        return

    cols = list(rows[0].keys())
    widths = {c: max(len(str(c)),
                     max(len(_pretty(r.get(c))) for r in rows)) for c in cols}
    sep    = "+-" + "-+-".join("-" * widths[c] for c in cols) + "-+"
    header = "| " + " | ".join(str(c).ljust(widths[c]) for c in cols) + " |"

    print(f"\n{'='*72}\n{title}\n{'='*72}")
    if note:
        print(note)
    print(sep)
    print(header)
    print(sep)
    for row in rows:
        print("| " + " | ".join(_pretty(row.get(c)).ljust(widths[c]) for c in cols) + " |")
    print(sep)


def _save_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    cols = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: _pretty(r.get(c)) for c in cols})
    print(f"  Saved -> {path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = _parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    partb_dir    = Path(args.partb_dir)
    pipeline_dir = Path(args.pipeline_dir)
    apr_dir      = Path(args.apr_dir)

    print("Building tables...\n")

    partb_rows    = build_partb_dynamic_table(partb_dir)
    apr_rows      = build_apr_table(apr_dir)
    pipeline_rows = build_pipeline_table(pipeline_dir)

    _print_table(
        "Table 1 — Part B: Test Generation (vs Baseline + Ablation)",
        partb_rows,
        note=("Dataset: PartB/eval_lite/testgen_eval_files (102 real Python files). "
              "Compares Baseline vs every RAG component vs full TestMate."),
    )
    _print_table(
        "Table 2 — Part C: Automated Program Repair",
        apr_rows,
        note=("Dataset: HumanEval-164. APR runs against the gold HumanEval tests. "
              "Shows how much LoRA + iterative repair add on top of base Qwen."),
    )
    _print_table(
        "Table 3 — Full Pipeline (Spec -> Tests -> Repair)",
        pipeline_rows,
        note=("Dataset: HumanEval-164. Tests are generated by Part B from the spec; "
              "Part C repairs using those tests. True repair = gold-test pass."),
    )

    _save_csv(out_dir / "table1_partb.csv",     partb_rows)
    _save_csv(out_dir / "table2_apr.csv",       apr_rows)
    _save_csv(out_dir / "table3_pipeline.csv",  pipeline_rows)

    # Headline numbers
    print("\n" + "="*72)
    print("HEADLINE NUMBERS")
    print("="*72)
    if partb_rows:
        base = next((r for r in partb_rows if r["#"] == "Baseline"), {})
        tm   = next((r for r in partb_rows if r["#"] == "TestMate"), {})
        if base and tm:
            print(f"  Part B test quality (baseline -> TestMate):")
            print(f"    Any-pass@1 %    {_pretty(base.get('Any-pass@1 %'))}  ->  {_pretty(tm.get('Any-pass@1 %'))}")
            print(f"    All-pass@1 %    {_pretty(base.get('All-pass@1 %'))}  ->  {_pretty(tm.get('All-pass@1 %'))}")
            print(f"    Line coverage % {_pretty(base.get('Line coverage %'))}  ->  {_pretty(tm.get('Line coverage %'))}")
    if apr_rows:
        baseline = next((r for r in apr_rows if r["#"] == "APR Baseline"), {})
        testmate = next((r for r in apr_rows if r["#"] == "APR LoRA (TestMate)"), {})
        if baseline and testmate:
            print(f"  Part C APR repair rate (baseline -> TestMate):")
            print(f"    True repair %   {_pretty(baseline.get('True repair %'))}  ->  {_pretty(testmate.get('True repair %'))}")
    if pipeline_rows:
        full = next((r for r in pipeline_rows if r["#"] == "Full Pipeline"), {})
        if full:
            print(f"  Full Pipeline:")
            print(f"    Test detection  {_pretty(full.get('Test detection %'))} %")
            print(f"    Apparent repair {_pretty(full.get('Apparent repair %'))} %")
            print(f"    True repair     {_pretty(full.get('True repair %'))} %")
            print(f"    (Plausible-not-correct {_pretty(full.get('Plausible-not-correct %'))} %)")

    print(f"\nTables saved -> {out_dir}/")


if __name__ == "__main__":
    main()
