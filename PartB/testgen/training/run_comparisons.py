"""
Run the ablation comparison matrix on one test set, then print a table.

Sweeps the key variants needed for the paper and writes one JSON per variant
plus a combined comparison table:

  testmate    full production stack (all RAG layers + self-correction + LoRA)
  no_rag      LoRA only, no RAG layers, no loop   (isolates total RAG value)
  graph_only  only the graph layer                (isolates graph)
  no_graph    docs+vector+memory, no graph        (isolates graph vs vector)
  no_lora     full RAG + loop, base model         (isolates LoRA value)

Headline (HumanEval): testmate any_pass_1_rate.
Contribution (testgenevallite): RAG-lift = testmate - no_rag on coverage.

Each variant reloads the model (run_ablation owns model loading), so a full
sweep is a long GPU job — start with a small --sample.

Usage:
    python run_comparisons.py --dataset humaneval --sample 10
    python run_comparisons.py --dataset testgenevallite --sample 20 \
        --variants testmate no_rag no_lora
"""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import sys
# Force UTF-8 stdout/stderr so the emoji status prints (🗑️ ✓ ▶️ 💾 …) don't
# crash on Windows consoles / redirected streams that default to cp1252.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from _ablation_common import AblationConfig, run_ablation, run_mutation_pass  # noqa: E402

# Local model (zero download) + prefer the clean retrained adapter.
_MODEL_CANDIDATES = [
    r"D:\donwloader\Qwen2.5-Coder-7B-Instruct",
    r"D:\TestMate\huggingface_cache\hub\models--Qwen--Qwen2.5-Coder-7B",
]
_models = Path(__file__).parent.parent.parent / "models"
_LORA_CANDIDATES = [
    str(_models / "graphrag_lora_clean" / "final"),
    str(_models / "graphrag_lora" / "final"),
]

# variant name -> (toggle overrides). Booleans omitted default per AblationConfig.
VARIANTS = {
    "testmate":   dict(enable_layer1_docs=True,  enable_layer2_graph=True,  enable_layer2_vector=True,  enable_rag_memory=True,  enable_self_correction_loop=True,  enable_lora=True),
    "no_rag":     dict(enable_layer1_docs=False, enable_layer2_graph=False, enable_layer2_vector=False, enable_rag_memory=False, enable_self_correction_loop=False, enable_lora=True),
    "graph_only": dict(enable_layer1_docs=False, enable_layer2_graph=True,  enable_layer2_vector=False, enable_rag_memory=False, enable_self_correction_loop=False, enable_lora=True),
    "no_graph":   dict(enable_layer1_docs=True,  enable_layer2_graph=False, enable_layer2_vector=True,  enable_rag_memory=True,  enable_self_correction_loop=False, enable_lora=True),
    "no_lora":    dict(enable_layer1_docs=True,  enable_layer2_graph=True,  enable_layer2_vector=True,  enable_rag_memory=True,  enable_self_correction_loop=True,  enable_lora=False),
}

_TABLE_KEYS = [
    "any_pass_1_rate", "mean_pass_rate", "mean_line_coverage",
    "valid_line_coverage", "mean_branch_coverage", "mean_mutation_score",
    "graphrag_hit_rate", "mean_bes_score", "mean_wall_time",
]


def _is_complete(out_path: Path, sample) -> bool:
    """A variant is 'complete' when its output JSON has a `summary` block AND
    its file count matches the requested sample (so a smaller earlier run
    doesn't satisfy a larger one). `sample=None` means ALL files."""
    if not out_path.exists():
        return False
    try:
        d = json.loads(out_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    summ = d.get("summary")
    if not summ:
        return False
    if sample is None:
        return True                      # full run finished
    return summ.get("total_files", 0) >= sample


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["testeval", "humaneval", "testgenevallite"], default="humaneval")
    ap.add_argument("--sample", type=int, default=10, help="files per variant (None-like 0 = all)")
    ap.add_argument("--variants", nargs="*", default=list(VARIANTS),
                    help=f"subset of {list(VARIANTS)}")
    ap.add_argument("--max-seconds", type=float, default=None,
                    help="per-file deadline; defaults 300 (testeval/humaneval) / 900 (testgenevallite)")
    ap.add_argument("--force", action="store_true",
                    help="re-run variants even if a complete result already exists")
    ap.add_argument("--fresh", action="store_true",
                    help="full clean re-run: delete each variant's existing result + "
                         "checkpoint JSONs AND disable per-file resume. Use this after "
                         "editing generation CODE — a config-identical re-run would "
                         "otherwise reuse the old per-file outputs.")
    ap.add_argument("--suite", action="store_true",
                    help="coverage/suite mode (TestEval protocol): comprehensive "
                         "multi-case suite per program, no BES gate. Results go to "
                         "compare_<dataset>_suite/ so they don't collide with quality mode.")
    ap.add_argument("--mutation", action="store_true",
                    help="after each variant, run mutation testing (bug-detection) "
                         "on the files with passing tests, and fold mean_mutation_score "
                         "into the table. This is the quality axis the leaderboard ignores.")
    args = ap.parse_args()

    model_dir = next((p for p in _MODEL_CANDIDATES if os.path.isdir(p)), _MODEL_CANDIDATES[0])
    lora = next((p for p in _LORA_CANDIDATES if os.path.isdir(p)), _LORA_CANDIDATES[-1])
    sample = None if args.sample in (0, None) else args.sample
    max_sec = args.max_seconds or (900.0 if args.dataset == "testgenevallite" else 300.0)
    _suffix = "_suite" if args.suite else ""
    out_dir = Path("results") / f"compare_{args.dataset}{_suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Model: {model_dir}")
    print(f"LoRA:  {lora}  ({'CLEAN' if 'graphrag_lora_clean' in lora else 'OLD fallback'})")
    print(f"Mode:  {'SUITE/coverage (TestEval protocol, no BES gate)' if args.suite else 'quality (per-target, BES gate)'}")
    print(f"Dataset: {args.dataset} | sample: {sample} | variants: {args.variants}\n")

    summaries = {}
    for name in args.variants:
        if name not in VARIANTS:
            print(f"  (skip unknown variant {name})")
            continue
        toggles = VARIANTS[name]
        cfg = AblationConfig(
            variant=name,
            enable_plan_mode=False,
            lora_path=lora,
            model_id=model_dir,
            dataset=args.dataset,
            sample_size=sample,
            max_file_seconds=max_sec,
            max_retries=2,
            suite_mode=args.suite,
            resume=not args.fresh,        # --fresh disables per-file resume
            **toggles,
        )
        out_path = out_dir / f"{name}.json"

        # --fresh: delete this variant's result + checkpoint JSONs so nothing is
        # reused. Without this, the per-file resume + skip-if-complete logic
        # silently serves OLD outputs after a code edit.
        if args.fresh:
            for p in (out_path, out_path.with_suffix(".partial.json"),
                      out_path.with_name(f"{name}_with_mut.json"),
                      out_path.with_name(f"{name}_with_mut.partial.json")):
                if p.exists():
                    p.unlink()
                    print(f"  🗑️  fresh: removed {p.name}")

        # Skip-if-complete: a finished variant has a `summary` block in its
        # output JSON (the .partial.json is only the in-progress checkpoint).
        # On re-run we reuse it and move to the next variant unless --force/--fresh.
        if not args.force and not args.fresh and _is_complete(out_path, sample):
            print(f"\n{'='*70}\n  VARIANT: {name}  ✓ already complete — skipping generation\n{'='*70}")
        else:
            print(f"\n{'='*70}\n  VARIANT: {name}\n{'='*70}")
            run_ablation(cfg, output_path=str(out_path))

        try:
            summaries[name] = json.loads(out_path.read_text(encoding="utf-8"))["summary"]
        except Exception as exc:
            print(f"  (could not read summary for {name}: {exc})")
            continue

        # Optional Phase 2: mutation testing (bug-detection) on files that passed.
        if args.mutation:
            mut_path = out_path.with_name(f"{name}_with_mut.json")
            if args.force or not _is_complete(mut_path, sample):
                print(f"  🧬 mutation pass for {name} ...")
                try:
                    run_mutation_pass(str(out_path), str(mut_path))
                except Exception as exc:
                    print(f"  (mutation pass failed for {name}: {exc})")
            try:
                msum = json.loads(mut_path.read_text(encoding="utf-8"))["summary"]
                summaries[name]["mean_mutation_score"] = msum.get("mean_mutation_score")
                summaries[name]["mutation_tested_files"] = msum.get("mutation_tested_files")
            except Exception:
                pass

    # Comparison table
    print(f"\n{'='*70}\n  COMPARISON — {args.dataset}\n{'='*70}")
    header = "variant".ljust(12) + "".join(k[:14].rjust(15) for k in _TABLE_KEYS)
    print(header)
    print("-" * len(header))
    for name in args.variants:
        s = summaries.get(name)
        if not s:
            continue
        def _cell(k):
            v = s.get(k)
            return ("-" if v is None else f"{v:.3f}").rjust(15)
        row = name.ljust(12) + "".join(_cell(k) for k in _TABLE_KEYS)
        print(row)

    table_path = out_dir / "comparison_table.json"
    table_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(f"\nSaved combined table -> {table_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
