"""
Standalone subprocess entry-point for PartC auto-repair.

Called by testgen_api.py via subprocess.run() AFTER the PartB model exits.
BitsAndBytes 4-bit models on Windows don't release VRAM cleanly in-process,
so this subprocess loads the PartC model fresh, runs repair, then exits —
which fully reclaims ALL VRAM via OS-level cleanup.

Input (stdin JSON):
  {
    "bug_reports_path": "/abs/path/to/bug_reports.jsonl",
    "repo_dir":         "/abs/path/to/project",
    "primary_test":     "/abs/path/to/test_core_utils.py",
    "extra_test_files": ["/abs/path/to/test_core_utils_testmate.py"]
  }

Output (stdout):
  JSON list of repair result dicts (one per source file).
  Lines starting with "LOG:" are SSE log events forwarded to the parent.
"""
import sys
import os
import json
from pathlib import Path

os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
_PARTC = _ROOT / "PartC"

for p in [str(_HERE), str(_PARTC / "api"), str(_PARTC)]:
    if p not in sys.path:
        sys.path.insert(0, p)


def main() -> int:
    try:
        args = json.loads(sys.stdin.read())
    except Exception as exc:
        print(f"stdin parse failed: {exc}", file=sys.stderr)
        return 1

    bug_reports_path = args["bug_reports_path"]
    repo_dir = args["repo_dir"]

    if not os.path.isfile(bug_reports_path):
        print(json.dumps([]))
        return 0

    def log_callback(event_type, *cb_args):
        if event_type == "log":
            level = cb_args[0] if len(cb_args) > 1 else "info"
            msg = cb_args[-1] if cb_args else ""
            print(f"LOG:{level}:{msg}", file=sys.stderr, flush=True)
        else:
            print(f"EVT:{event_type}:{json.dumps(cb_args[0] if cb_args else {}, default=str)}",
                  file=sys.stderr, flush=True)

    try:
        from partc_api import execute_part_c

        from bug_to_partc import (
            _read_bug_reports, _group_by_source_file,
            _find_source_abs_path, _rewrite_bug_reports,
            _write_patched_file, _try_git_branch,
        )
        import time

        all_reports = _read_bug_reports(bug_reports_path)
        groups = _group_by_source_file(all_reports)

        if not groups:
            print(json.dumps([]))
            return 0

        print(f"LOG:info:  [subprocess] Repairing {len(groups)} source file(s)", file=sys.stderr, flush=True)
        print(f"EVT:repair_start:{json.dumps({'total_files': len(groups)})}", file=sys.stderr, flush=True)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        outputs_dir = _HERE / "outputs" / "patched"
        all_results = []

        for source_filename, bugs in groups.items():
            print(f"LOG:info:    Repairing: {source_filename}", file=sys.stderr, flush=True)

            source_abs = _find_source_abs_path(source_filename, repo_dir)
            if not source_abs:
                for b in bugs:
                    tf = b.get("test_file", "")
                    candidate = os.path.join(os.path.dirname(tf), source_filename)
                    if os.path.isfile(candidate):
                        source_abs = candidate
                        break
            if not source_abs:
                print(f"LOG:warning:    Could not locate {source_filename}", file=sys.stderr, flush=True)
                continue

            test_files = list({b["test_file"] for b in bugs
                              if b.get("test_file") and os.path.isfile(b["test_file"])})
            if not test_files:
                print(f"LOG:warning:    No valid test files for {source_filename}", file=sys.stderr, flush=True)
                continue

            primary_test = test_files[0]

            # Find generated test too
            _stem = os.path.splitext(source_filename)[0]
            extra = []
            for _gd in [_HERE / "generated_tests", Path(source_abs).parent]:
                _gc = _gd / f"test_{_stem}_testmate.py"
                if _gc.is_file() and str(_gc) not in test_files:
                    extra.append(str(_gc))

            print(f"LOG:info:    test: {os.path.basename(primary_test)} + {len(extra)} extra",
                  file=sys.stderr, flush=True)

            t0 = time.time()
            repair_result = execute_part_c(
                source_file=source_abs,
                test_file=primary_test,
                extra_test_files=extra,
                model=None,
                tokenizer=None,
                max_attempts=3,
                log_callback=log_callback,
            )
            elapsed = round(time.time() - t0, 1)

            repair_success = repair_result.get("success", False)
            patched_content = repair_result.get("patched", "")

            if repair_success:
                print(f"LOG:info:    REPAIRED in {elapsed}s", file=sys.stderr, flush=True)
            else:
                _err = repair_result.get("error", "all attempts exhausted")
                print(f"LOG:warning:    FAILED after {elapsed}s: {_err}", file=sys.stderr, flush=True)

            patched_path = None
            git_branch = None
            if repair_success and patched_content:
                patched_path = _write_patched_file(source_abs, patched_content, outputs_dir)

            # Update verdicts
            for bug in bugs:
                old = bug.get("verdict", "suspected")
                new = "confirmed" if repair_success else ("suspected" if old == "confirmed" else "discarded")
                bug["verdict"] = new
                bug["repair"] = {
                    "attempted": True, "success": repair_success,
                    "elapsed_sec": elapsed, "patched_path": patched_path,
                    "git_branch": git_branch, "verify_pass": repair_success,
                    "attempts": repair_result.get("attempts", []),
                    "suspicious": repair_result.get("suspicious", []),
                }

            _rewrite_bug_reports(bug_reports_path, all_reports)

            file_result = {
                "source_file": source_filename, "source_abs": source_abs,
                "repair_success": repair_success, "patched_path": patched_path,
                "git_branch": git_branch, "elapsed_sec": elapsed,
                "bugs_repaired": len(bugs),
                "attempts": repair_result.get("attempts", []),
                "suspicious": repair_result.get("suspicious", []),
                "original": repair_result.get("original", ""),
                "patched": repair_result.get("patched", ""),
            }
            all_results.append(file_result)
            print(f"EVT:repair_complete:{json.dumps(file_result, default=str)}", file=sys.stderr, flush=True)

        print(json.dumps(all_results, default=str))
        return 0

    except Exception as exc:
        import traceback
        print(f"Repair subprocess crashed: {exc}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        print(json.dumps([]))
        return 1


if __name__ == "__main__":
    sys.exit(main())
