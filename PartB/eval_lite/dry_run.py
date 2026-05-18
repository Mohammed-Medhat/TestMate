import sys
import os
import json
import time
import re
import subprocess
import threading

# Add testgen to path so we can import directly
sys.path.insert(0, r"D:\TestMate\TestMate\PartB\testgen")

FILES_DIR = "testgen_eval_files"
RESULTS_FILE = "testgeneval_results.json"
TIMEOUT_PER_FILE = 5400    # 90 minutes max per file (P19)
MAX_TOTAL_HOURS = 10
MAX_FILES = 50
SKIP_LINE_LIMIT = 2000     # P9: skip files larger than this


# ── P15: TeeStream captures output while still printing to console ──
class TeeStream:
    """Writes to both the real stream and a StringIO buffer.
    Fully compatible with the stream protocol (isatty, fileno, encoding, etc.)."""
    def __init__(self, real_stream):
        self.real = real_stream
        self.buffer = []
    def write(self, text):
        self.real.write(text)
        self.buffer.append(text)
    def flush(self):
        self.real.flush()
    def getvalue(self):
        return "".join(self.buffer)
    def isatty(self):
        return hasattr(self.real, 'isatty') and self.real.isatty()
    def fileno(self):
        return self.real.fileno()
    @property
    def encoding(self):
        return getattr(self.real, 'encoding', 'utf-8')
    @property
    def errors(self):
        return getattr(self.real, 'errors', 'strict')
    def readable(self):
        return False
    def writable(self):
        return True
    def seekable(self):
        return False
    def __getattr__(self, name):
        # Delegate any other attribute access to the real stream
        return getattr(self.real, name)

def load_existing_results():
    if os.path.exists(RESULTS_FILE):
        try:
            with open(RESULTS_FILE) as f:
                return json.load(f)
        except json.JSONDecodeError:
            return []
    return []


def parse_output(output: str) -> dict:
    metrics = {
        "targets_covered": 0,
        "targets_total": 0,
        "line_coverage": 0.0,
        "branch_coverage": 0.0,
        "mutation_score": 0.0,
        "composite_score": 0.0,
        "assertion_score": 0.0,
        "diversity_score": 0.0,
        "edge_cases_score": 0.0,
        "tests_generated": 0,
        "absolute_success": False,
    }
    m = re.search(r"Chunking complete: (\d+)/(\d+)", output)
    if m:
        covered_raw = int(m.group(1))
        total_raw   = int(m.group(2))
        # Cap targets_covered at targets_total (retries can exceed total)
        metrics["targets_total"]   = total_raw
        metrics["targets_covered"] = min(covered_raw, total_raw)
    m = re.search(r"composite quality:\s*(\d+)/100", output)
    if m:
        metrics["composite_score"] = float(m.group(1))
    m = re.search(r"Branch cov:\s*(\d+\.?\d*)/100", output)
    if m:
        metrics["branch_coverage"] = float(m.group(1))
    m = re.search(r"Mutation:\s*(\d+\.?\d*)/100", output)
    if m:
        metrics["mutation_score"] = float(m.group(1))
    m = re.search(r"Assertion:\s*(\d+\.?\d*)/100", output)
    if m:
        metrics["assertion_score"] = float(m.group(1))
    m = re.search(r"Diversity:\s*(\d+\.?\d*)/100", output)
    if m:
        metrics["diversity_score"] = float(m.group(1))
    m = re.search(r"Edge cases:\s*(\d+\.?\d*)/100", output)
    if m:
        metrics["edge_cases_score"] = float(m.group(1))
    m = re.search(r"Line cov:\s*(\d+\.?\d*)/100", output)
    if m:
        metrics["line_coverage"] = float(m.group(1))
    # P14: Tests: N is now always printed
    m = re.search(r"Tests: (\d+)", output)
    if m:
        metrics["tests_generated"] = int(m.group(1))
    if "ABSOLUTE SUCCESS" in output:
        metrics["absolute_success"] = True
    # Use >= so retries that exceed target count still register as all_pass
    metrics["all_pass_at_1"] = (
        metrics["targets_covered"] >= metrics["targets_total"]
        and metrics["targets_total"] > 0
    )
    metrics["any_pass_at_1"] = metrics["targets_covered"] > 0
    return metrics


def print_summary(results):
    if not results:
        return
    ok       = [r for r in results if r.get("status") == "ok"]
    no_tests = [r for r in results if r.get("status") == "no_tests"]
    skipped  = [r for r in results if r.get("status") == "import_error"]
    timeout  = [r for r in results if r.get("status") == "timeout"]
    too_big  = [r for r in results if r.get("status") == "skipped_large"]

    print(f"\n{'='*60}")
    print("FINAL RESULTS TABLE")
    print(f"{'='*60}")
    print(f"{'File':<32} {'Pass%':<7} {'Line':<7} "
          f"{'Branch':<8} {'Mut':<7} {'Comp':<6}")
    print("-" * 60)
    for r in results:
        total   = r.get("targets_total", 0)
        covered = r.get("targets_covered", 0)
        # P8: Safe division — only compute pass_rate for entries with targets
        pass_rate = round(covered / total * 100, 1) if total > 0 else 0
        tag = ""
        if r.get("status") == "import_error":   tag = " [SKIP]"
        elif r.get("status") == "timeout":      tag = " [TIMEOUT]"
        elif r.get("status") == "no_tests":     tag = " [0 tests]"
        elif r.get("status") == "skipped_large": tag = " [BIG]"
        fname = r["file"][:28] + tag
        print(f"{fname:<32} {pass_rate:<7} "
              f"{r.get('line_coverage',0):<7} "
              f"{r.get('branch_coverage',0):<8} "
              f"{r.get('mutation_score',0):<7} "
              f"{r.get('composite_score',0):<6}")
    valid = ok
    n_total = len(results)
    # Averages over OK files (standard SWE-bench methodology)
    if valid:
        avg_pass   = sum(
            r["targets_covered"] / max(r["targets_total"], 1) * 100
            for r in valid) / len(valid)
        avg_line   = sum(r.get("line_coverage", 0) for r in valid) / len(valid)
        avg_branch = sum(r.get("branch_coverage", 0) for r in valid) / len(valid)
        avg_mut    = sum(r.get("mutation_score", 0) for r in valid) / len(valid)
        avg_comp   = sum(r.get("composite_score", 0) for r in valid) / len(valid)
        print("-" * 60)
        print(f"{'AVG (ok files)':<32} {avg_pass:<7.1f} "
              f"{avg_line:<7.1f} {avg_branch:<8.1f} "
              f"{avg_mut:<7.1f} {avg_comp:<6.1f}")
    # Averages over ALL files (transparent reporting)
    if n_total > 0:
        tot_line = sum(r.get("line_coverage", 0) for r in results) / n_total
        tot_mut  = sum(r.get("mutation_score", 0) for r in results) / n_total
        tot_comp = sum(r.get("composite_score", 0) for r in results) / n_total
        print(f"{'AVG (all files)':<32} {'—':<7} "
              f"{tot_line:<7.1f} {'—':<8} "
              f"{tot_mut:<7.1f} {tot_comp:<6.1f}")
    print(f"\nSummary:")
    print(f"  OK:           {len(ok)}/{n_total}")
    print(f"  No tests:     {len(no_tests)}/{n_total}")
    print(f"  Import error: {len(skipped)}/{n_total}")
    print(f"  Timeout:      {len(timeout)}/{n_total}")
    if too_big:
        print(f"  Skipped (big):{len(too_big)}/{n_total}")
    # Pass@1 uses ALL files as denominator (fair comparison)
    all_pass  = sum(1 for r in results if r.get("all_pass_at_1"))
    any_pass  = sum(1 for r in results if r.get("any_pass_at_1"))
    print(f"\nPass@1 Metrics (denominator = all {n_total} files):")
    print(f"  All Pass@1:  "
          f"{all_pass/max(n_total,1)*100:.1f}%  "
          f"(vs CodeLlama 7B: 3.2%, GPT-4o: 7.5%)")
    print(f"  Any Pass@1:  "
          f"{any_pass/max(n_total,1)*100:.1f}%  "
          f"(vs CodeLlama 7B: 4.1%, GPT-4o: 64.0%)")
    if valid:
        print(f"  Coverage:    {avg_line:.1f}%  "
              f"(vs CodeLlama 7B: 1.2%, GPT-4o: 35.2%)")
        print(f"  Mutation:    {avg_mut:.1f}%  "
              f"(vs CodeLlama 7B: 0.5%, GPT-4o: 18.8%)")


def main():
    # ── Load model ONCE for all files (P1) ──
    print("Loading model once for entire evaluation run...")
    from main import load_model, autonomous_loop
    import torch

    model, tokenizer = load_model()
    print("Model loaded. Starting evaluation...\n")

    existing   = load_existing_results()
    done_files = {r["file"] for r in existing}
    results    = list(existing)

    if done_files:
        print(f"Resuming — {len(done_files)} files already done")

    all_files = sorted([
        f for f in os.listdir(FILES_DIR)
        if f.endswith(".py") and not f.startswith("test_")
    ])
    remaining = [f for f in all_files if f not in done_files]
    remaining = remaining[:MAX_FILES]

    print(f"Files total: {len(all_files)} | "
          f"Already done: {len(done_files)} | "
          f"Remaining: {len(remaining)}")

    # P16: Clean stale test files before starting
    for stale in os.listdir(FILES_DIR):
        if stale.startswith("test_") and stale.endswith(".py"):
            stale_path = os.path.join(FILES_DIR, stale)
            try:
                os.remove(stale_path)
                print(f"   🧹 Cleaned stale: {stale}")
            except OSError:
                pass

    session_start = time.time()
    runtimes = []  # P13: Track runtimes for ETA

    for file_idx, filename in enumerate(remaining, 1):
        elapsed_hours = (time.time() - session_start) / 3600
        if elapsed_hours >= MAX_TOTAL_HOURS:
            print(f"\n⏰ Reached {MAX_TOTAL_HOURS}h limit. Stopping.")
            break

        filepath = os.path.join(
            os.path.abspath(FILES_DIR), filename)
        try:
            with open(filepath, encoding="utf-8") as f:
                lines = len(f.readlines())
        except Exception:
            lines = 0

        total_done = len(done_files) + file_idx

        # P13: ETA display
        if runtimes:
            avg_runtime = sum(runtimes) / len(runtimes)
            files_left = len(remaining) - file_idx
            eta_minutes = (avg_runtime * files_left) / 60
            eta_str = f"ETA: {eta_minutes:.0f}min"
        else:
            eta_str = "ETA: calculating..."

        print(f"\n[{total_done}/{len(all_files)}] {filename} "
              f"({lines} lines) | "
              f"session: {elapsed_hours:.1f}h | {eta_str}")

        # P9: Skip very large files
        if lines > SKIP_LINE_LIMIT:
            print(f"   ⏭️  Skipped: {lines} lines > {SKIP_LINE_LIMIT} limit")
            res = {"file": filename,
                   "status": "skipped_large",
                   "lines": lines}
            results.append(res)
            with open(RESULTS_FILE, "w") as f:
                json.dump(results, f, indent=2)
            done_files.add(filename)
            runtimes.append(1.0)  # negligible
            continue

        # Quick import check
        check = subprocess.run(
            ["python", "-c",
             f"import importlib.util, sys; "
             f"spec = importlib.util.spec_from_file_location"
             f"('m', r'{filepath}'); "
             f"mod = importlib.util.module_from_spec(spec); "
             f"spec.loader.exec_module(mod)"],
            capture_output=True, text=True, timeout=20
        )
        if check.returncode != 0:
            err = (check.stderr.strip().splitlines()[-1]
                   if check.stderr else "unknown")
            missing = re.search(
                r"No module named '([^']+)'",
                check.stderr or "")
            if missing:
                pkg = missing.group(1).split(".")[0]
                pip_map = {
                    "PIL": "pillow",
                    "cv2": "opencv-python",
                    "sklearn": "scikit-learn",
                    "bs4": "beautifulsoup4",
                    "yaml": "pyyaml",
                }
                pip_pkg = pip_map.get(pkg, pkg)
                subprocess.run(
                    ["pip", "install", pip_pkg, "-q"],
                    capture_output=True, timeout=60)
                check2 = subprocess.run(
                    ["python", "-c",
                     f"import importlib.util; "
                     f"spec = importlib.util.spec_from_file_location"
                     f"('m', r'{filepath}'); "
                     f"mod = importlib.util.module_from_spec(spec); "
                     f"spec.loader.exec_module(mod)"],
                    capture_output=True, text=True, timeout=20)
                if check2.returncode != 0:
                    print(f"   ❌ Skipped (import after install): {err}")
                    res = {"file": filename,
                           "status": "import_error",
                           "import_error": err,
                           "lines": lines}
                    results.append(res)
                    with open(RESULTS_FILE, "w") as f:
                        json.dump(results, f, indent=2)
                    done_files.add(filename)
                    runtimes.append(1.0)
                    continue
            else:
                # Only fall through for Django-related errors
                # (handled by framework header in test file)
                # All other import errors should skip
                is_django_error = (
                    "django" in err.lower() or
                    "settings" in err.lower() or
                    "apps" in err.lower()
                )
                if is_django_error:
                    print(f"   ⚠️  Django import warning "
                          f"(will try with setup): {err}")
                else:
                    print(f"   ❌ Skipped (import error): {err}")
                    res = {"file": filename,
                           "status": "import_error",
                           "import_error": err,
                           "lines": lines}
                    results.append(res)
                    with open(RESULTS_FILE, "w") as f:
                        json.dump(results, f, indent=2)
                    done_files.add(filename)
                    runtimes.append(1.0)
                    continue

        # Set max targets based on file size
        max_targets = None
        if lines > 1000:
            max_targets = 10
        elif lines > 300:
            max_targets = 15

        print(f"   🚀 Running TestMate "
              f"(max {TIMEOUT_PER_FILE//60}min)...")
        start = time.time()

        # ── P15: Use TeeStream instead of redirect_stdout ──
        # This captures output for parsing while still printing to console
        tee = TeeStream(sys.__stdout__)
        tee_err = TeeStream(sys.__stderr__)
        old_stdout = sys.stdout
        old_stderr = sys.stderr

        # P19: Use threading for timeout on Windows
        result_holder = {"success": False, "error": None}

        def _run_loop():
            try:
                sys.stdout = tee
                sys.stderr = tee_err
                result_holder["success"] = autonomous_loop(
                    model, tokenizer,
                    filepath,
                    import_path=None,
                    max_targets=max_targets,
                )
            except Exception as e:
                result_holder["error"] = str(e)
                print(f"\nERROR: {e}")
            finally:
                sys.stdout = old_stdout
                sys.stderr = old_stderr

        worker = threading.Thread(target=_run_loop, daemon=True)
        worker.start()
        worker.join(timeout=TIMEOUT_PER_FILE)
        timed_out = worker.is_alive()

        # Restore streams regardless
        sys.stdout = old_stdout
        sys.stderr = old_stderr

        output  = tee.getvalue() + tee_err.getvalue()
        runtime = round(time.time() - start, 1)
        runtimes.append(runtime)

        if timed_out:
            print(f"   ⏳ Timeout after "
                  f"{TIMEOUT_PER_FILE//60}min")
            res = {"file": filename, "status": "timeout",
                   "runtime_seconds": TIMEOUT_PER_FILE,
                   "lines": lines}
            results.append(res)
            with open(RESULTS_FILE, "w") as f:
                json.dump(results, f, indent=2)
            done_files.add(filename)
            continue

        metrics = parse_output(output)
        metrics["file"]            = filename
        metrics["lines"]           = lines
        metrics["runtime_seconds"] = runtime
        metrics["runtime_minutes"] = round(runtime / 60, 1)

        if metrics["targets_total"] == 0:
            status = "no_tests"
        elif metrics["targets_covered"] == 0:
            status = "no_tests"
        else:
            status = "ok"
        metrics["status"] = status

        results.append(metrics)
        done_files.add(filename)
        with open(RESULTS_FILE, "w") as f:
            json.dump(results, f, indent=2)

        emoji = "✅" if status == "ok" else "⚠️ "
        print(f"   {emoji} "
              f"{metrics['targets_covered']}/"
              f"{metrics['targets_total']} targets | "
              f"mut:{metrics['mutation_score']:.0f}% | "
              f"comp:{metrics['composite_score']:.0f}/100 | "
              f"{metrics['runtime_minutes']}min | "
              f"tests:{metrics['tests_generated']}")

        # P11: Clear CUDA cache between files to prevent memory buildup
        torch.cuda.empty_cache()

    print_summary(results)
    print(f"\nResults saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()