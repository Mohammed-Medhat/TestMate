"""
TestGenEval-Lite Full Evaluation on Kaggle
==========================================
Runs TestMate on ALL 160 files from the TestGenEval-Lite benchmark,
installing the correct package version per-repo to fix import errors.

Usage on Kaggle:
  1. Upload this file + main.py + rag_store.py to Kaggle
  2. Attach your model weights as a Kaggle dataset
  3. Run cells in order

This script groups files by (repo, version), installs the matching
pip package so relative imports work, then runs TestMate's autonomous_loop.
"""

import os
import sys
import json
import time
import re
import subprocess
import shutil
from pathlib import Path

# ── Configuration ──
RESULTS_FILE = "testgeneval_results_full.json"
EVAL_DIR = "testgen_eval_files_full"
MAX_TOTAL_HOURS = 30       # Kaggle gives 12h per session; set higher for multi-session
TIMEOUT_PER_FILE = 3600    # 60 min max per file
SKIP_LINE_LIMIT = 5000     # Skip files larger than this (was 2000)
MAX_TARGETS = 15           # Max targets per file

# ── Repo → pip package mapping ──
# Maps GitHub repo slug to (pip_package_name, version_prefix_map)
# version_prefix_map maps dataset version strings to pip version strings
REPO_PIP_MAP = {
    "django/django": {
        "package": "Django",
        "versions": {
            "3.0": "3.0.14", "3.1": "3.1.14", "3.2": "3.2.25",
            "4.0": "4.0.10", "4.1": "4.1.13", "4.2": "4.2.16",
        },
        "extras": [],  # additional packages needed
    },
    "sympy/sympy": {
        "package": "sympy",
        "versions": {
            "1.0": "1.0.1", "1.1": "1.1.1", "1.2": "1.2",
            "1.4": "1.4", "1.5": "1.5", "1.6": "1.6.2",
            "1.7": "1.7.1", "1.8": "1.8", "1.9": "1.9",
            "1.10": "1.10.1", "1.11": "1.11.1", "1.12": "1.12",
        },
        "extras": ["mpmath"],
    },
    "pytest-dev/pytest": {
        "package": "pytest",
        "versions": {
            "4.4": "4.4.2", "4.5": "4.5.0", "4.6": "4.6.11",
            "5.0": "5.0.2", "5.1": "5.1.3", "5.2": "5.2.4",
            "5.3": "5.3.5", "5.4": "5.4.3", "6.0": "6.0.2",
            "7.0": "7.0.1", "7.1": "7.1.3", "7.2": "7.2.2",
        },
        "extras": [],
    },
    "scikit-learn/scikit-learn": {
        "package": "scikit-learn",
        "versions": {
            "0.20": "0.20.4", "0.21": "0.21.3", "0.22": "0.22.2",
            "1.0": "1.0.2", "1.1": "1.1.3", "1.2": "1.2.2",
            "1.3": "1.3.2",
        },
        "extras": ["numpy", "scipy"],
    },
    "sphinx-doc/sphinx": {
        "package": "Sphinx",
        "versions": {
            "3.0": "3.0.4", "3.1": "3.1.2", "3.2": "3.2.1",
            "3.3": "3.3.1", "3.4": "3.4.3", "3.5": "3.5.4",
            "4.0": "4.0.3", "4.1": "4.1.2",
        },
        "extras": [],
    },
    "mwaskom/seaborn": {
        "package": "seaborn",
        "versions": {
            "0.12": "0.12.2", "0.13": "0.13.2",
        },
        "extras": ["matplotlib", "pandas"],
    },
    "pandas-dev/pandas": {
        "package": "pandas",
        "versions": {
            "1.3": "1.3.5", "1.4": "1.4.4", "1.5": "1.5.3",
            "2.0": "2.0.3",
        },
        "extras": ["numpy"],
    },
    "matplotlib/matplotlib": {
        "package": "matplotlib",
        "versions": {
            "3.3": "3.3.4", "3.4": "3.4.3", "3.5": "3.5.3",
            "3.6": "3.6.3", "3.7": "3.7.5",
        },
        "extras": [],
    },
    "pallets/flask": {
        "package": "flask",
        "versions": {
            "2.0": "2.0.3", "2.1": "2.1.3", "2.2": "2.2.5",
        },
        "extras": [],
    },
    "psf/requests": {
        "package": "requests",
        "versions": {
            "2.25": "2.25.1", "2.26": "2.26.0", "2.27": "2.27.1",
            "2.28": "2.28.2", "3.0": "3.0.0",
        },
        "extras": [],
    },
    "pydata/xarray": {
        "package": "xarray",
        "versions": {
            "0.19": "0.19.0", "0.20": "0.20.2",
            "2022.03": "2022.3.0", "2022.06": "2022.6.0",
        },
        "extras": ["numpy", "pandas"],
    },
}


def derive_import_path(code_file: str) -> str:
    """Derive Python import path from the dataset's code_file field.

    E.g., 'django/utils/autoreload.py' → 'django.utils.autoreload'
         'sympy/core/numbers.py'       → 'sympy.core.numbers'
         'sklearn/base.py'             → 'sklearn.base'
    """
    if not code_file:
        return None
    # Normalize and strip .py
    path = code_file.replace("\\", "/")
    if path.endswith(".py"):
        path = path[:-3]
    # Convert slashes to dots
    import_path = path.replace("/", ".")
    # Remove trailing .__init__ if present
    if import_path.endswith(".__init__"):
        import_path = import_path[:-9]
    return import_path


def get_pip_version(repo: str, version: str) -> str:
    """Map dataset (repo, version) to a pip install string."""
    info = REPO_PIP_MAP.get(repo)
    if not info:
        return None

    pkg = info["package"]

    # Try exact match first
    if version in info["versions"]:
        return f"{pkg}=={info['versions'][version]}"

    # Try prefix match (e.g., "3.2" matches "3.2.x")
    for v_prefix, v_pip in info["versions"].items():
        if version.startswith(v_prefix):
            return f"{pkg}=={v_pip}"

    # Fallback: install the closest version
    return f"{pkg}=={version}"


def install_repo(repo: str, version: str) -> bool:
    """Install the matching pip package for a repo+version. Returns success."""
    pip_spec = get_pip_version(repo, version)
    if not pip_spec:
        print(f"   ⚠️  Unknown repo: {repo}")
        return False

    info = REPO_PIP_MAP.get(repo, {})
    extras = info.get("extras", [])

    print(f"   📦 Installing {pip_spec}...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", pip_spec, "-q", "--no-warn-conflicts"],
        capture_output=True, text=True, timeout=120
    )
    if result.returncode != 0:
        print(f"   ❌ Failed to install {pip_spec}: {result.stderr[-200:]}")
        return False

    # Install extras
    for extra in extras:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", extra, "-q"],
            capture_output=True, timeout=60
        )

    print(f"   ✅ Installed {pip_spec}")
    return True


def uninstall_repo(repo: str) -> None:
    info = REPO_PIP_MAP.get(repo)
    if not info:
        return
    pkg = info["package"]
    # NEVER uninstall pytest — tests need it
    if pkg.lower() in ("pytest", "py.test"):
        return
    subprocess.run(
        [sys.executable, "-m", "pip", "uninstall", pkg, "-y", "-q"],
        capture_output=True, timeout=30
    )


def find_installed_file(repo: str, code_file: str) -> str:
    """Find where the code_file lives in the installed package.

    E.g., code_file='django/core/checks/templates.py'
    →  /path/to/site-packages/django/core/checks/templates.py
    """
    parts = code_file.replace("\\", "/").split("/")
    if not parts:
        return None

    # The first part is usually the package name
    top_pkg = parts[0]
    try:
        mod = __import__(top_pkg)
        pkg_dir = os.path.dirname(mod.__file__)
        # Join remaining path parts
        installed_path = os.path.join(pkg_dir, *parts[1:])
        if os.path.exists(installed_path):
            return installed_path
    except (ImportError, AttributeError):
        pass

    return None


def setup_target_file(sample: dict, eval_dir: str) -> tuple:
    """Write code_src to a file and return (filepath, is_installed).

    Strategy:
    1. Try to find the file in the installed package and overwrite it
       (so relative imports work naturally)
    2. Fallback: write to eval_dir as standalone file

    Returns:
        (filepath, is_installed): is_installed=True means the file is
        inside site-packages and import_path should be used.
    """
    code_src = sample.get("code_src", "")
    code_file = sample.get("code_file", "")
    idx = sample.get("id", 0)

    if not code_src:
        return None, False

    # Strategy 1: Overwrite in installed package
    installed_path = find_installed_file(sample.get("repo", ""), code_file)
    if installed_path:
        # Backup original
        backup = installed_path + ".bak"
        if not os.path.exists(backup):
            shutil.copy2(installed_path, backup)
        with open(installed_path, "w", encoding="utf-8") as f:
            f.write(code_src)
        return installed_path, True

    # Strategy 2: Standalone file in eval_dir
    basename = os.path.basename(code_file) if code_file else f"sample_{idx}.py"
    safe_name = f"{idx}_{basename}"
    filepath = os.path.join(eval_dir, safe_name)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(code_src)
    return filepath, False


def restore_original(sample: dict) -> None:
    """Restore the original installed file from backup."""
    code_file = sample.get("code_file", "")
    installed_path = find_installed_file(sample.get("repo", ""), code_file)
    if installed_path:
        backup = installed_path + ".bak"
        if os.path.exists(backup):
            shutil.copy2(backup, installed_path)
            os.remove(backup)


def quick_import_check(filepath: str) -> tuple:
    """Quick check if the file can be imported. Returns (ok, error_msg)."""
    result = subprocess.run(
        [sys.executable, "-c",
         f"import importlib.util, sys; "
         f"spec = importlib.util.spec_from_file_location"
         f"('m', r'{filepath}'); "
         f"mod = importlib.util.module_from_spec(spec); "
         f"spec.loader.exec_module(mod)"],
        capture_output=True, text=True, timeout=20
    )
    if result.returncode != 0:
        err = (result.stderr.strip().splitlines()[-1]
               if result.stderr else "unknown")
        return False, err
    return True, ""


def load_existing_results():
    """Load previously saved results for resuming."""
    if os.path.exists(RESULTS_FILE):
        try:
            with open(RESULTS_FILE) as f:
                return json.load(f)
        except json.JSONDecodeError:
            return []
    return []


def parse_output(output: str) -> dict:
    """Parse TestMate output for metrics."""
    metrics = {
        "targets_covered": 0, "targets_total": 0,
        "line_coverage": 0.0, "branch_coverage": 0.0,
        "mutation_score": 0.0, "composite_score": 0.0,
        "assertion_score": 0.0, "diversity_score": 0.0,
        "edge_cases_score": 0.0, "tests_generated": 0,
        "absolute_success": False,
    }
    m = re.search(r"Chunking complete: (\d+)/(\d+)", output)
    if m:
        metrics["targets_total"] = int(m.group(2))
        metrics["targets_covered"] = min(int(m.group(1)), int(m.group(2)))
    m = re.search(r"composite quality:\s*(\d+)/100", output)
    if m: metrics["composite_score"] = float(m.group(1))
    m = re.search(r"Branch cov:\s*(\d+\.?\d*)/100", output)
    if m: metrics["branch_coverage"] = float(m.group(1))
    m = re.search(r"Mutation:\s*(\d+\.?\d*)/100", output)
    if m: metrics["mutation_score"] = float(m.group(1))
    m = re.search(r"Assertion:\s*(\d+\.?\d*)/100", output)
    if m: metrics["assertion_score"] = float(m.group(1))
    m = re.search(r"Diversity:\s*(\d+\.?\d*)/100", output)
    if m: metrics["diversity_score"] = float(m.group(1))
    m = re.search(r"Edge cases:\s*(\d+\.?\d*)/100", output)
    if m: metrics["edge_cases_score"] = float(m.group(1))
    m = re.search(r"Line cov:\s*(\d+\.?\d*)/100", output)
    if m: metrics["line_coverage"] = float(m.group(1))
    m = re.search(r"Tests: (\d+)", output)
    if m: metrics["tests_generated"] = int(m.group(1))
    if "ABSOLUTE SUCCESS" in output:
        metrics["absolute_success"] = True
    metrics["all_pass_at_1"] = (
        metrics["targets_covered"] >= metrics["targets_total"]
        and metrics["targets_total"] > 0
    )
    metrics["any_pass_at_1"] = metrics["targets_covered"] > 0
    return metrics


def print_summary(results):
    """Print final summary table with comparisons to baselines."""
    if not results:
        return

    ok = [r for r in results if r.get("status") == "ok"]
    no_tests = [r for r in results if r.get("status") == "no_tests"]
    skipped = [r for r in results if r.get("status") == "import_error"]
    timeout = [r for r in results if r.get("status") == "timeout"]
    too_big = [r for r in results if r.get("status") == "skipped_large"]
    n_total = len(results)

    print(f"\n{'='*70}")
    print("TESTGENEVAL-LITE FULL RESULTS (160 files)")
    print(f"{'='*70}")
    print(f"{'File':<35} {'Pass%':<7} {'Line':<7} "
          f"{'Branch':<8} {'Mut':<7} {'Comp':<6}")
    print("-" * 70)

    for r in results:
        total = r.get("targets_total", 0)
        covered = r.get("targets_covered", 0)
        pass_rate = round(covered / total * 100, 1) if total > 0 else 0
        tag = ""
        if r.get("status") == "import_error": tag = " [SKIP]"
        elif r.get("status") == "timeout": tag = " [TIMEOUT]"
        elif r.get("status") == "no_tests": tag = " [0 tests]"
        elif r.get("status") == "skipped_large": tag = " [BIG]"
        fname = r["file"][:30] + tag
        print(f"{fname:<35} {pass_rate:<7} "
              f"{r.get('line_coverage', 0):<7} "
              f"{r.get('branch_coverage', 0):<8} "
              f"{r.get('mutation_score', 0):<7} "
              f"{r.get('composite_score', 0):<6}")

    # Averages
    if ok:
        avg_line = sum(r.get("line_coverage", 0) for r in ok) / len(ok)
        avg_branch = sum(r.get("branch_coverage", 0) for r in ok) / len(ok)
        avg_mut = sum(r.get("mutation_score", 0) for r in ok) / len(ok)
        avg_comp = sum(r.get("composite_score", 0) for r in ok) / len(ok)
        print("-" * 70)
        print(f"{'AVG (ok files)':<35} {'—':<7} "
              f"{avg_line:<7.1f} {avg_branch:<8.1f} "
              f"{avg_mut:<7.1f} {avg_comp:<6.1f}")

    if n_total > 0:
        tot_line = sum(r.get("line_coverage", 0) for r in results) / n_total
        tot_mut = sum(r.get("mutation_score", 0) for r in results) / n_total
        print(f"{'AVG (all files)':<35} {'—':<7} "
              f"{tot_line:<7.1f} {'—':<8} "
              f"{tot_mut:<7.1f} {'—':<6}")

    print(f"\nSummary:")
    print(f"  OK:           {len(ok)}/{n_total}")
    print(f"  No tests:     {len(no_tests)}/{n_total}")
    print(f"  Import error: {len(skipped)}/{n_total}")
    print(f"  Timeout:      {len(timeout)}/{n_total}")
    if too_big:
        print(f"  Skipped (big):{len(too_big)}/{n_total}")

    all_pass = sum(1 for r in results if r.get("all_pass_at_1"))
    any_pass = sum(1 for r in results if r.get("any_pass_at_1"))
    print(f"\nPass@1 Metrics (denominator = all {n_total} files):")
    print(f"  All Pass@1:  {all_pass / max(n_total, 1) * 100:.1f}%  "
          f"(vs CodeLlama 7B: 3.2%, GPT-4o: 7.5%)")
    print(f"  Any Pass@1:  {any_pass / max(n_total, 1) * 100:.1f}%  "
          f"(vs CodeLlama 7B: 4.1%, GPT-4o: 64.0%)")
    if ok:
        print(f"  Coverage:    {avg_line:.1f}%  "
              f"(vs CodeLlama 7B: 1.2%, GPT-4o: 35.2%)")
        print(f"  Mutation:    {avg_mut:.1f}%  "
              f"(vs CodeLlama 7B: 0.5%, GPT-4o: 18.8%)")


# ============================================================
# MAIN
# ============================================================
def main():
    import threading

    # ── Add testgen to path ──
    sys.path.insert(0, os.path.abspath("../testgen"))

    print("=" * 60)
    print("TestGenEval-Lite FULL Evaluation (160 files)")
    print("=" * 60)

    print("📌 Pinning pytest...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "pytest>=7.0", "-q", "--no-warn-conflicts"],
        capture_output=True, timeout=60
    )

    # ── Load dataset ──
    print("\n1️⃣  Loading dataset from HuggingFace...")
    from datasets import load_dataset
    dataset = load_dataset("kjain14/testgenevallite", split="test")
    print(f"   Loaded {len(dataset)} samples")

    # ── Group by (repo, version) ──
    repo_groups = {}
    for i, sample in enumerate(dataset):
        repo = sample.get("repo", "unknown")
        version = sample.get("version", "0.0")
        key = (repo, version)
        if key not in repo_groups:
            repo_groups[key] = []
        repo_groups[key].append((i, sample))

    print(f"   Found {len(repo_groups)} repo-version groups:")
    for (repo, ver), samples in sorted(repo_groups.items()):
        print(f"     {repo} v{ver}: {len(samples)} files")

    # ── Load model ──
    print("\n2️⃣  Loading model...")
    from main import load_model, autonomous_loop
    import torch
    model, tokenizer = load_model()
    print("   Model ready.\n")

    # ── Setup ──
    os.makedirs(EVAL_DIR, exist_ok=True)
    existing = load_existing_results()
    done_ids = {r.get("dataset_id") for r in existing if "dataset_id" in r}
    results = list(existing)

    if done_ids:
        print(f"   Resuming — {len(done_ids)} samples already done\n")

    session_start = time.time()
    runtimes = []

    # ── Process repo by repo ──
    for group_idx, ((repo, version), samples) in enumerate(sorted(repo_groups.items())):
        elapsed_hours = (time.time() - session_start) / 3600
        if elapsed_hours >= MAX_TOTAL_HOURS:
            print(f"\n⏰ Reached {MAX_TOTAL_HOURS}h limit. Stopping.")
            break

        remaining_in_group = [(i, s) for i, s in samples if i not in done_ids]
        if not remaining_in_group:
            print(f"\n[Group {group_idx + 1}/{len(repo_groups)}] "
                  f"{repo} v{version} — all done, skipping")
            continue

        print(f"\n{'=' * 60}")
        print(f"[Group {group_idx + 1}/{len(repo_groups)}] "
              f"{repo} v{version} ({len(remaining_in_group)} files)")
        print(f"{'=' * 60}")

        # Install the repo's package
        installed = install_repo(repo, version)

        for file_idx, (dataset_idx, sample) in enumerate(remaining_in_group):
            elapsed_hours = (time.time() - session_start) / 3600
            if elapsed_hours >= MAX_TOTAL_HOURS:
                break

            code_src = sample.get("code_src", "")
            code_file = sample.get("code_file", "")
            lines = len(code_src.splitlines())
            instance_id = sample.get("instance_id", f"idx_{dataset_idx}")

            # ETA
            if runtimes:
                total_remaining = sum(
                    len([i for i, s in samps if i not in done_ids])
                    for samps in repo_groups.values()
                ) - file_idx
                eta_min = (sum(runtimes) / len(runtimes) * total_remaining) / 60
                eta_str = f"ETA: {eta_min:.0f}min"
            else:
                eta_str = "ETA: calculating..."

            print(f"\n  [{file_idx + 1}/{len(remaining_in_group)}] "
                  f"{os.path.basename(code_file)} ({lines} lines) | {eta_str}")

            # Skip large files
            if lines > SKIP_LINE_LIMIT:
                print(f"   ⏭️  Skipped: {lines} lines > {SKIP_LINE_LIMIT}")
                res = {"file": os.path.basename(code_file),
                       "dataset_id": dataset_idx,
                       "instance_id": instance_id,
                       "repo": repo, "version": version,
                       "status": "skipped_large", "lines": lines}
                results.append(res)
                done_ids.add(dataset_idx)
                runtimes.append(1.0)
                with open(RESULTS_FILE, "w") as f:
                    json.dump(results, f, indent=2)
                continue

            # Setup target file
            filepath, is_installed = setup_target_file(sample, EVAL_DIR)
            if not filepath:
                print(f"   ❌ No source code for this sample")
                continue

            # Derive import_path for package files
            file_import_path = derive_import_path(code_file) if is_installed else None
            if file_import_path:
                print(f"   📦 Import path: {file_import_path}")

            # Import check — SKIP for installed package files
            # (they'll be imported via their module path, not standalone)
            if not is_installed:
                ok, err = quick_import_check(filepath)
                if not ok:
                    is_django = "django" in err.lower() or "settings" in err.lower()
                    if not is_django:
                        print(f"   ❌ Import error: {err}")
                        res = {"file": os.path.basename(code_file),
                               "dataset_id": dataset_idx,
                               "instance_id": instance_id,
                               "repo": repo, "version": version,
                               "status": "import_error",
                               "import_error": err, "lines": lines}
                        results.append(res)
                        done_ids.add(dataset_idx)
                        runtimes.append(1.0)
                        restore_original(sample)
                        with open(RESULTS_FILE, "w") as f:
                            json.dump(results, f, indent=2)
                        continue
                    else:
                        print(f"   ⚠️  Django import warning (will try): {err}")

            # Run TestMate
            print(f"   🚀 Running TestMate (max {TIMEOUT_PER_FILE // 60}min)...")
            start = time.time()

            # Capture output
            import io
            
            class TeeStream:
                def __init__(self, stream1, stream2):
                    self.stream1 = stream1
                    self.stream2 = stream2
                    self.encoding = getattr(stream1, 'encoding', 'utf-8')
                def write(self, data):
                    self.stream1.write(data)
                    self.stream2.write(data)
                def flush(self):
                    self.stream1.flush()
                    self.stream2.flush()
                def isatty(self):
                    return False
                def fileno(self):
                    return self.stream1.fileno()
                def readable(self):
                    return False
                def writable(self):
                    return True
                def seekable(self):
                    return False
                def __getattr__(self, name):
                    # Fallback: delegate any other attribute to the real stream
                    return getattr(self.stream1, name)

            output_buf = io.StringIO()
            tee = TeeStream(sys.stdout, output_buf)
            result_holder = {"success": False, "error": None}

            def _run():
                original_stdout = sys.stdout
                sys.stdout = tee
                try:
                    result_holder["success"] = autonomous_loop(
                        model, tokenizer, filepath,
                        import_path=file_import_path,
                        max_targets=MAX_TARGETS,
                    )
                except Exception as e:
                    result_holder["error"] = str(e)
                    print(f"\nERROR: {e}")
                finally:
                    sys.stdout = original_stdout

            worker = threading.Thread(target=_run, daemon=True)
            worker.start()
            worker.join(timeout=TIMEOUT_PER_FILE)
            timed_out = worker.is_alive()

            runtime = round(time.time() - start, 1)
            runtimes.append(runtime)

            # Restore original file
            restore_original(sample)

            if timed_out:
                print(f"   ⏳ Timeout after {TIMEOUT_PER_FILE // 60}min")
                res = {"file": os.path.basename(code_file),
                       "dataset_id": dataset_idx,
                       "instance_id": instance_id,
                       "repo": repo, "version": version,
                       "status": "timeout",
                       "runtime_seconds": TIMEOUT_PER_FILE, "lines": lines}
                results.append(res)
                done_ids.add(dataset_idx)
                with open(RESULTS_FILE, "w") as f:
                    json.dump(results, f, indent=2)
                continue

            # Parse results from stdout (TestMate prints to stdout)
            output = output_buf.getvalue() if hasattr(output_buf, 'getvalue') else ""
            # Also try to get from the result
            metrics = parse_output(output)
            metrics["file"] = os.path.basename(code_file)
            metrics["dataset_id"] = dataset_idx
            metrics["instance_id"] = instance_id
            metrics["repo"] = repo
            metrics["version"] = version
            metrics["lines"] = lines
            metrics["runtime_seconds"] = runtime
            metrics["runtime_minutes"] = round(runtime / 60, 1)

            if metrics["targets_total"] == 0 or metrics["targets_covered"] == 0:
                metrics["status"] = "no_tests"
            else:
                metrics["status"] = "ok"

            results.append(metrics)
            done_ids.add(dataset_idx)

            with open(RESULTS_FILE, "w") as f:
                json.dump(results, f, indent=2)

            emoji = "✅" if metrics["status"] == "ok" else "⚠️ "
            print(f"   {emoji} "
                  f"{metrics['targets_covered']}/"
                  f"{metrics['targets_total']} targets | "
                  f"mut:{metrics['mutation_score']:.0f}% | "
                  f"comp:{metrics['composite_score']:.0f}/100 | "
                  f"{metrics['runtime_minutes']}min | "
                  f"tests:{metrics['tests_generated']}")

            # Clear GPU cache
            torch.cuda.empty_cache()

        # Uninstall repo package after processing all its files
        print(f"\n   🧹 Uninstalling {repo}...")
        uninstall_repo(repo)

    # ── Final Report ──
    print_summary(results)
    print(f"\nResults saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
