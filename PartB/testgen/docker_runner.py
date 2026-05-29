"""
TestMate — Test Runner (Local + Docker)
========================================
Runs generated tests with pytest + coverage + mutation testing.
Supports both local execution and Docker-based isolated execution.

Called by gui.py to evaluate generated tests and produce metrics.
"""

import os
import sys
import re
import subprocess
import json
import shutil
import tempfile
import time

# ── Coverage data store (populated after each run) ──
# Maps filename → {covered_lines: [...], uncovered_lines: [...], source: "..."}
_coverage_data = {}


def get_coverage_data():
    """Return the per-file coverage data from the last run."""
    return _coverage_data


def _parse_coverage_json(cov_json_path: str, source_files: list) -> dict:
    """Parse coverage.json for per-line coverage data.
    
    Returns:
        Dict mapping basename → {
            covered_lines: [1, 2, 5, ...],
            uncovered_lines: [3, 4, ...],
            line_coverage_pct: float,
        }
    """
    global _coverage_data
    _coverage_data = {}
    
    if not os.path.exists(cov_json_path):
        return {}
    
    try:
        with open(cov_json_path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}
    
    files_data = data.get("files", {})
    source_basenames = {os.path.basename(sf): sf for sf in source_files}
    
    for file_path, file_info in files_data.items():
        basename = os.path.basename(file_path)
        
        # Match to our source files
        matched_source = None
        if basename in source_basenames:
            matched_source = source_basenames[basename]
        else:
            # Try matching by full path suffix
            for sb, sp in source_basenames.items():
                if file_path.endswith(sb) or sp.endswith(basename):
                    matched_source = sp
                    break
        
        if not matched_source:
            continue
        
        # Extract line data
        executed = file_info.get("executed_lines", [])
        missing = file_info.get("missing_lines", [])
        excluded = file_info.get("excluded_lines", [])
        
        # Read the source file for display
        source_text = ""
        if os.path.exists(matched_source):
            try:
                with open(matched_source, encoding="utf-8", errors="replace") as f:
                    source_text = f.read()
            except IOError:
                pass
        
        summary = file_info.get("summary", {})
        
        _coverage_data[basename] = {
            "covered_lines": sorted(executed),
            "uncovered_lines": sorted(missing),
            "excluded_lines": sorted(excluded),
            "line_coverage_pct": summary.get("percent_covered", 0.0),
            "num_statements": summary.get("num_statements", 0),
            "source": source_text,
            "source_path": matched_source,
        }
    
    return _coverage_data


def run_tests_locally(repo_dir: str, test_files: list, source_files: list,
                      cov_package: str = None) -> dict:
    """
    Run all generated test files with pytest + coverage + mutation testing.

    Args:
        repo_dir: Path to the cloned repository
        test_files: List of generated test file paths
        source_files: List of source file paths being tested
        cov_package: Package name for coverage (e.g. 'requests')

    Returns:
        Dict with: pass_rate, line_coverage, branch_coverage, mutation_score,
                   passed_tests, total_tests, errors, per_file_coverage
    """
    results = {
        "pass_rate": 0.0,
        "line_coverage": 0.0,
        "branch_coverage": 0.0,
        "mutation_score": 0.0,
        "passed_tests": 0,
        "total_tests": 0,
        "errors": [],
        "per_file_results": [],
        "per_file_coverage": {},   # basename → {line_coverage_pct, branch_coverage_pct}
        "bug_reports": [],
        "per_source_files": [],
    }

    if not test_files:
        results["errors"].append("No test files provided")
        return results

    # ── Step 1: Run pytest with coverage ──
    cov_target = cov_package or repo_dir
    cov_json_path = os.path.join(
        os.path.dirname(test_files[0]) if test_files else repo_dir,
        "coverage.json"
    )
    
    pytest_cmd = [
        sys.executable, "-m", "pytest",
        *test_files,
        "-v", "--tb=short", "--no-header",
        f"--cov={cov_target}",
        "--cov-branch",
        "--cov-report=term-missing",
        f"--cov-report=json:{cov_json_path}",
    ]

    env = os.environ.copy()
    if repo_dir:
        # Add repo root + src/ subdir (for src-layout repos like psf/requests)
        paths = [repo_dir]
        src_dir = os.path.join(repo_dir, "src")
        if os.path.isdir(src_dir):
            paths.append(src_dir)
        env["PYTHONPATH"] = os.pathsep.join(paths) + os.pathsep + env.get("PYTHONPATH", "")

    try:
        result = subprocess.run(
            pytest_cmd,
            capture_output=True, text=True, timeout=120,
            cwd=os.path.dirname(test_files[0]) if test_files else repo_dir,
            env=env,
        )
        output = result.stdout + "\n" + result.stderr
    except subprocess.TimeoutExpired:
        results["errors"].append("pytest timed out (120s)")
        return results
    except Exception as e:
        results["errors"].append(f"pytest error: {str(e)[:200]}")
        return results

    # Parse test counts
    passed_m = re.search(r'(\d+) passed', output)
    failed_m = re.search(r'(\d+) failed', output)
    error_m  = re.search(r'(\d+) error', output)

    passed = int(passed_m.group(1)) if passed_m else 0
    failed = int(failed_m.group(1)) if failed_m else 0
    errors = int(error_m.group(1))  if error_m  else 0

    total = passed + failed + errors
    results["passed_tests"] = passed
    results["total_tests"]  = total
    results["failed_tests"] = failed
    results["pass_rate"]    = round((passed / total * 100) if total > 0 else 0.0, 1)

    # Parse per-test results from pytest verbose output
    per_test = []
    for line in output.splitlines():
        if " PASSED" in line or " FAILED" in line or " ERROR" in line:
            test_match = re.match(r'^(.*?)\s+(PASSED|FAILED|ERROR)', line)
            if test_match:
                per_test.append({
                    "name":   test_match.group(1).strip(),
                    "status": test_match.group(2),
                })
    results["per_test_results"] = per_test

    # ── Step 2: Coverage from JSON (primary, accurate) ──
    # coverage.json files→<path>→summary has per-file stats.
    # totals gives the aggregate weighted across all instrumented source files.
    cov_loaded = False
    if os.path.exists(cov_json_path):
        try:
            with open(cov_json_path) as f:
                cov_json = json.load(f)

            # Per-file breakdown (only our source files)
            source_basenames = {os.path.basename(sf): sf for sf in source_files}
            total_stmts  = 0
            total_miss   = 0
            total_br     = 0
            total_br_part = 0
            per_file_cov = {}

            for file_path, file_info in cov_json.get("files", {}).items():
                basename = os.path.basename(file_path)
                if basename not in source_basenames:
                    # Try suffix match
                    matched = None
                    for sb in source_basenames:
                        if file_path.endswith(sb):
                            matched = sb
                            break
                    if not matched:
                        continue
                    basename = matched

                summary = file_info.get("summary", {})
                s  = summary.get("num_statements", 0)
                m  = summary.get("missing_lines",  0)
                b  = summary.get("num_branches",   0)
                bp = summary.get("num_partial_branches", 0)

                total_stmts   += s
                total_miss    += m
                total_br      += b
                total_br_part += bp

                lc = round((s - m) / s * 100, 1) if s > 0 else 0.0
                bc = round((b - bp) / b * 100, 1) if b > 0 else 100.0
                per_file_cov[basename] = {
                    "line_coverage_pct":   lc,
                    "branch_coverage_pct": bc,
                    "num_statements":      s,
                }

            if total_stmts > 0:
                results["line_coverage"] = round(
                    (total_stmts - total_miss) / total_stmts * 100, 1)
                cov_loaded = True
            if total_br > 0:
                results["branch_coverage"] = round(
                    (total_br - total_br_part) / total_br * 100, 1)
            elif cov_loaded:
                results["branch_coverage"] = results["line_coverage"]

            results["per_file_coverage"] = per_file_cov

        except (json.JSONDecodeError, IOError, KeyError):
            pass

    # ── Fallback: parse terminal output if JSON unavailable ──
    if not cov_loaded:
        source_basenames_list = [os.path.basename(sf) for sf in source_files]
        cov_rows = []  # list of (stmts, miss, branches, br_part)
        for line in output.splitlines():
            if "%" not in line:
                continue
            # Only look at TOTAL row for accurate aggregate
            if "TOTAL" in line:
                cols = re.findall(r'(\d+)', line.split("TOTAL")[-1])
                if len(cols) >= 4:
                    try:
                        cov_rows.append(
                            (int(cols[0]), int(cols[1]), int(cols[2]), int(cols[3]))
                        )
                    except ValueError:
                        pass
                break   # only one TOTAL row

        if cov_rows:
            ts = sum(r[0] for r in cov_rows)
            tm = sum(r[1] for r in cov_rows)
            tb = sum(r[2] for r in cov_rows)
            tbp = sum(r[3] for r in cov_rows)
            if ts > 0:
                results["line_coverage"] = round((ts - tm) / ts * 100, 1)
            if tb > 0:
                results["branch_coverage"] = round((tb - tbp) / tb * 100, 1)
            elif ts > 0:
                results["branch_coverage"] = results["line_coverage"]

    # Parse coverage.json for per-line heatmap data
    _parse_coverage_json(cov_json_path, source_files)

    # Collect error messages (failing test names + first error lines)
    error_lines = [l.strip() for l in output.splitlines()
                   if (l.strip().startswith("E ") and "Error" in l) or
                      ("FAILED" in l and "::" in l)]
    results["errors"] = error_lines[:10]

    # Build per-source-file summary row for GUI table
    source_basenames_set = {os.path.basename(sf) for sf in source_files}
    # Map test file → source file
    per_src = {}
    for t in per_test:
        test_path = t["name"].split("::")[0].replace("\\", "/")
        test_base = os.path.basename(test_path)
        # test_auth.py → auth.py
        src_base = test_base[5:] if test_base.startswith("test_") else test_base
        if src_base not in per_src:
            per_src[src_base] = {"passed": 0, "failed": 0, "error": 0}
        status = t["status"]
        if status == "PASSED":
            per_src[src_base]["passed"] += 1
        elif status == "FAILED":
            per_src[src_base]["failed"] += 1
        else:
            per_src[src_base]["error"] += 1

    # ── Read bug_reports.jsonl ──
    bugs_list = []
    if test_files:
        bug_reports_path = os.path.join(os.path.dirname(os.path.abspath(test_files[0])), "bug_reports.jsonl")
        if os.path.exists(bug_reports_path):
            bug_dict = {}
            with open(bug_reports_path) as f:
                for line in f:
                    try:
                        b = json.loads(line)
                        k = (b.get("source_file"), b.get("target"), b.get("bug_type"), b.get("evidence", ""))
                        bug_dict[k] = b
                    except:
                        pass
            bugs_list = list(bug_dict.values())
    results["bug_reports"] = bugs_list
    results["bugs_found"] = len([b for b in bugs_list if b.get("bug_type") != "mutation_survivor"])

    file_summary = []
    for src_base, counts in per_src.items():
        cov_info = results["per_file_coverage"].get(src_base, {})
        test_file_name = f"test_{src_base}"
        
        # BUGS
        file_bugs = [b for b in bugs_list if (b.get("source_file") == src_base or b.get("source_file") == src_base+".py") and b.get("bug_type") != "mutation_survivor"]
        file_mutations = [b for b in bugs_list if (b.get("source_file") == src_base or b.get("source_file") == src_base+".py") and b.get("bug_type") == "mutation_survivor"]

        file_summary.append({
            "source_file": src_base,
            "test_file":   test_file_name,
            "passed":      counts["passed"],
            "failed":      counts["failed"],
            "errors":      counts["error"],
            "total":       counts["passed"] + counts["failed"] + counts["error"],
            "line_coverage_pct":   cov_info.get("line_coverage_pct", 0.0),
            "branch_coverage_pct": cov_info.get("branch_coverage_pct", 0.0),
            "bugs": len(file_bugs),
            "mutations": len(file_mutations)
        })
    results["per_source_files"] = file_summary

    # Note: Mutation testing is handled by autonomous_loop in main.py (CLI only).
    # It is intentionally NOT run here to keep GUI evaluations fast.

    return results


# ── Docker execution ──

def _check_docker() -> bool:
    """Check if Docker is available."""
    try:
        r = subprocess.run(
            ["docker", "--version"],
            capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=10,
        )
        return r.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def run_tests_in_docker(repo_dir: str, test_files: list, source_files: list,
                        repo_name: str = "project",
                        extra_packages: list = None) -> dict:
    """
    Run generated tests inside a Docker container for clean isolation.

    Steps:
      1. Create a temp build context with Dockerfile + repo + tests
      2. docker build → docker run → capture output
      3. Parse pytest results and coverage.json from container

    Args:
        repo_dir: Path to the cloned repository
        test_files: List of generated test file paths
        source_files: List of source file paths being tested
        repo_name: Name of the package to install in Docker
        extra_packages: Additional pip packages to install

    Returns:
        Same dict format as run_tests_locally()
    """
    results = {
        "pass_rate": 0.0,
        "line_coverage": 0.0,
        "branch_coverage": 0.0,
        "mutation_score": 0.0,
        "passed_tests": 0,
        "total_tests": 0,
        "errors": [],
        "docker": True,
        "per_test_results": [],
        "bug_reports": [],
        "per_source_files": [],
    }

    if not _check_docker():
        results["errors"].append("Docker not available. Install Docker Desktop.")
        return results

    if not test_files:
        results["errors"].append("No test files provided")
        return results

    extra_packages = extra_packages or []

    # Create a temp build context
    with tempfile.TemporaryDirectory(prefix="testmate_docker_") as build_ctx:
        # Copy repo into build context
        repo_dest = os.path.join(build_ctx, "repo")
        shutil.copytree(repo_dir, repo_dest, 
                       ignore=shutil.ignore_patterns(
                           '.git', '__pycache__', '*.pyc', 'node_modules',
                           '.tox', '.eggs', 'build', 'dist',
                       ))

        # Copy test files
        tests_dest = os.path.join(build_ctx, "tests")
        os.makedirs(tests_dest, exist_ok=True)
        test_basenames = []
        for tf in test_files:
            basename = os.path.basename(tf)
            shutil.copy2(tf, os.path.join(tests_dest, basename))
            test_basenames.append(basename)

        # Detect requirements file in repo
        req_file = None
        for candidate in ["requirements.txt", "requirements-dev.txt", "setup.py",
                          "pyproject.toml"]:
            if os.path.exists(os.path.join(repo_dest, candidate)):
                req_file = candidate
                break

        # Write Dockerfile
        pip_extras = " ".join(extra_packages)
        install_lines = []
        install_lines.append("RUN pip install --no-cache-dir pytest pytest-cov")
        if pip_extras:
            install_lines.append(f"RUN pip install --no-cache-dir {pip_extras}")
        
        # Install the repo itself
        if os.path.exists(os.path.join(repo_dest, "setup.py")) or \
           os.path.exists(os.path.join(repo_dest, "pyproject.toml")):
            install_lines.append("RUN pip install --no-cache-dir -e /workspace/repo 2>/dev/null || true")
        elif req_file and "requirements" in req_file:
            install_lines.append(
                f"RUN pip install --no-cache-dir -r /workspace/repo/{req_file} 2>/dev/null || true"
            )

        install_block = "\n".join(install_lines)
        # Build the test command: list all test files
        test_cmd_files = " ".join(f"/workspace/tests/{tb}" for tb in test_basenames)
        
        # Determine coverage source
        if repo_name and os.path.exists(os.path.join(repo_dir, repo_name)):
            cov_source = f"/workspace/repo/{repo_name}"
        else:
            cov_source = "/workspace/repo"

        dockerfile_content = f"""\
FROM python:3.11-slim
WORKDIR /workspace
COPY repo/ /workspace/repo/
COPY tests/ /workspace/tests/
{install_block}
CMD pytest {test_cmd_files} -v --tb=short \\
    --cov={cov_source} --cov-branch \\
    --cov-report=term-missing \\
    --cov-report=json:/workspace/coverage.json
"""
        with open(os.path.join(build_ctx, "Dockerfile"), "w") as f:
            f.write(dockerfile_content)

        image_tag = f"testmate-eval-{repo_name}".lower().replace("/", "-")

        # ── Build ──
        print(f"   🐳 Building Docker image: {image_tag}...")
        build_result = subprocess.run(
            ["docker", "build", "-t", image_tag, "."],
            capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=300,
            cwd=build_ctx,
        )
        if build_result.returncode != 0:
            err_msg = build_result.stderr[-500:] if build_result.stderr else "Unknown build error"
            results["errors"].append(f"Docker build failed: {err_msg}")
            return results

        print(f"   ✅ Image built: {image_tag}")

        # ── Run ──
        print("   🚀 Running tests in Docker container...")
        
        # Mount a volume for coverage.json output
        cov_output_dir = os.path.join(build_ctx, "output")
        os.makedirs(cov_output_dir, exist_ok=True)

        # Clean up any stale container from a previous run
        subprocess.run(["docker", "rm", "-f", f"{image_tag}-run"], capture_output=True)

        run_result = subprocess.run(
            [
                "docker", "run", "--name", f"{image_tag}-run",
                image_tag,
                "bash", "-c",
                f"pytest {test_cmd_files} -v --tb=short "
                f"--cov={cov_source} --cov-branch "
                f"--cov-report=term-missing "
                f"--cov-report=json:/workspace/coverage.json "
                f"&& echo '___DOCKER_EXIT_0___' || echo '___DOCKER_EXIT_1___'"
            ],
            capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=300,
        )
        
        # Safely extract coverage.json avoiding strict Windows volume mounting failures
        subprocess.run(["docker", "cp", f"{image_tag}-run:/workspace/coverage.json", os.path.join(cov_output_dir, "coverage.json")], capture_output=True)
        subprocess.run(["docker", "rm", "-f", f"{image_tag}-run"], capture_output=True)
        output = run_result.stdout + "\n" + run_result.stderr
        print(output)

        # Parse test results from output
        passed_m = re.search(r'(\d+) passed', output)
        failed_m = re.search(r'(\d+) failed', output)
        error_m = re.search(r'(\d+) error', output)

        passed = int(passed_m.group(1)) if passed_m else 0
        failed = int(failed_m.group(1)) if failed_m else 0
        errs = int(error_m.group(1)) if error_m else 0

        total = passed + failed + errs
        results["passed_tests"] = passed
        results["total_tests"] = total
        results["failed_tests"] = failed
        results["pass_rate"] = round((passed / total * 100) if total > 0 else 0.0, 1)

        # Parse per-test results
        per_test = []
        for line in output.splitlines():
            if " PASSED" in line or " FAILED" in line or " ERROR" in line:
                test_match = re.match(r'^(.*?)\s+(PASSED|FAILED|ERROR)', line)
                if test_match:
                    per_test.append({
                        "name": test_match.group(1).strip(),
                        "status": test_match.group(2),
                    })
        results["per_test_results"] = per_test

        # ── Coverage from JSON (primary) ──
        cov_json_file = os.path.join(cov_output_dir, "coverage.json")
        cov_loaded = False
        per_file_cov = {}
        if os.path.exists(cov_json_file):
            try:
                with open(cov_json_file) as f:
                    cov_data_raw = json.load(f)
                source_basenames = {os.path.basename(sf): sf for sf in source_files}
                total_stmts = total_miss = total_br = total_br_part = 0
                for file_path, file_info in cov_data_raw.get("files", {}).items():
                    basename = os.path.basename(file_path)
                    if basename not in source_basenames:
                        matched = None
                        for sb in source_basenames:
                            if file_path.endswith(sb):
                                matched = sb
                                break
                        if not matched:
                            continue
                        basename = matched
                    summary = file_info.get("summary", {})
                    s  = summary.get("num_statements", 0)
                    m  = summary.get("missing_lines",  0)
                    b  = summary.get("num_branches",   0)
                    bp = summary.get("num_partial_branches", 0)
                    total_stmts += s; total_miss += m
                    total_br += b;    total_br_part += bp
                    lc = round((s - m) / s * 100, 1) if s > 0 else 0.0
                    bc = round((b - bp) / b * 100, 1) if b > 0 else 100.0
                    per_file_cov[basename] = {
                        "line_coverage_pct": lc, "branch_coverage_pct": bc, "num_statements": s}
                if total_stmts > 0:
                    results["line_coverage"] = round(
                        (total_stmts - total_miss) / total_stmts * 100, 1)
                    cov_loaded = True
                if total_br > 0:
                    results["branch_coverage"] = round(
                        (total_br - total_br_part) / total_br * 100, 1)
                elif cov_loaded:
                    results["branch_coverage"] = results["line_coverage"]
                results["per_file_coverage"] = per_file_cov
            except (json.JSONDecodeError, IOError, KeyError):
                pass

        # ── Fallback: TOTAL line from terminal ──
        if not cov_loaded:
            for line in output.splitlines():
                if "%" in line and "TOTAL" in line:
                    cols = re.findall(r'(\d+)', line.split("TOTAL")[-1])
                    if len(cols) >= 4:
                        try:
                            s, m, b, bp = int(cols[0]), int(cols[1]), int(cols[2]), int(cols[3])
                            if s > 0:
                                results["line_coverage"] = round((s - m) / s * 100, 1)
                            if b > 0:
                                results["branch_coverage"] = round((b - bp) / b * 100, 1)
                            elif s > 0:
                                results["branch_coverage"] = results["line_coverage"]
                        except (ValueError, ZeroDivisionError):
                            pass
                    break

        # Parse coverage.json from container output for heatmap
        if os.path.exists(cov_json_file):
            _parse_coverage_json(cov_json_file, source_files)

        # Collect errors
        error_lines = [l.strip() for l in output.splitlines()
                       if (l.strip().startswith("E ") and "Error" in l) or
                          ("FAILED" in l and "::" in l)]
        results["errors"] = error_lines[:10]

        # ── Read bug_reports.jsonl ──
        bugs_list = []
        if test_files:
            bug_reports_path = os.path.join(os.path.dirname(os.path.abspath(test_files[0])), "bug_reports.jsonl")
            if os.path.exists(bug_reports_path):
                bug_dict = {}
                with open(bug_reports_path) as f:
                    for line in f:
                        try:
                            b = json.loads(line)
                            k = (b.get("source_file"), b.get("target"), b.get("bug_type"), b.get("evidence", ""))
                            bug_dict[k] = b
                        except:
                            pass
                bugs_list = list(bug_dict.values())
        results["bug_reports"] = bugs_list
        results["bugs_found"] = len([b for b in bugs_list if b.get("bug_type") != "mutation_survivor"])

        # Build per-source-file summary
        per_src = {}
        for t in per_test:
            test_base = os.path.basename(t["name"].split("::")[0])
            src_base = test_base[5:] if test_base.startswith("test_") else test_base
            if src_base not in per_src:
                per_src[src_base] = {"passed": 0, "failed": 0, "error": 0}
            if t["status"] == "PASSED":
                per_src[src_base]["passed"] += 1
            elif t["status"] == "FAILED":
                per_src[src_base]["failed"] += 1
            else:
                per_src[src_base]["error"] += 1
        file_summary = []
        for src_base, counts in per_src.items():
            ci = results.get("per_file_coverage", {}).get(src_base, {})
            # BUGS
            file_bugs = [b for b in bugs_list if (b.get("source_file") == src_base or b.get("source_file") == src_base+".py") and b.get("bug_type") != "mutation_survivor"]
            file_mutations = [b for b in bugs_list if (b.get("source_file") == src_base or b.get("source_file") == src_base+".py") and b.get("bug_type") == "mutation_survivor"]
            file_summary.append({
                "source_file": src_base,
                "test_file":   f"test_{src_base}",
                "passed":      counts["passed"],
                "failed":      counts["failed"],
                "errors":      counts["error"],
                "total":       counts["passed"] + counts["failed"] + counts["error"],
                "line_coverage_pct":   ci.get("line_coverage_pct", 0.0),
                "branch_coverage_pct": ci.get("branch_coverage_pct", 0.0),
                "bugs": len(file_bugs),
                "mutations": len(file_mutations)
            })
        results["per_source_files"] = file_summary

        # Cleanup image
        subprocess.run(
            ["docker", "rmi", image_tag, "-f"],
            capture_output=True, timeout=30,
        )

    return results


def docker_deep_scan(repo_dir: str, repo_name: str = None,
                     source_files: list = None) -> dict:
    """
    Optional Docker-based pre-scan before test generation.
    
    Runs in a lightweight container to:
      1. Install all requirements (requirements.txt / setup.py)
      2. Verify which source file imports actually work
      3. Run pytest --collect-only to discover existing test patterns
      4. Check available fixtures from conftest.py
    
    Args:
        repo_dir: Path to the cloned repository
        repo_name: Package name (e.g., 'requests')
        source_files: List of source file paths to check imports for
    
    Returns:
        Dict with: working_imports, failed_imports, existing_tests,
                   existing_fixtures, installed_packages
    """
    scan_result = {
        "working_imports": [],
        "failed_imports": [],
        "existing_tests": [],
        "existing_fixtures": [],
        "installed_packages": [],
        "test_patterns": [],  # e.g., ["pytest", "unittest", "fixtures"]
        "errors": [],
    }

    if not _check_docker():
        scan_result["errors"].append("Docker not available")
        return scan_result

    source_files = source_files or []

    with tempfile.TemporaryDirectory(prefix="testmate_scan_") as build_ctx:
        # Copy repo
        repo_dest = os.path.join(build_ctx, "repo")
        shutil.copytree(repo_dir, repo_dest,
                       ignore=shutil.ignore_patterns(
                           '.git', '__pycache__', '*.pyc', 'node_modules',
                           '.tox', '.eggs', 'build', 'dist',
                       ))

        # Build import check script
        import_checks = []
        for sf in source_files:
            basename = os.path.basename(sf)
            module = basename.replace(".py", "")
            # Try both package import and direct import
            if repo_name:
                # Find relative path within repo
                rel = os.path.relpath(sf, repo_dir).replace(os.sep, "/")
                import_path = rel.replace("/", ".").replace(".py", "")
                import_checks.append(
                    f"try:\n"
                    f"    import {import_path}\n"
                    f"    print('OK:{basename}')\n"
                    f"except Exception as e:\n"
                    f"    print(f'FAIL:{basename}:{{e}}')\n"
                )
            else:
                import_checks.append(
                    f"try:\n"
                    f"    import {module}\n"
                    f"    print('OK:{basename}')\n"
                    f"except Exception as e:\n"
                    f"    print(f'FAIL:{basename}:{{e}}')\n"
                )

        import_script = "\n".join(import_checks)

        # Write the scan script
        scan_script = f"""\
import subprocess, sys, json

# 1. List installed packages
result = subprocess.run([sys.executable, '-m', 'pip', 'list', '--format=columns'],
                       capture_output=True, text=True)
print('___PACKAGES_START___')
print(result.stdout)
print('___PACKAGES_END___')

# 2. Check imports
print('___IMPORTS_START___')
{import_script}
print('___IMPORTS_END___')

# 3. Collect existing tests
print('___TESTS_START___')
result = subprocess.run(
    [sys.executable, '-m', 'pytest', '--collect-only', '-q', '/workspace/repo'],
    capture_output=True, text=True, timeout=30
)
print(result.stdout[:3000])
print('___TESTS_END___')

# 4. Check conftest fixtures
print('___FIXTURES_START___')
result = subprocess.run(
    [sys.executable, '-m', 'pytest', '--fixtures', '-q', '/workspace/repo'],
    capture_output=True, text=True, timeout=30
)
# Extract just fixture names (lines that don't start with whitespace)
for line in result.stdout.splitlines()[:50]:
    stripped = line.strip()
    if stripped and not stripped.startswith('-') and ' -- ' in stripped:
        print(stripped.split(' -- ')[0].strip())
print('___FIXTURES_END___')
"""
        scan_script_path = os.path.join(build_ctx, "scan.py")
        with open(scan_script_path, "w") as f:
            f.write(scan_script)

        # Detect install method
        install_lines = ["RUN pip install --no-cache-dir pytest 2>/dev/null || true"]
        if os.path.exists(os.path.join(repo_dest, "setup.py")) or \
           os.path.exists(os.path.join(repo_dest, "pyproject.toml")):
            install_lines.append(
                "RUN pip install --no-cache-dir -e /workspace/repo 2>/dev/null || true"
            )
        for req in ["requirements.txt", "requirements-dev.txt"]:
            if os.path.exists(os.path.join(repo_dest, req)):
                install_lines.append(
                    f"RUN pip install --no-cache-dir -r /workspace/repo/{req} 2>/dev/null || true"
                )
        if repo_name:
            install_lines.append(
                f"RUN pip install --no-cache-dir {repo_name} 2>/dev/null || true"
            )

        install_block = "\n".join(install_lines)

        dockerfile = f"""\
FROM python:3.11-slim
WORKDIR /workspace
COPY repo/ /workspace/repo/
COPY scan.py /workspace/scan.py
{install_block}
CMD python /workspace/scan.py
"""
        with open(os.path.join(build_ctx, "Dockerfile"), "w") as f:
            f.write(dockerfile)

        image_tag = "testmate-scan"

        # Build
        print("   🔍 Docker deep scan: building scan image...")
        build_r = subprocess.run(
            ["docker", "build", "-t", image_tag, "."],
            capture_output=True, text=True, timeout=300,
            cwd=build_ctx,
            encoding="utf-8", errors="replace",
        )
        if build_r.returncode != 0:
            scan_result["errors"].append(f"Scan build failed: {build_r.stderr[-300:]}")
            return scan_result

        # Run
        print("   🔍 Docker deep scan: scanning repo...")
        run_r = subprocess.run(
            ["docker", "run", "--rm", image_tag],
            capture_output=True, text=True, timeout=120,
            encoding="utf-8", errors="replace",
        )
        output = run_r.stdout

        # Parse results
        # Packages
        pkg_match = _extract_section(output, "___PACKAGES_START___", "___PACKAGES_END___")
        if pkg_match:
            for line in pkg_match.splitlines()[2:]:  # Skip header lines
                parts = line.split()
                if len(parts) >= 1:
                    scan_result["installed_packages"].append(parts[0])

        # Imports
        imp_match = _extract_section(output, "___IMPORTS_START___", "___IMPORTS_END___")
        if imp_match:
            for line in imp_match.splitlines():
                if line.startswith("OK:"):
                    scan_result["working_imports"].append(line[3:])
                elif line.startswith("FAIL:"):
                    scan_result["failed_imports"].append(line[5:])

        # Tests
        test_match = _extract_section(output, "___TESTS_START___", "___TESTS_END___")
        if test_match:
            for line in test_match.splitlines():
                if "::" in line and "test" in line.lower():
                    scan_result["existing_tests"].append(line.strip())
            # Detect patterns
            if "unittest.TestCase" in test_match or "self.assert" in test_match:
                scan_result["test_patterns"].append("unittest")
            if "def test_" in test_match:
                scan_result["test_patterns"].append("pytest")

        # Fixtures
        fix_match = _extract_section(output, "___FIXTURES_START___", "___FIXTURES_END___")
        if fix_match:
            for line in fix_match.splitlines():
                stripped = line.strip()
                if stripped and not stripped.startswith("_"):
                    scan_result["existing_fixtures"].append(stripped)

        # Cleanup
        subprocess.run(
            ["docker", "rmi", image_tag, "-f"],
            capture_output=True, timeout=30,
        )

        print(f"   ✅ Deep scan done: "
              f"{len(scan_result['working_imports'])} imports OK, "
              f"{len(scan_result['failed_imports'])} failed, "
              f"{len(scan_result['existing_tests'])} existing tests found")

    return scan_result


def _extract_section(text: str, start_marker: str, end_marker: str) -> str:
    """Extract text between two markers."""
    start_idx = text.find(start_marker)
    end_idx = text.find(end_marker)
    if start_idx == -1 or end_idx == -1:
        return ""
    return text[start_idx + len(start_marker):end_idx].strip()


# ──────────────────────────────────────────────────────────────────────────
# Inner-loop Docker helpers (used by autonomous_loop via run_pytest)
# ──────────────────────────────────────────────────────────────────────────

import hashlib

# Per-process cache of image tags we've already verified exist.
# Key: (repo_dir_realpath, requirements_hash) -> image tag.
_IMAGE_CACHE: dict[tuple, str] = {}


def _hash_requirements(repo_dir: str) -> str:
    """Hash whatever dependency manifest exists in the repo.
    Image-cache key changes whenever the manifest does, so users get a fresh
    image after editing requirements.txt without us having to track timestamps."""
    h = hashlib.sha256()
    for cand in ("requirements.txt", "requirements-dev.txt",
                 "pyproject.toml", "setup.py", "setup.cfg"):
        p = os.path.join(repo_dir, cand)
        if os.path.exists(p):
            try:
                h.update(cand.encode())
                with open(p, "rb") as f:
                    h.update(f.read())
            except OSError:
                continue
    return h.hexdigest()[:12]


def _image_exists(tag: str) -> bool:
    try:
        r = subprocess.run(
            ["docker", "image", "inspect", tag],
            capture_output=True, text=True, timeout=10,
        )
        return r.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def ensure_repo_image(repo_dir: str, repo_name: str = "project") -> str:
    """Build (or reuse) a Docker image that has the repo + its deps installed.

    Returns the image tag.  Subsequent inner-loop pytest calls bind-mount the
    test file at runtime, so this image is reused for every retry — image
    builds are not paid per-iteration.

    Raises RuntimeError if Docker is unavailable or build fails.
    """
    if not _check_docker():
        raise RuntimeError("Docker is not available — is Docker Desktop running?")

    repo_dir = os.path.realpath(repo_dir)
    cache_key = (repo_dir, _hash_requirements(repo_dir))
    if cache_key in _IMAGE_CACHE:
        tag = _IMAGE_CACHE[cache_key]
        if _image_exists(tag):
            return tag  # still cached on disk

    # Stable per-repo tag — survives across processes (docker checks disk).
    tag_seed = hashlib.sha256(f"{repo_dir}|{cache_key[1]}".encode()).hexdigest()[:12]
    safe_name = re.sub(r"[^a-z0-9_-]+", "-", repo_name.lower())[:30] or "project"
    image_tag = f"testmate-repo-{safe_name}-{tag_seed}"

    if _image_exists(image_tag):
        _IMAGE_CACHE[cache_key] = image_tag
        return image_tag

    print(f"   🐳 Building Docker image for {repo_name} (one-time)...")
    with tempfile.TemporaryDirectory(prefix="testmate_repoimg_") as build_ctx:
        repo_dest = os.path.join(build_ctx, "repo")
        shutil.copytree(repo_dir, repo_dest,
                        ignore=shutil.ignore_patterns(
                            '.git', '__pycache__', '*.pyc', 'node_modules',
                            '.tox', '.eggs', 'build', 'dist',
                        ))

        install_lines = ["RUN pip install --no-cache-dir pytest pytest-cov"]
        if os.path.exists(os.path.join(repo_dest, "pyproject.toml")) or \
           os.path.exists(os.path.join(repo_dest, "setup.py")):
            install_lines.append(
                "RUN pip install --no-cache-dir -e /workspace/repo 2>/dev/null || true"
            )
        for req in ("requirements.txt", "requirements-dev.txt"):
            if os.path.exists(os.path.join(repo_dest, req)):
                install_lines.append(
                    f"RUN pip install --no-cache-dir -r /workspace/repo/{req} 2>/dev/null || true"
                )
        install_block = "\n".join(install_lines)

        dockerfile = (
            "FROM python:3.11-slim\n"
            "WORKDIR /workspace\n"
            "COPY repo/ /workspace/repo/\n"
            f"{install_block}\n"
            'CMD ["python", "-c", "print(\\"image ready\\")"]\n'
        )
        with open(os.path.join(build_ctx, "Dockerfile"), "w", encoding="utf-8") as f:
            f.write(dockerfile)

        r = subprocess.run(
            ["docker", "build", "-t", image_tag, build_ctx],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=600,
        )
        if r.returncode != 0:
            raise RuntimeError(f"Docker build failed: {r.stderr[-500:]}")

    _IMAGE_CACHE[cache_key] = image_tag
    print(f"   ✓ Image built: {image_tag}")
    return image_tag


def run_pytest_in_image(
    image_tag: str,
    test_file: str,
    target_file: str,
    *,
    with_coverage: bool = False,
    cov_module: str | None = None,
    deadline_ts: float | None = None,
) -> tuple:
    """Run pytest inside the cached repo image.  Test file is bind-mounted at
    runtime so it can change between retries without rebuilding.

    Returns:
        with_coverage=False: (passed: bool, output: str)
        with_coverage=True:  (passed, output, line_coverage_pct, branch_coverage_pct, covered_lines_set)

    On any container-level failure returns the same shape with passed=False.
    """
    test_dir = os.path.dirname(os.path.abspath(test_file))
    test_basename = os.path.basename(test_file)

    timeout = 30 if not with_coverage else 60
    if deadline_ts is not None:
        timeout = max(5, min(timeout, int(deadline_ts - time.time()) - 2))

    cmd = [
        "docker", "run", "--rm",
        "-v", f"{test_dir}:/tests:ro",
        "-w", "/workspace",
        image_tag,
        "pytest", f"/tests/{test_basename}", "-v", "--tb=short", "--no-header",
    ]
    if with_coverage:
        cmd += [
            f"--cov={cov_module or '/workspace/repo'}",
            "--cov-branch",
            "--cov-report=term-missing",
            "--cov-report=json:/tmp/cov.json",
        ]

    try:
        r = subprocess.run(
            cmd, capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        if with_coverage:
            return False, "Docker pytest timed out", 0.0, 0.0, set()
        return False, "Docker pytest timed out"

    output = (r.stdout or "") + "\n" + (r.stderr or "")
    passed = r.returncode == 0

    if not with_coverage:
        return passed, output

    # Best-effort: try to extract coverage from the terminal output (the cov
    # JSON is inside the container; copying it back would add another docker
    # cp call per iteration, which is expensive).  Terminal-table parse is
    # cheap and good enough for the inner loop's coverage estimate; the final
    # accurate coverage still comes from the post-loop run_pytest_with_coverage.
    line_cov = 0.0
    branch_cov = 0.0
    for raw in output.splitlines():
        s = raw.strip()
        if "TOTAL" in s:
            cols = s.split()
            try:
                # TOTAL  Stmts  Miss  Branch  BrPart  Cover
                stmts = int(cols[1]); miss = int(cols[2])
                branches = int(cols[3]); br_part = int(cols[4])
                if stmts > 0:
                    line_cov = round((stmts - miss) / stmts * 100, 1)
                if branches > 0:
                    branch_cov = round((branches - br_part) / branches * 100, 1)
            except (ValueError, IndexError):
                m = re.search(r"(\d+)%", s)
                if m:
                    line_cov = float(m.group(1))
            break
    return passed, output, line_cov, branch_cov, set()
