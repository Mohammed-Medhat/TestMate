"""
bug_to_partc.py — PartB → PartC repair bridge.

After post_run_audit() writes confirmed/suspected bugs to bug_reports.jsonl,
this module reads those reports, groups them by source file, and invokes
PartC's repair pipeline for each file.

PartC's success/failure becomes the final verdict:
  confirmed + repair OK    → keep confirmed   (with patch attached)
  confirmed + repair FAIL  → demote to suspected
  suspected + repair OK    → promote to confirmed
  suspected + repair FAIL  → drop to discarded

Artifacts written:
  <testgen_dir>/outputs/patched/<filename>.py     (always)
  git branch 'testmate-repair-<timestamp>'        (if .git exists and auto_git=True)
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import shutil
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

_PARTC_ROOT = Path(__file__).parent.parent.parent / "PartC"
_PARTC_API  = _PARTC_ROOT / "api"

_VERIFY_TIMEOUT = 60   # seconds for post-repair test verification


# ── Helpers ───────────────────────────────────────────────────────────────────

def _log(msg: str, callback: Optional[Callable] = None, level: str = "info"):
    getattr(logger, level)(msg)
    if callback:
        callback("log", level, msg)


def _emit(event: str, data: Any, callback: Optional[Callable] = None):
    if callback:
        callback(event, data)


def _read_bug_reports(path: str) -> list[dict]:
    """Read all entries from bug_reports.jsonl for this run."""
    reports = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        reports.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    except FileNotFoundError:
        pass
    return reports


def _rewrite_bug_reports(path: str, reports: list[dict]):
    """Overwrite bug_reports.jsonl with updated entries."""
    with open(path, "w", encoding="utf-8") as f:
        for r in reports:
            f.write(json.dumps(r) + "\n")


def _group_by_source_file(reports: list[dict]) -> dict[str, list[dict]]:
    """Group bug reports by source_file, only including repairable verdicts."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in reports:
        if r.get("verdict") in ("confirmed", "suspected") and r.get("test_file"):
            src = r.get("source_file", "")
            if src:
                groups[src].append(r)
    return dict(groups)


def _find_source_abs_path(source_filename: str, repo_dir: str) -> Optional[str]:
    """Walk repo_dir to find the absolute path for a bare filename."""
    for root, _, files in os.walk(repo_dir):
        if source_filename in files:
            return os.path.join(root, source_filename)
    return None


def _verify_tests(test_files: list[str], cwd: str, timeout: int = _VERIFY_TIMEOUT) -> tuple[bool, str]:
    """Run pytest on test_files, return (passed, output)."""
    if not test_files:
        return True, ""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", *test_files, "-q", "--tb=short", "--no-header"],
            capture_output=True, text=True, timeout=timeout, cwd=cwd,
        )
        return result.returncode == 0, result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return False, "Timeout during verification"
    except Exception as e:
        return False, str(e)


def _write_patched_file(source_abs: str, patched_content: str, outputs_dir: Path) -> str:
    """Write patched source to outputs/patched/<filename>.py and return the path."""
    outputs_dir.mkdir(parents=True, exist_ok=True)
    dest = outputs_dir / Path(source_abs).name
    dest.write_text(patched_content, encoding="utf-8")
    return str(dest)


def _try_git_branch(repo_dir: str, patch_path: str, source_abs: str, timestamp: str) -> Optional[str]:
    """
    If repo_dir is a git repository, create a branch 'testmate-repair-<timestamp>',
    copy the patch onto the source path, and commit it.
    Returns branch name on success, None otherwise.
    """
    git_dir = Path(repo_dir)
    while git_dir != git_dir.parent:
        if (git_dir / ".git").exists():
            break
        git_dir = git_dir.parent
    else:
        return None  # no .git found

    branch = f"testmate-repair-{timestamp}"
    try:
        # Check working tree is not in a broken state
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, cwd=str(git_dir), timeout=10,
        )
        if status.returncode != 0:
            return None

        # Save original content BEFORE overwriting — git checkout will delete
        # the file when switching back if it was untracked on the original branch.
        original_content = Path(source_abs).read_bytes() if Path(source_abs).exists() else None

        subprocess.run(["git", "checkout", "-b", branch],
                       check=True, capture_output=True, cwd=str(git_dir), timeout=15)
        shutil.copy2(patch_path, source_abs)
        subprocess.run(["git", "add", source_abs],
                       check=True, capture_output=True, cwd=str(git_dir), timeout=10)
        subprocess.run(
            ["git", "commit", "-m", f"TestMate: auto-repair {Path(source_abs).name}"],
            check=True, capture_output=True, cwd=str(git_dir), timeout=15,
        )
        # Switch back to original branch
        subprocess.run(["git", "checkout", "-"],
                       capture_output=True, cwd=str(git_dir), timeout=10)

        # Restore original file — git checkout removes untracked files added on the branch
        if original_content is not None and not Path(source_abs).exists():
            Path(source_abs).write_bytes(original_content)
            logger.info("Restored original %s after git branch switch", Path(source_abs).name)

        return branch
    except Exception as e:
        logger.warning("git branch creation failed: %s", e)
        try:
            subprocess.run(["git", "checkout", "-"],
                           capture_output=True, cwd=str(git_dir), timeout=10)
        except Exception:
            pass
        return None


def _swap_adapter(model: Any, to: str) -> bool:
    """
    Hot-swap PEFT adapter by name.
    If the model was loaded without named adapters (e.g. by main.py's load_model()),
    try to load the PartC adapter on top and give it the target name.
    Returns True if successful.
    """
    # Fast path: named adapter already exists
    try:
        model.set_adapter(to)
        logger.info("Swapped to adapter: %s", to)
        return True
    except Exception:
        pass

    # Slow path: adapter not yet loaded by name — load it now
    if to == "partc":
        partc_lora = _PARTC_ROOT / "models" / "adapter"
        weights_ok = (
            (partc_lora / "adapter_model.safetensors").exists()
            or (partc_lora / "adapter_model.bin").exists()
        )
        if not partc_lora.exists() or not weights_ok:
            logger.warning("PartC adapter weights not found at %s — repair skipped.", partc_lora)
            return False
        try:
            partc_path = partc_lora.resolve().as_posix()
            model.load_adapter(partc_path, adapter_name="partc")
            model.set_adapter("partc")
            logger.info("Loaded and activated PartC adapter from %s", partc_lora)
            return True
        except Exception as e:
            logger.warning("PartC adapter load failed: %s", e)
            return False

    logger.warning("Adapter swap to '%s' failed — no fallback available", to)
    return False


# ── Main bridge ───────────────────────────────────────────────────────────────

def repair_pending_bugs(
    bug_reports_path: str,
    repo_dir: str,
    model: Any,
    tokenizer: Any,
    log_callback: Optional[Callable] = None,
    auto_git: bool = True,
) -> list[dict]:
    """
    Read bug_reports.jsonl, group confirmed/suspected bugs by source file,
    run PartC's repair on each file, update verdicts, write artifacts.

    Args:
        bug_reports_path: Absolute path to bug_reports.jsonl written by post_run_audit.
        repo_dir:         Root of the project/repo being tested.
        model:            Pre-loaded Qwen model (PartB adapter active).
        tokenizer:        Pre-loaded tokenizer.
        log_callback:     SSE event callback(event_type, *args).
        auto_git:         If True, attempt git branch creation.

    Returns:
        List of repair result dicts (one per source file attempted).
    """
    sys.path.insert(0, str(_PARTC_API))
    sys.path.insert(0, str(_PARTC_ROOT))

    try:
        from partc_api import execute_part_c  # type: ignore[import]
    except ImportError as e:
        _log(f"  PartC not available — skipping repair ({e})", log_callback, "warning")
        return []

    all_reports = _read_bug_reports(bug_reports_path)
    groups = _group_by_source_file(all_reports)

    if not groups:
        _log("  No confirmed/suspected bugs with test files — nothing to repair.", log_callback)
        return []

    total_files = len(groups)
    _log(f"  Bug Repair: {total_files} source file(s) queued", log_callback)
    _emit("repair_start", {"total_files": total_files}, log_callback)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    outputs_dir = Path(bug_reports_path).parent / "patched"
    all_repair_results: list[dict] = []

    # ── Swap to PartC adapter ────────────────────────────────────────────
    swapped = _swap_adapter(model, "partc")
    if not swapped:
        _log("  Could not swap to PartC adapter — repair skipped.", log_callback, "warning")
        return []

    try:
        for source_filename, bugs in groups.items():
            _log(f"\n  Repairing: {source_filename} ({len(bugs)} bug(s))", log_callback)
            _emit("repair_file_start", {"file": source_filename, "num_bugs": len(bugs)}, log_callback)

            # Locate absolute source path
            source_abs = _find_source_abs_path(source_filename, repo_dir)
            if not source_abs or not os.path.isfile(source_abs):
                # Try from bug report directly
                for b in bugs:
                    tf = b.get("test_file", "")
                    candidate = os.path.join(os.path.dirname(tf), source_filename)
                    if os.path.isfile(candidate):
                        source_abs = candidate
                        break

            if not source_abs:
                _log(f"    Could not locate {source_filename} — skipping", log_callback, "warning")
                continue

            # Collect unique test files from all bugs for this source
            test_files = list({b["test_file"] for b in bugs if b.get("test_file") and os.path.isfile(b["test_file"])})
            if not test_files:
                _log(f"    No valid test files for {source_filename} — skipping", log_callback, "warning")
                continue

            # Use the first (primary) test file; SBFL will benefit from all failing tests in it
            primary_test = test_files[0]
            cwd = os.path.dirname(source_abs)

            # ── Run PartC repair ────────────────────────────────────────
            t0 = time.time()
            repair_result = execute_part_c(
                source_file=source_abs,
                test_file=primary_test,
                model=model,
                tokenizer=tokenizer,
                max_attempts=3,
                log_callback=log_callback,
            )
            elapsed = round(time.time() - t0, 1)

            repair_success = repair_result.get("success", False)
            patched_content = repair_result.get("patched", "")

            _log(
                f"    PartC: {'patched in' if repair_success else 'failed after'} {elapsed}s",
                log_callback,
            )
            _emit("repair_attempt", {
                "file":    source_filename,
                "success": repair_success,
                "elapsed": elapsed,
                "attempts": repair_result.get("attempts", []),
            }, log_callback)

            # ── Verification ────────────────────────────────────────────
            # execute_part_c already verifies inside the workspace (isolated copy).
            # If it reports success=True, the tests passed there — trust it.
            # We do NOT re-run against the original directory because the original
            # source file is still the buggy version (workspace is isolated).
            if repair_success and patched_content:
                repair_result["verify_pass"] = True
                repair_result["verify_output"] = "Verified inside PartC workspace"
                _log(f"    Verification: passed inside repair workspace", log_callback)
            else:
                repair_result["verify_pass"] = False
                repair_result["verify_output"] = ""

            # ── Write artifacts ─────────────────────────────────────────
            patched_path: Optional[str] = None
            git_branch: Optional[str] = None

            if repair_success and patched_content:
                patched_path = _write_patched_file(source_abs, patched_content, outputs_dir)
                _log(f"    Patch saved → {os.path.relpath(patched_path, repo_dir)}", log_callback)

                if auto_git:
                    git_branch = _try_git_branch(repo_dir, patched_path, source_abs, timestamp)
                    if git_branch:
                        _log(f"    Git branch: {git_branch}", log_callback)

            # ── Update verdicts in bug_reports ──────────────────────────
            verdicts_updated: list[dict] = []
            for bug in bugs:
                old_verdict = bug.get("verdict", "suspected")
                if repair_success:
                    new_verdict = "confirmed"   # PartC proved it — always confirmed now
                else:
                    new_verdict = "suspected" if old_verdict == "confirmed" else "discarded"

                if old_verdict != new_verdict:
                    _log(f"    Verdict: {bug['target']} {old_verdict} → {new_verdict}", log_callback)
                    _emit("verdict_update", {
                        "target":      bug.get("target"),
                        "old_verdict": old_verdict,
                        "new_verdict": new_verdict,
                    }, log_callback)

                verdicts_updated.append({"target": bug.get("target"),
                                         "old": old_verdict, "new": new_verdict})

                # Mutate in-place so the rewrite below has updated values
                bug["verdict"] = new_verdict
                bug["repair"] = {
                    "attempted":    True,
                    "success":      repair_success,
                    "elapsed_sec":  elapsed,
                    "patched_path": patched_path,
                    "git_branch":   git_branch,
                    "verify_pass":  repair_result.get("verify_pass", False),
                    "attempts":     repair_result.get("attempts", []),
                    "suspicious":   repair_result.get("suspicious", []),
                }

            # ── Persist updated reports ─────────────────────────────────
            _rewrite_bug_reports(bug_reports_path, all_reports)

            file_result = {
                "source_file":      source_filename,
                "source_abs":       source_abs,
                "repair_success":   repair_success,
                "patched_path":     patched_path,
                "git_branch":       git_branch,
                "elapsed_sec":      elapsed,
                "bugs_repaired":    len(bugs),
                "verdicts_updated": verdicts_updated,
                "attempts":         repair_result.get("attempts", []),
                "suspicious":       repair_result.get("suspicious", []),
            }
            all_repair_results.append(file_result)

            _emit("repair_complete", file_result, log_callback)
            icon = "Fixed" if repair_success else "Not fixed"
            _log(f"    {icon}: {source_filename}", log_callback)

    finally:
        # Always swap back to PartB adapter
        _swap_adapter(model, "partb")

    succeeded = sum(1 for r in all_repair_results if r["repair_success"])
    _log(
        f"\n  Repair complete: {succeeded}/{total_files} file(s) patched",
        log_callback,
    )
    return all_repair_results
