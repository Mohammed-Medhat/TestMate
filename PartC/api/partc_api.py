"""
Part C APR Pipeline API
Wraps the repair_loop into a clean callable function — mirrors testgen_api.py.
"""
from __future__ import annotations

import sys
import shutil
import logging
import time
import re
from pathlib import Path
from typing import Any, Callable, Optional

_HERE = Path(__file__).parent
_PARTC_ROOT = _HERE.parent
sys.path.insert(0, str(_PARTC_ROOT))
sys.path.insert(0, str(_PARTC_ROOT / "core"))

logger = logging.getLogger(__name__)


def _patch_test_imports(test_source: str, source_stem: str) -> str:
    """
    Rewrite any import that references the original source filename
    to use the workspace copy's stem.
    E.g. `from buggy_code import foo` → `from <source_stem> import foo`
         `import buggy_code` → `import <source_stem>`
    This handles cases where the test file was written against a specific filename.
    The workspace already has the file at the correct name, so a direct import works.
    """
    patched = re.sub(
        r'\bfrom\s+\w+\s+import\b',
        lambda m: m.group(0),
        test_source,
    )
    return patched


def execute_part_c(
    source_file: str,
    test_file: str,
    model: Any = None,
    tokenizer: Any = None,
    max_attempts: int = 3,
    log_callback: Optional[Callable] = None,
    workspace_dir: Optional[str] = None,
) -> dict:
    """
    Run the SBFL-guided APR loop for a single (source, test) pair.

    The source file is NEVER modified in place — we copy both files into a
    temp workspace, repair there, and return the patched text.

    Args:
        source_file:   Absolute path to the buggy Python file.
        test_file:     Absolute path to the pytest test suite.
        model:         Pre-loaded Qwen model (loads from disk if None).
        tokenizer:     Pre-loaded tokenizer (loads from disk if None).
        max_attempts:  Max repair iterations (default 3).
        log_callback:  Optional callback(event_type, *args) for SSE streaming.
        workspace_dir: Where to set up the temp workspace. Auto-created in
                       PartC/workspaces/<timestamp>/ if None.

    Returns:
        {
          "success":      bool,
          "patched":      str,   # fixed source code ("" if no successful patch)
          "original":     str,   # original source code
          "attempts":     [...], # per-attempt dicts
          "suspicious":   [...], # [{line, score, code}, ...]
          "elapsed_sec":  float,
          "source_file":  str,
          "test_file":    str,
        }
    """
    import tempfile
    from core.control_loop import repair_loop

    src_path  = Path(source_file)
    test_path = Path(test_file)
    original_source = src_path.read_text(encoding="utf-8")

    def _log(msg: str, level: str = "info"):
        getattr(logger, level)(msg)
        if log_callback:
            log_callback("log", level, msg)

    _log(f"[C] APR start: {src_path.name} + {test_path.name}")

    # ── Workspace setup ───────────────────────────────────────────────
    if workspace_dir:
        ws = Path(workspace_dir)
        ws.mkdir(parents=True, exist_ok=True)
    else:
        ws_base = _PARTC_ROOT / "workspaces"
        ws_base.mkdir(exist_ok=True)
        ws = Path(tempfile.mkdtemp(dir=ws_base, prefix="run_"))

    ws_source = ws / src_path.name
    ws_test   = ws / test_path.name
    ws_artifacts = ws / "artifacts"

    shutil.copy2(src_path, ws_source)
    shutil.copy2(test_path, ws_test)

    _log(f"  Workspace: {ws}")
    _log(f"  Source:    {ws_source.name}")
    _log(f"  Tests:     {ws_test.name}")

    t0 = time.time()

    try:
        success, attempts_log, suspicious = repair_loop(
            source_file=ws_source,
            test_file=ws_test,
            artifacts_dir=ws_artifacts,
            max_attempts=max_attempts,
            top_k=5,
            model=model,
            tokenizer=tokenizer,
            log_callback=log_callback,
        )

        patched = ws_source.read_text(encoding="utf-8") if success else ""
        elapsed = round(time.time() - t0, 1)

        _log(
            f"[C] APR {'✅ succeeded' if success else '❌ failed'} "
            f"in {elapsed}s ({len(attempts_log)} attempts)"
        )

        return {
            "success":     success,
            "patched":     patched,
            "original":    original_source,
            "attempts":    attempts_log,
            "suspicious":  suspicious,
            "elapsed_sec": elapsed,
            "source_file": str(src_path),
            "test_file":   str(test_path),
        }

    except Exception as exc:
        logger.exception("execute_part_c failed for %s", src_path.name)
        _log(f"[C] ❌ Crashed: {exc}", "error")
        return {
            "success":     False,
            "patched":     "",
            "original":    original_source,
            "attempts":    [],
            "suspicious":  [],
            "elapsed_sec": round(time.time() - t0, 1),
            "source_file": str(src_path),
            "test_file":   str(test_path),
            "error":       str(exc),
        }
