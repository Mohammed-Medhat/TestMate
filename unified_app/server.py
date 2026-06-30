"""
TestMate Unified Backend — FastAPI on port 8080
Three tab groups: Combined | Part A Only | Part B Only
"""
from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import sys
import threading
import uuid
from pathlib import Path
from typing import Optional

# Force UTF-8 stdout/stderr so emoji in model-loading print() calls
# don't crash worker threads on Windows terminals.
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
if hasattr(sys.stderr, "buffer"):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True)

import uvicorn
from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

# ── Paths ──────────────────────────────────────────────────────────────────
_HERE = Path(__file__).parent
# Dev layout:      unified_app/server.py  → PartB is at _HERE.parent/PartB
# Packaged layout: resources/server.py    → PartB is at _HERE/PartB (sibling)
_ROOT = _HERE if (_HERE / "PartB").exists() else _HERE.parent
_PART_A_README = _ROOT / "PartA" / "readme_extractor"

sys.path.insert(0, str(_ROOT / "PartA" / "srs_pipeline"))
sys.path.insert(0, str(_ROOT / "PartA" / "readme_extractor"))
sys.path.insert(0, str(_ROOT / "PartB" / "testgen"))
sys.path.insert(0, str(_HERE))

# ── File logging (survives process crash — always written to Desktop) ───────
_LOG_PATH = Path.home() / "Desktop" / "testmate_server.log"
_CRASH_LOG_PATH = Path.home() / "Desktop" / "testmate_crash.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(str(_LOG_PATH), encoding="utf-8"),
    ],
)
logger = logging.getLogger("unified_server")

# faulthandler writes the Python C-level stack trace to a file on segfault/abort.
# This is the only way to capture crashes that bypass try/except entirely.
import faulthandler as _fh
_crash_fh = open(str(_CRASH_LOG_PATH), "w", encoding="utf-8")
_fh.enable(file=_crash_fh)
logger.info("Server starting — log: %s  crash: %s", _LOG_PATH, _CRASH_LOG_PATH)
logger.info("_ROOT=%s  _HERE=%s", _ROOT, _HERE)
logger.info("sys.path=%s", sys.path[:6])

# ── App ────────────────────────────────────────────────────────────────────
app = FastAPI(title="TestMate Unified", version="1.0.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)
# ── Job registry ───────────────────────────────────────────────────────────
# Each job_id → asyncio.Queue of SSE event dicts
_jobs: dict[str, asyncio.Queue] = {}
_job_lock = threading.Lock()


def _new_job() -> tuple[str, asyncio.Queue]:
    job_id = str(uuid.uuid4())[:8]
    q: asyncio.Queue = asyncio.Queue()
    with _job_lock:
        _jobs[job_id] = q
    return job_id, q


def _run_readme_extractor(content: str, repo_name: str, user_input: Optional[dict] = None) -> dict:
    """
    Run README extraction in an isolated subprocess (mirrors orchestrator.py's
    combined pipeline). PartA's BnB 4-bit model must NOT be loaded in-process
    here: even after PARTB_RESIDENT.force_release(), del+empty_cache() cannot
    fully release a previously-loaded BnB model on Windows, and a second
    in-process BnB load can segfault the whole server instead of raising a
    catchable error.
    """
    import subprocess
    payload = json.dumps({
        "content": content,
        "repo_name": repo_name,
        "user_input": user_input or {},
    }, ensure_ascii=False)
    proc = subprocess.run(
        [sys.executable, str(_PART_A_README / "extractor_subprocess.py")],
        input=payload,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env={**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"},
        timeout=600,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"README extraction subprocess failed (exit {proc.returncode}): {proc.stderr[-1000:]}"
        )
    return json.loads(proc.stdout)


def _get_job_queue(job_id: str) -> Optional[asyncio.Queue]:
    with _job_lock:
        return _jobs.get(job_id)


def _make_log_callback(q: asyncio.Queue, loop: asyncio.AbstractEventLoop):
    """Thread-safe callback that puts SSE events into an asyncio Queue."""
    def callback(event_type: str, *args):
        evt: dict = {"type": event_type}
        if event_type in ("log", "ai_status", "code_stream", "code_clear",
                          "pipeline_stage", "result", "file_result",
                          "complete", "error", "progress",
                          "repair_start", "repair_file_start", "repair_attempt",
                          "repair_complete", "repair_summary", "verdict_update",
                          "sbfl_result", "attempt_complete",
                          "prepass_result", "stale_fixed"):
            if args:
                if event_type == "log":
                    evt["level"] = args[0] if len(args) > 1 else "info"
                    evt["message"] = args[-1]
                elif event_type == "ai_status":
                    evt["status"] = args[0] if args else ""
                    evt["detail"] = args[1] if len(args) > 1 else ""
                elif event_type == "code_stream":
                    evt["code"] = args[0] if args else ""
                elif event_type == "progress":
                    evt["current"] = args[0] if len(args) > 0 else 0
                    evt["total"]   = args[1] if len(args) > 1 else 0
                    evt["file"]    = args[2] if len(args) > 2 else ""
                elif event_type == "file_result":
                    evt["data"] = args[0] if args else {}
                else:
                    evt["data"] = args[0] if args else {}
        loop.call_soon_threadsafe(q.put_nowait, evt)
    return callback


async def _sse_stream(job_id: str):
    q = _get_job_queue(job_id)
    if q is None:
        yield f"data: {json.dumps({'type':'error','message':'job not found'})}\n\n"
        return
    while True:
        try:
            evt = await asyncio.wait_for(q.get(), timeout=60.0)
            yield f"data: {json.dumps(evt)}\n\n"
            if evt.get("type") in ("complete", "error"):
                break
        except asyncio.TimeoutError:
            yield "data: {\"type\":\"ping\"}\n\n"


# ══════════════════════════════════════════════════════════════════════════
# PART A — SRS Only
# ══════════════════════════════════════════════════════════════════════════
@app.post("/api/parta/srs/run")
async def parta_srs_run(
    file: UploadFile = File(...),
    fuzzy_threshold: float = Form(0.85),
):
    """Upload SRS document → returns aligned requirements JSON."""
    import tempfile
    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=Path(file.filename).suffix,
    ) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        from srs_api import execute_part_a_srs  # type: ignore[import]
        result = execute_part_a_srs(tmp_path, fuzzy_threshold=fuzzy_threshold)
        return JSONResponse({"status": "success", **result})
    except Exception as exc:
        logger.exception("parta_srs_run failed")
        return JSONResponse({"status": "error", "message": str(exc)}, status_code=500)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# ══════════════════════════════════════════════════════════════════════════
# PART A — README Only
# ══════════════════════════════════════════════════════════════════════════
@app.post("/api/parta/readme/run")
async def parta_readme_run(request: Request):
    """
    Body: {
      "content": "<readme text>",
      "repo_name": "optional name",
      "user_input": { "description":…, "problems":…, "expected":…, "edge_cases":… }
    }
    """
    body = await request.json()
    content = body.get("content", "")
    if not content.strip():
        return JSONResponse({"status": "error", "message": "content is empty"}, status_code=400)

    try:
        # Free any keep-alive PartB model first — PartA's model cannot coexist.
        from model_lifecycle import PARTB_RESIDENT  # type: ignore[import]
        if not PARTB_RESIDENT.force_release():
            return JSONResponse(
                {"status": "error",
                 "message": "GPU is still releasing memory from the previous model — please retry in a few seconds."},
                status_code=503,
            )
        result = _run_readme_extractor(
            content=content,
            repo_name=body.get("repo_name", "upload"),
            user_input=body.get("user_input"),
        )
        return JSONResponse({"status": "success", **result})
    except Exception as exc:
        logger.exception("parta_readme_run failed")
        return JSONResponse({"status": "error", "message": str(exc)}, status_code=500)


# ══════════════════════════════════════════════════════════════════════════
# PART A — Both (SRS + README, merged output)
# ══════════════════════════════════════════════════════════════════════════
@app.post("/api/parta/both/run")
async def parta_both_run(
    srs_file: Optional[UploadFile] = File(None),
    readme_content: str = Form(""),
    repo_name: str = Form("upload"),
    fuzzy_threshold: float = Form(0.85),
):
    import tempfile
    srs_path = None
    if srs_file and srs_file.filename:
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=Path(srs_file.filename).suffix
        ) as tmp:
            tmp.write(await srs_file.read())
            srs_path = tmp.name

    try:
        merged: dict = {}

        if srs_path:
            from srs_api import execute_part_a_srs  # type: ignore[import]
            srs_result = execute_part_a_srs(srs_path, fuzzy_threshold=fuzzy_threshold)
            merged["srs"] = srs_result

        if readme_content.strip():
            # README extractor needs the GPU — release the keep-alive PartB model.
            from model_lifecycle import PARTB_RESIDENT  # type: ignore[import]
            if not PARTB_RESIDENT.force_release():
                return JSONResponse(
                    {"status": "error",
                     "message": "GPU is still releasing memory from the previous model — please retry in a few seconds."},
                    status_code=503,
                )
            readme_result = _run_readme_extractor(
                content=readme_content, repo_name=repo_name
            )
            merged["readme"] = readme_result

        if not merged:
            return JSONResponse(
                {"status": "error", "message": "Provide at least one input (SRS or README)"},
                status_code=400,
            )

        return JSONResponse({"status": "success", **merged})
    except Exception as exc:
        logger.exception("parta_both_run failed")
        return JSONResponse({"status": "error", "message": str(exc)}, status_code=500)
    finally:
        if srs_path:
            try:
                os.unlink(srs_path)
            except OSError:
                pass


# ══════════════════════════════════════════════════════════════════════════
# PART B — Proxy core endpoints (discover + run + stream)
# ══════════════════════════════════════════════════════════════════════════

@app.post("/api/partb/discover")
async def partb_discover(request: Request):
    body = await request.json()
    url = body.get("url", "")
    branch = body.get("branch", "")
    try:
        sys.path.insert(0, str(_ROOT / "PartB" / "testgen"))
        from api_server import auto_discover_files, clone_repo  # type: ignore[import]
        import re, os as _os

        is_local = (
            _os.path.isdir(url)
            or (len(url) >= 2 and url[1] == ":")
            or url.startswith("/")
            or url.startswith("\\")
        )
        if is_local:
            repo_dir = _os.path.abspath(url)
            repo_name = _os.path.basename(repo_dir)
        else:
            if not re.match(r"https?://", url):
                url = "https://github.com/" + url
            repo_dir, repo_name = clone_repo(url, branch or None)

        include_tests = bool(body.get("include_tests", False))
        files = auto_discover_files(repo_dir, include_tests=include_tests)
        return JSONResponse({"files": files, "repo_name": repo_name, "repo_dir": repo_dir})
    except Exception as exc:
        logger.exception("partb_discover failed")
        return JSONResponse({"error": str(exc)}, status_code=500)


@app.post("/api/partb/run")
async def partb_run(request: Request):
    body = await request.json()
    files = body.get("files", [])
    if not files:
        return JSONResponse({"error": "no files selected"}, status_code=400)

    loop = asyncio.get_event_loop()
    job_id, q = _new_job()

    def _worker():
        log_cb = _make_log_callback(q, loop)
        try:
            from testgen_api import execute_part_b  # type: ignore[import]
            from model_lifecycle import PARTB_RESIDENT  # type: ignore[import]

            # Load the model once and keep it resident, so the post-run chat
            # bubble can refine these tests without a cold reload. Falls back to
            # per-file loading inside execute_part_b if the warm load fails.
            try:
                _model, _tokenizer = PARTB_RESIDENT.acquire(
                    use_lora=not body.get("use_base_only", False))
            except Exception as _e:
                logger.warning("resident acquire failed (%s) — per-file load", _e)
                _model = _tokenizer = None

            for fi in files:
                log_cb("log", "info", f"Generating tests for {fi['path']}...")
                result = execute_part_b(
                    target_file=fi["abs_path"],
                    import_path=fi.get("import_path"),
                    deep_scan=body.get("deep_scan", False),
                    max_retries=body.get("max_retries", 3),
                    log_callback=log_cb,
                    plan_mode=body.get("plan_mode", False),
                    use_base_only=body.get("use_base_only", False),
                    quality_mode=body.get("quality_mode", "fast"),
                    auto_repair=body.get("auto_repair", False),
                    use_docker=body.get("use_docker", False),
                    model=_model,
                    tokenizer=_tokenizer,
                )
                loop.call_soon_threadsafe(q.put_nowait, {
                    "type": "file_result",
                    "data": {**result, "file": fi["path"]},
                })
            PARTB_RESIDENT.touch()  # start the keep-alive idle window
            loop.call_soon_threadsafe(q.put_nowait, {"type": "complete"})
        except Exception as exc:
            logger.exception("partb_run worker failed")
            loop.call_soon_threadsafe(q.put_nowait, {
                "type": "error", "message": str(exc)
            })

    threading.Thread(target=_worker, daemon=True).start()
    return JSONResponse({"job_id": job_id})


# ── PartB chat refinement (edit a generated test by instruction) ───────────
@app.post("/api/partb/chat")
async def partb_chat(request: Request):
    """Apply one natural-language edit to an already-generated test file.

    Body: { target_file, test_path, test_code, message, history[], use_docker?,
            use_base_only?, max_retries? }
    Returns { job_id } — stream via /api/partb/chat/stream?job_id=<id>.
    """
    body = await request.json()
    target_file = body.get("target_file", "")
    message     = (body.get("message") or "").strip()
    if not message:
        return JSONResponse({"error": "message is empty"}, status_code=400)
    if not target_file:
        return JSONResponse({"error": "target_file is required"}, status_code=400)

    loop = asyncio.get_event_loop()
    job_id, q = _new_job()

    def _worker():
        log_cb = _make_log_callback(q, loop)
        try:
            from model_lifecycle import PARTB_RESIDENT  # type: ignore[import]
            from testgen_api import refine_test_file     # type: ignore[import]

            def _warm(detail: str):
                loop.call_soon_threadsafe(q.put_nowait, {
                    "type": "ai_status", "status": "Loading model", "detail": detail})

            model, tokenizer = PARTB_RESIDENT.acquire(
                use_lora=not body.get("use_base_only", False), status_cb=_warm)
            PARTB_RESIDENT.touch()

            result = refine_test_file(
                target_file=target_file,
                test_path=body.get("test_path", ""),
                test_code=body.get("test_code", ""),
                instruction=message,
                model=model,
                tokenizer=tokenizer,
                history=body.get("history", []),
                max_retries=body.get("max_retries", 3),
                log_callback=log_cb,
                use_docker=body.get("use_docker", False),
            )
            PARTB_RESIDENT.touch()

            loop.call_soon_threadsafe(q.put_nowait, {"type": "chat_reply", "data": result})
            if result.get("edited"):
                loop.call_soon_threadsafe(q.put_nowait, {
                    "type": "file_result",
                    "data": {**result, "file": target_file},
                })
            loop.call_soon_threadsafe(q.put_nowait, {"type": "complete"})
        except Exception as exc:
            logger.exception("partb_chat worker failed")
            loop.call_soon_threadsafe(q.put_nowait, {"type": "error", "message": str(exc)})

    threading.Thread(target=_worker, daemon=True).start()
    return JSONResponse({"job_id": job_id})


@app.get("/api/partb/chat/stream")
async def partb_chat_stream(job_id: str):
    return StreamingResponse(
        _sse_stream(job_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/partb/save")
async def partb_save(request: Request):
    """Persist a manual edit straight to disk (no model needed)."""
    body = await request.json()
    test_path = body.get("test_path", "")
    test_code = body.get("test_code", "")
    if not test_path:
        return JSONResponse({"error": "test_path is required"}, status_code=400)
    try:
        Path(test_path).write_text(test_code, encoding="utf-8")
        return JSONResponse({"status": "ok", "test_path": test_path})
    except Exception as exc:
        logger.exception("partb_save failed")
        return JSONResponse({"error": str(exc)}, status_code=500)


@app.get("/api/partb/stream")
async def partb_stream(job_id: str):
    return StreamingResponse(
        _sse_stream(job_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ══════════════════════════════════════════════════════════════════════════
# COMBINED — Full pipeline (Part A → Part B)
# ══════════════════════════════════════════════════════════════════════════
@app.post("/api/combined/run")
async def combined_run(
    srs_file: Optional[UploadFile] = File(None),
    readme_content: str = Form(""),
    files_json: str = Form("[]"),               # NEW: JSON-encoded list of FileInfo
    target_code_file: str = Form(""),           # back-compat: single-file path (auto-wrapped)
    user_input_json: str = Form("{}"),
    import_path: str = Form(""),                # back-compat: only used when target_code_file is set
    deep_scan: bool = Form(False),
    max_retries: int = Form(3),
    plan_mode: bool = Form(False),
    top_k_requirements: int = Form(10),
    use_base_only: bool = Form(False),
    quality_mode: str = Form("fast"),   # fast | balanced | best
    auto_repair: bool = Form(False),    # opt-in: invoke PartC on confirmed bugs
    use_docker: bool = Form(False),     # opt-in: pytest in per-repo Docker image
):
    import tempfile, json as _json

    srs_path = None
    if srs_file and srs_file.filename:
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=Path(srs_file.filename).suffix
        ) as tmp:
            tmp.write(await srs_file.read())
            srs_path = tmp.name

    try:
        user_input = _json.loads(user_input_json) if user_input_json.strip() else {}
    except Exception:
        user_input = {}

    # Build the target_files list — supports both multi-file (files_json)
    # and single-file (target_code_file) inputs for backward compatibility.
    try:
        target_files = _json.loads(files_json) if files_json.strip() else []
        if not isinstance(target_files, list):
            target_files = []
    except Exception:
        target_files = []

    if not target_files and target_code_file:
        target_files = [{
            "path":        Path(target_code_file).name,
            "abs_path":    target_code_file,
            "import_path": import_path or None,
        }]

    if not target_files:
        return JSONResponse({"error": "No target files provided"}, status_code=400)

    loop = asyncio.get_event_loop()
    job_id, q = _new_job()

    _srs_path_capture = srs_path

    def _worker():
        logger.info("[worker] combined_run thread started (job=%s)", job_id)
        log_cb = _make_log_callback(q, loop)
        try:
            logger.info("[worker] importing PARTB_RESIDENT")
            from model_lifecycle import PARTB_RESIDENT  # type: ignore[import]
            logger.info("[worker] calling force_release")
            released = PARTB_RESIDENT.force_release()
            if not released:
                # Warn but don't abort — let the model load attempt proceed.
                # If VRAM is truly exhausted the load will raise OOM with a clear
                # message; blocking here just hides the real error from the user.
                logger.warning("force_release: GPU memory low — proceeding anyway (may OOM)")
                loop.call_soon_threadsafe(q.put_nowait, {
                    "type": "log", "level": "warning",
                    "message": "⚠️ GPU memory still clearing — loading model anyway (may be slow)",
                })
            from orchestrator import execute_combined  # type: ignore[import]
            result = execute_combined(
                srs_path=_srs_path_capture,
                target_files=target_files,
                readme_content=readme_content or None,
                user_input=user_input or None,
                deep_scan=deep_scan,
                max_retries=max_retries,
                plan_mode=plan_mode,
                top_k_requirements=top_k_requirements,
                use_base_only=use_base_only,
                quality_mode=quality_mode,
                auto_repair=auto_repair,
                use_docker=use_docker,
                log_callback=log_cb,
            )
            loop.call_soon_threadsafe(q.put_nowait, {"type": "result", "data": result})
            loop.call_soon_threadsafe(q.put_nowait, {"type": "complete"})
        except Exception as exc:
            import traceback as _tb
            full = _tb.format_exc()
            logger.exception("combined_run worker failed")
            loop.call_soon_threadsafe(q.put_nowait, {
                "type": "error", "message": f"{exc}\n\n{full}"
            })
        finally:
            if _srs_path_capture:
                try:
                    os.unlink(_srs_path_capture)
                except OSError:
                    pass

    threading.Thread(target=_worker, daemon=True).start()
    return JSONResponse({"job_id": job_id})


@app.get("/api/combined/stream")
async def combined_stream(job_id: str):
    return StreamingResponse(
        _sse_stream(job_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ══════════════════════════════════════════════════════════════════════════
# PART C — APR (Automated Program Repair)
# ══════════════════════════════════════════════════════════════════════════

@app.post("/api/partc/run")
async def partc_run(request: Request):
    """
    Body: {
      "source_file":  "/abs/path/to/source.py",
      "test_file":    "/abs/path/to/test_source.py",
      "max_attempts": 3
    }
    Returns: { "job_id": "<id>" }  — stream via /api/partc/stream?job_id=<id>
    """
    body = await request.json()
    source_file  = body.get("source_file", "")
    test_file    = body.get("test_file", "")
    max_attempts = int(body.get("max_attempts", 3))

    if not source_file or not test_file:
        return JSONResponse({"error": "source_file and test_file are required"}, status_code=400)

    import os as _os
    if not _os.path.isfile(source_file):
        return JSONResponse({"error": f"source_file not found: {source_file}"}, status_code=400)
    if not _os.path.isfile(test_file):
        return JSONResponse({"error": f"test_file not found: {test_file}"}, status_code=400)

    loop = asyncio.get_event_loop()
    job_id, q = _new_job()

    def _worker():
        log_cb = _make_log_callback(q, loop)
        try:
            sys.path.insert(0, str(_ROOT / "PartC" / "api"))
            sys.path.insert(0, str(_ROOT / "PartC"))
            from partc_api import execute_part_c  # type: ignore[import]
            from model_lifecycle import part_c_model, PARTB_RESIDENT  # type: ignore[import]

            # Standalone PartC loads its own session — drop the keep-alive model.
            if not PARTB_RESIDENT.force_release():
                loop.call_soon_threadsafe(q.put_nowait, {
                    "type": "error",
                    "message": "GPU is still releasing memory from the previous model — please retry in a few seconds.",
                })
                return
            with part_c_model() as (model, tokenizer):
                result = execute_part_c(
                    source_file=source_file,
                    test_file=test_file,
                    model=model,
                    tokenizer=tokenizer,
                    max_attempts=max_attempts,
                    log_callback=log_cb,
                )

            loop.call_soon_threadsafe(q.put_nowait, {"type": "result", "data": result})
            loop.call_soon_threadsafe(q.put_nowait, {"type": "complete"})
        except Exception as exc:
            logger.exception("partc_run worker failed")
            loop.call_soon_threadsafe(q.put_nowait, {
                "type": "error", "message": str(exc)
            })

    threading.Thread(target=_worker, daemon=True).start()
    return JSONResponse({"job_id": job_id})


@app.get("/api/partc/stream")
async def partc_stream(job_id: str):
    return StreamingResponse(
        _sse_stream(job_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ══════════════════════════════════════════════════════════════════════════
# PART C — Coverage & Mutation analysis endpoints
# ══════════════════════════════════════════════════════════════════════════

@app.post("/api/partc/coverage")
async def partc_coverage(request: Request):
    """
    Body: { "source_file": "/abs/path/source.py", "test_file": "/abs/path/test_source.py" }
    Returns line coverage data using coverage.py (pytest --cov).
    """
    import subprocess as _sp
    import json as _json
    import tempfile as _tmp
    body = await request.json()
    source_file = body.get("source_file", "")
    test_file   = body.get("test_file", "")
    if not source_file or not test_file:
        return JSONResponse({"error": "source_file and test_file are required"}, status_code=400)

    src_path  = Path(source_file)
    test_path = Path(test_file)
    if not src_path.is_file():
        return JSONResponse({"error": f"source_file not found: {source_file}"}, status_code=400)
    if not test_path.is_file():
        return JSONResponse({"error": f"test_file not found: {test_file}"}, status_code=400)

    work_dir  = src_path.parent
    cov_json  = work_dir / "coverage_testmate.json"
    stem      = src_path.stem
    try:
        r = _sp.run(
            [
                sys.executable, "-m", "pytest", str(test_path),
                f"--cov={stem}",
                "--cov-report=json:coverage_testmate.json",
                "--cov-report=term-missing",
                "-q", "--no-header",
            ],
            capture_output=True, text=True, cwd=str(work_dir), timeout=120
        )
        if not cov_json.exists():
            return JSONResponse({
                "error": "coverage.json not generated",
                "raw": (r.stdout + r.stderr)[-2000:]
            }, status_code=500)

        data = _json.loads(cov_json.read_text(encoding="utf-8"))
        file_data = {}
        for fname, finfo in data.get("files", {}).items():
            if Path(fname).stem == stem or fname == source_file:
                file_data = finfo
                break
        if not file_data:
            # fallback: first entry
            items = list(data.get("files", {}).values())
            file_data = items[0] if items else {}

        summary   = file_data.get("summary", {})
        functions = file_data.get("functions", {})

        fn_results = []
        for fn_name, fn_info in functions.items():
            s = fn_info.get("summary", {})
            fn_results.append({
                "name":    fn_name,
                "covered": s.get("covered_lines", 0),
                "total":   s.get("num_statements", 0),
                "pct":     round(s.get("percent_covered", 0), 1),
                "missing": fn_info.get("missing_lines", []),
            })

        overall_pct = round(summary.get("percent_covered", 0), 1)

        # Also build coverage_map format for the CoverageView component
        covered_lines   = sorted(file_data.get("executed_lines", []))
        missing_lines   = sorted(file_data.get("missing_lines", []))
        uncovered_lines = missing_lines

        source_text = ""
        try:
            source_text = src_path.read_text(encoding="utf-8")
        except Exception:
            pass

        cov_json.unlink(missing_ok=True)
        return JSONResponse({
            "overall_pct":   overall_pct,
            "covered_lines": summary.get("covered_lines", 0),
            "total_lines":   summary.get("num_statements", 0),
            "missing_lines": missing_lines,
            "functions":     fn_results,
            # coverage_map format for CoverageView component
            "coverage_map": {
                src_path.name: {
                    "source":             source_text,
                    "covered_lines":      covered_lines,
                    "uncovered_lines":    uncovered_lines,
                    "line_coverage_pct":  overall_pct,
                    "num_statements":     summary.get("num_statements", 0),
                }
            },
            # scalar for RightSidebar gauge
            "line_coverage": overall_pct,
            "raw": (r.stdout)[-1000:],
        })
    except _sp.TimeoutExpired:
        return JSONResponse({"error": "Coverage run timed out"}, status_code=500)
    except Exception as exc:
        import traceback
        return JSONResponse({"error": str(exc), "trace": traceback.format_exc()}, status_code=500)


# ── AST Mutation engine (Windows-compatible, no mutmut needed) ─────────────
def _ast_mutation_score(source_file: str, test_file: str) -> dict:
    """
    AST-based mutation testing. Mutates operators/comparators/arithmetic in
    source_file, runs test_file against each mutant, and returns kill stats.
    """
    import ast as _ast
    import copy as _copy
    import subprocess as _sp

    src_path  = Path(source_file)
    test_path = Path(test_file)
    original  = src_path.read_text(encoding="utf-8")

    BINOP_MAP = {
        _ast.Add:      [_ast.Sub, _ast.Mult],
        _ast.Sub:      [_ast.Add, _ast.Mult],
        _ast.Mult:     [_ast.Add, _ast.Sub],
        _ast.Div:      [_ast.FloorDiv, _ast.Mult],
        _ast.FloorDiv: [_ast.Div, _ast.Add],
        _ast.Mod:      [_ast.Add, _ast.Mult],
    }
    CMP_MAP = {
        _ast.Lt:    [_ast.LtE, _ast.Gt,  _ast.GtE, _ast.Eq],
        _ast.LtE:   [_ast.Lt,  _ast.Gt,  _ast.GtE, _ast.Eq],
        _ast.Gt:    [_ast.GtE, _ast.Lt,  _ast.LtE, _ast.Eq],
        _ast.GtE:   [_ast.Gt,  _ast.Lt,  _ast.LtE, _ast.Eq],
        _ast.Eq:    [_ast.NotEq, _ast.Lt, _ast.Gt],
        _ast.NotEq: [_ast.Eq,   _ast.Lt,  _ast.Gt],
    }

    src_lines = original.splitlines()
    orig_tree = _ast.parse(original)
    mutants   = []

    def _make_mutant(orig_node, mutated_tree, label):
        try:
            src = _ast.unparse(mutated_tree)
            ln  = getattr(orig_node, "lineno", "?")
            orig_line = src_lines[ln - 1].strip() if isinstance(ln, int) and 0 < ln <= len(src_lines) else ""
            mutants.append({"label": label, "line": ln, "orig": orig_line, "source": src})
        except Exception:
            pass

    for node in _ast.walk(orig_tree):
        # BinOp mutations
        if isinstance(node, _ast.BinOp):
            for orig_t, reps in BINOP_MAP.items():
                if isinstance(node.op, orig_t):
                    for rep in reps:
                        tc = _copy.deepcopy(orig_tree)
                        for n in _ast.walk(tc):
                            if (isinstance(n, _ast.BinOp) and isinstance(n.op, orig_t)
                                    and getattr(n, "lineno", None) == getattr(node, "lineno", None)
                                    and getattr(n, "col_offset", None) == getattr(node, "col_offset", None)):
                                n.op = rep()
                                break
                        _make_mutant(node, tc, f"BinOp {orig_t.__name__}→{rep.__name__} (line {node.lineno})")
        # Compare mutations
        if isinstance(node, _ast.Compare):
            for op in node.ops:
                for orig_t, reps in CMP_MAP.items():
                    if isinstance(op, orig_t):
                        for rep in reps:
                            tc = _copy.deepcopy(orig_tree)
                            for n in _ast.walk(tc):
                                if (isinstance(n, _ast.Compare)
                                        and getattr(n, "lineno", None) == getattr(node, "lineno", None)
                                        and getattr(n, "col_offset", None) == getattr(node, "col_offset", None)):
                                    n.ops = [rep() if type(o) is orig_t else o for o in n.ops]
                                    break
                            _make_mutant(node, tc, f"Cmp {orig_t.__name__}→{rep.__name__} (line {node.lineno})")

    results_list = []
    killed = 0
    for i, m in enumerate(mutants):
        try:
            src_path.write_text(m["source"], encoding="utf-8")
            r = _sp.run(
                [sys.executable, "-m", "pytest", str(test_path), "-x", "-q", "--no-header", "--tb=no"],
                capture_output=True, text=True, cwd=str(src_path.parent), timeout=30
            )
            is_killed = (r.returncode != 0)
        except _sp.TimeoutExpired:
            is_killed = True
        except Exception:
            is_killed = False
        finally:
            src_path.write_text(original, encoding="utf-8")

        if is_killed:
            killed += 1
        results_list.append({
            "id":     i + 1,
            "label":  m["label"],
            "line":   m["line"],
            "orig":   m["orig"],
            "status": "killed" if is_killed else "survived",
        })

    total    = len(results_list)
    survived = total - killed
    score    = round((killed / total) * 100, 1) if total > 0 else 0.0

    return {
        "score":    score,
        "killed":   killed,
        "survived": survived,
        "total":    total,
        "mutants":  results_list,
        # scalar for RightSidebar gauge
        "mutation_score": score,
    }


@app.post("/api/partc/mutation")
async def partc_mutation(request: Request):
    """
    Body: { "source_file": "/abs/path/source.py", "test_file": "/abs/path/test_source.py" }
    Runs AST-based mutation testing. May take 30-120 seconds.
    """
    body = await request.json()
    source_file = body.get("source_file", "")
    test_file   = body.get("test_file", "")
    if not source_file or not test_file:
        return JSONResponse({"error": "source_file and test_file are required"}, status_code=400)
    if not Path(source_file).is_file():
        return JSONResponse({"error": f"source_file not found: {source_file}"}, status_code=400)
    if not Path(test_file).is_file():
        return JSONResponse({"error": f"test_file not found: {test_file}"}, status_code=400)
    try:
        result = await asyncio.get_event_loop().run_in_executor(
            None, _ast_mutation_score, source_file, test_file
        )
        return JSONResponse(result)
    except Exception as exc:
        import traceback
        return JSONResponse({"error": str(exc), "trace": traceback.format_exc()}, status_code=500)


# ── Health check (polled by Electron main process) ────────────────────────
@app.get("/api/status")
async def status():
    return {"status": "ok"}


# ── Import smoke-test (call before first run to surface missing packages) ──
@app.get("/api/debug/imports")
async def debug_imports():
    """Test every package the pipeline needs. Returns pass/fail per package."""
    import importlib
    results = {}
    packages = [
        ("torch",              "import torch; results['torch_cuda'] = torch.cuda.is_available()"),
        ("transformers",       None),
        ("peft",               None),
        ("bitsandbytes",       None),
        ("accelerate",         None),
        ("sentence_transformers", None),
        ("spacy",              "import spacy; results['spacy_model'] = bool(spacy.util.is_package('en_core_web_sm'))"),
        ("networkx",           None),
        ("fastapi",            None),
        ("pytest",             None),
        ("pytest_cov",         "import pytest_cov"),
    ]
    for pkg, extra in packages:
        try:
            importlib.import_module(pkg)
            results[pkg] = "ok"
            if extra:
                try:
                    exec(extra, {"results": results})  # noqa: S102
                except Exception as e:
                    results[f"{pkg}_extra"] = str(e)
        except Exception as exc:
            results[pkg] = f"MISSING — {exc}"

    # Check model paths
    import main as _m  # type: ignore
    results["base_model_path"] = str(_m.BASE_MODEL)
    results["base_model_exists"] = Path(_m.BASE_MODEL).exists()
    from model_lifecycle import _PARTB_LORA_PATH  # type: ignore
    results["lora_path"] = str(_PARTB_LORA_PATH)
    results["lora_exists"] = _PARTB_LORA_PATH.exists()
    results["python_exe"] = sys.executable
    results["log_file"] = str(_LOG_PATH)

    logger.info("/api/debug/imports result: %s", results)
    return results


# ══════════════════════════════════════════════════════════════════════════
# HISTORY — persist run records so Landing can show recent runs
# ══════════════════════════════════════════════════════════════════════════
_HISTORY_FILE = _HERE / "results" / "history.json"
_HISTORY_LOCK = threading.Lock()


def _load_history() -> list:
    try:
        if _HISTORY_FILE.exists():
            return json.loads(_HISTORY_FILE.read_text(encoding="utf-8"))
    except Exception:
        pass
    return []


def _save_history(records: list) -> None:
    try:
        _HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        _HISTORY_FILE.write_text(json.dumps(records, indent=2), encoding="utf-8")
    except Exception as exc:
        logger.warning("Could not save history: %s", exc)


def record_run(mode: str, summary: str, run_status: str) -> None:
    """Append a run to history.json (called after each run completes)."""
    import datetime
    with _HISTORY_LOCK:
        records = _load_history()
        records.insert(0, {
            "id": str(uuid.uuid4())[:8],
            "mode": mode,
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "summary": summary,
            "status": run_status,
        })
        _save_history(records[:100])   # cap at 100


@app.get("/api/history")
async def get_history():
    with _HISTORY_LOCK:
        return _load_history()[:10]


@app.post("/api/history/record")
async def post_history(request: Request):
    body = await request.json()
    record_run(
        mode=body.get("mode", "unknown"),
        summary=body.get("summary", ""),
        run_status=body.get("status", "success"),
    )
    return {"ok": True}


# ══════════════════════════════════════════════════════════════════════════
# HITL PROXIES — forward review/plan decisions to PartB backend (port 8000)
# ══════════════════════════════════════════════════════════════════════════
import urllib.request as _urllib_req
import urllib.error as _urllib_err

_PARTB_BASE = "http://127.0.0.1:8000"


def _proxy_post(path: str, body: dict) -> dict:
    data = json.dumps(body).encode()
    req = _urllib_req.Request(
        f"{_PARTB_BASE}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with _urllib_req.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read())
    except _urllib_err.URLError as exc:
        return {"error": str(exc)}


@app.post("/api/review")
async def proxy_review(request: Request):
    body = await request.json()
    return _proxy_post("/api/review", body)


@app.post("/api/plan_review")
async def proxy_plan_review(request: Request):
    body = await request.json()
    return _proxy_post("/api/plan_review", body)


# ── Entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="TestMate Unified Server")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()
    print(f"TestMate Unified Server -> http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
