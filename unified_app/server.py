"""
TestMate Unified Backend — FastAPI on port 8080
Three tab groups: Combined | Part A Only | Part B Only
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import threading
import uuid
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

# ── Paths ──────────────────────────────────────────────────────────────────
_HERE = Path(__file__).parent
_ROOT = _HERE.parent

sys.path.insert(0, str(_ROOT / "PartA" / "srs_pipeline"))
sys.path.insert(0, str(_ROOT / "PartA" / "readme_extractor"))
sys.path.insert(0, str(_ROOT / "PartB" / "testgen"))
sys.path.insert(0, str(_HERE))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("unified_server")

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


def _get_job_queue(job_id: str) -> Optional[asyncio.Queue]:
    with _job_lock:
        return _jobs.get(job_id)


def _make_log_callback(q: asyncio.Queue, loop: asyncio.AbstractEventLoop):
    """Thread-safe callback that puts SSE events into an asyncio Queue."""
    def callback(event_type: str, *args):
        evt: dict = {"type": event_type}
        if event_type in ("log", "ai_status", "code_stream", "code_clear",
                          "pipeline_stage", "result", "file_result",
                          "complete", "error", "progress"):
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
        from readme_api import execute_part_a_readme  # type: ignore[import]
        result = execute_part_a_readme(
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
            from readme_api import execute_part_a_readme  # type: ignore[import]
            readme_result = execute_part_a_readme(
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

        files = auto_discover_files(repo_dir)
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
                )
                loop.call_soon_threadsafe(q.put_nowait, {
                    "type": "file_result",
                    "data": {**result, "file": fi["path"]},
                })
            loop.call_soon_threadsafe(q.put_nowait, {"type": "complete"})
        except Exception as exc:
            logger.exception("partb_run worker failed")
            loop.call_soon_threadsafe(q.put_nowait, {
                "type": "error", "message": str(exc)
            })

    threading.Thread(target=_worker, daemon=True).start()
    return JSONResponse({"job_id": job_id})


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
        log_cb = _make_log_callback(q, loop)
        try:
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
                log_callback=log_cb,
            )
            loop.call_soon_threadsafe(q.put_nowait, {"type": "result", "data": result})
            loop.call_soon_threadsafe(q.put_nowait, {"type": "complete"})
        except Exception as exc:
            logger.exception("combined_run worker failed")
            loop.call_soon_threadsafe(q.put_nowait, {
                "type": "error", "message": str(exc)
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


# ── Health check (polled by Electron main process) ────────────────────────
@app.get("/api/status")
async def status():
    return {"status": "ok"}


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
