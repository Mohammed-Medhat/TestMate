"""
TestMate Stage 1 — Flask backend.

Endpoints:
  GET  /                      → serves index.html
  POST /api/upload            → accept PDF/DOCX, start pipeline in background thread
  GET  /api/status/<job_id>   → poll job status
  GET  /api/results/<job_id>  → return label==1 requirements + stats
  POST /api/export            → transform to Stage 2 JSON and send as download
"""

import json
import logging
import threading
import uuid
from pathlib import Path

from flask import Flask, jsonify, render_template, request, Response, abort
from flask_cors import CORS

import config
from pipeline_runner import init_pipeline, run_pipeline

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("app")

app = Flask(__name__, template_folder="templates")
CORS(app)

# ---------------------------------------------------------------------------
# Pipeline — initialised once at startup
# ---------------------------------------------------------------------------

logger.info("Starting pipeline initialisation…")
_pipeline = init_pipeline()
logger.info("Pipeline initialised. Server ready.")

# ---------------------------------------------------------------------------
# In-memory job store  (single-user offline tool — no persistence needed)
# ---------------------------------------------------------------------------

_jobs: dict = {}
_jobs_lock = threading.Lock()


def _set_job(job_id: str, **kwargs):
    with _jobs_lock:
        _jobs.setdefault(job_id, {}).update(kwargs)


def _get_job(job_id: str) -> dict:
    with _jobs_lock:
        return dict(_jobs.get(job_id, {}))


# ---------------------------------------------------------------------------
# Background worker
# ---------------------------------------------------------------------------

def _run_job(job_id: str, doc_path: str, output_path: str, original_name: str = ""):
    try:
        _set_job(job_id, progress_msg="Extracting text from document…")
        stats = run_pipeline(_pipeline, doc_path, output_path)
        # Replace UUID-based filename with the original uploaded name
        if original_name:
            stats["document"] = original_name
        _set_job(job_id, status="done", stats=stats, progress_msg="Done.")
        logger.info(f"Job {job_id} complete: {stats}")
    except Exception as exc:
        logger.exception(f"Job {job_id} failed")
        _set_job(job_id, status="error", error=str(exc))


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
def index():
    return render_template("index.html")


@app.post("/api/upload")
def upload():
    if "file" not in request.files:
        return jsonify(error="No file provided."), 400

    f = request.files["file"]
    suffix = Path(f.filename).suffix.lower()

    if suffix not in config.ALLOWED_EXT:
        return jsonify(error=f"Unsupported file type '{suffix}'. Use PDF or DOCX."), 400

    job_id = uuid.uuid4().hex
    safe_name = f"{job_id}{suffix}"
    doc_path = config.UPLOAD_DIR / safe_name
    output_path = config.UPLOAD_DIR / f"{job_id}_results.json"

    original_name = Path(f.filename).name
    f.save(str(doc_path))
    _set_job(job_id, status="running", original_name=original_name,
             progress_msg="Uploading complete. Starting pipeline…")

    thread = threading.Thread(
        target=_run_job,
        args=(job_id, str(doc_path), str(output_path), original_name),
        daemon=True,
    )
    thread.start()

    return jsonify(job_id=job_id), 202


@app.get("/api/status/<job_id>")
def status(job_id: str):
    job = _get_job(job_id)
    if not job:
        return jsonify(error="Unknown job."), 404
    return jsonify(
        status=job.get("status", "running"),
        progress_msg=job.get("progress_msg", ""),
        error=job.get("error"),
    )


@app.get("/api/results/<job_id>")
def results(job_id: str):
    job = _get_job(job_id)
    if not job:
        return jsonify(error="Unknown job."), 404
    if job.get("status") != "done":
        return jsonify(error="Job not finished yet."), 409

    output_path = config.UPLOAD_DIR / f"{job_id}_results.json"
    if not output_path.exists():
        return jsonify(error="Result file not found."), 500

    with open(output_path, encoding="utf-8") as fh:
        all_sentences = json.load(fh)

    requirements = [s for s in all_sentences if s.get("label") == 1]
    stats = job.get("stats", {})

    return jsonify(requirements=requirements, stats=stats)


@app.post("/api/export")
def export():
    body = request.get_json(silent=True)
    if not body or "requirements" not in body:
        return jsonify(error="Missing 'requirements' field."), 400

    stage2 = [
        {
            "id": r.get("sentence_id", ""),
            "text": r.get("text", ""),
            "page": r.get("page"),
        }
        for r in body["requirements"]
    ]

    payload = json.dumps(stage2, ensure_ascii=False, indent=2)
    return Response(
        payload,
        mimetype="application/json",
        headers={"Content-Disposition": "attachment; filename=stage2_requirements.json"},
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
