# src/api.py
import json
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional
from sqlalchemy.orm import Session

from ..utils.github_reader import fetch_readme, fetch_repository_data
from ..utils.cleaner import clean_readme
from .feature_extractor import extract_features
from ..utils.model_loader import load_model
from ..models.database import init_db, get_db, ExtractionRecord, UserFeedback
from .api.api_patch import router as scenario_router
from pymongo import MongoClient
import datetime
import os

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
llm_model = None
mongo_client = None
mongo_db = None   # FIX #1: renamed from `db` — was shadowing SQLAlchemy session param


# ---------------------------------------------------------------------------
# Lifespan  FIX #8: replaces deprecated @app.on_event("startup")
# FIX #2: MongoDB connection moved inside lifespan — app no longer crashes at
#          import time if Mongo is unavailable
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global llm_model, mongo_client, mongo_db

    init_db()

    print("Initializing LLM Engine...")
    model_path = os.getenv(
        "MODEL_PATH",
        r"C:\Users\dell\readme_feature_extractor\models\qwen2.5-coder-7b-instruct-q4_k_m.gguf"
    )
    llm_model = load_model(model_path)

    print("Connecting to MongoDB...")
    try:
        mongo_client = MongoClient(
            os.getenv("MONGO_URL", "mongodb://localhost:27017/"),
            serverSelectionTimeoutMS=3000,
        )
        mongo_client.admin.command("ping")
        mongo_db = mongo_client["testmate_db"]
        print("Connected to MongoDB successfully!")
    except Exception as e:
        print(f"[WARNING] MongoDB unavailable: {e} — feedback will be SQLite-only")
        mongo_client = None
        mongo_db = None

    yield

    if mongo_client:
        mongo_client.close()


app = FastAPI(title="TestMate API", version="2.0.0", lifespan=lifespan)
app.include_router(scenario_router)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class RepoRequest(BaseModel):
    repo_name: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.post("/extract_features")
def api_extract_features(req: RepoRequest, db: Session = Depends(get_db)):
    if req.repo_name.startswith("http"):
        repo_data = fetch_repository_data(req.repo_name)
        raw_readme = repo_data.get("readme")
        requirements_text = repo_data.get("requirements", "")
    else:
        raw_readme = fetch_readme(req.repo_name)
        requirements_text = ""

    if not raw_readme:
        raise HTTPException(status_code=404, detail="README not found")

    cleaned = clean_readme(raw_readme)
    features_json = extract_features(cleaned, requirements_text, llm_model)

    try:
        result = json.loads(features_json)
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="LLM output parsing failed")

    record = ExtractionRecord(
        repo_name=req.repo_name,
        project_name=result.get("features", {}).get("project_name"),
        description=result.get("features", {}).get("description"),
    )
    db.add(record)
    db.commit()
    db.refresh(record)

    return {"status": "success", "extraction_id": record.id, **result}


@app.post("/save_feedback")
def api_save_feedback(req: Dict[str, Any], db: Session = Depends(get_db)):
    """
    FIX #1: uses module-level `mongo_db` (not local `db`) for MongoDB.
    FIX #5: maps payload to correct UserFeedback column names.
    """
    print("User Feedback Received!")

    # MongoDB save
    if mongo_db is not None:
        try:
            mongo_db["user_feedbacks"].insert_one({
                "timestamp": datetime.datetime.utcnow(),
                "raw_feedback_data": req,
                "status": "reviewed_by_user",
            })
            print("Saved to MongoDB.")
        except Exception as e:
            print(f"MongoDB save failed: {e}")

    # SQLite save  FIX #5: correct column names
    try:
        sql_feedback = UserFeedback(
            extraction_id=req.get("extraction_id") or 0,
            repo_name=req.get("project_name", ""),
            feature_name="bulk_feedback",
            original_value=None,
            refined_value=None,
            action="edited",
            user_prompt=json.dumps(req, ensure_ascii=False),
        )
        db.add(sql_feedback)
        db.commit()
        print("Saved to SQLite.")
    except Exception as e:
        db.rollback()
        print(f"SQLite save failed: {e}")

    return {"status": "success", "message": "Feedback saved!"}