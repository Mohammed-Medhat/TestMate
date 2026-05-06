# src/api/extraction.py
import json
import datetime
from typing import Dict, Any
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

# 1. Import tools from our newly organized folders
from src.utils.github_reader import fetch_readme, fetch_repository_data
from src.utils.cleaner import clean_readme
from src.services.project_extractor import extract_features
from src.models.database import get_db, ExtractionRecord, UserFeedback

# 2. Create the router (This acts as our localized 'app' for these endpoints)
router = APIRouter(tags=["Feature Extraction"])

# 3. Define Request Schemas
class RepoRequest(BaseModel):
    repo_name: str

# 4. Endpoints
@router.post("/extract_features")
def api_extract_features(req: RepoRequest, db: Session = Depends(get_db)):
    """
    Endpoint to fetch README, clean it, and extract structured features using LLM.
    """
    # Fetch the README content
    if req.repo_name.startswith("http"):
        repo_data = fetch_repository_data(req.repo_name)
        raw_readme = repo_data.get("readme")
        requirements_text = repo_data.get("requirements", "")
    else:
        raw_readme = fetch_readme(req.repo_name)
        requirements_text = ""

    if not raw_readme:
        raise HTTPException(status_code=404, detail="README not found")

    # Clean the README text
    cleaned = clean_readme(raw_readme)
    
    # Import the globally loaded LLM model from main.py dynamically
    from src.main import llm_model 
    
    # Extract features using the LLM engine
    features_json = extract_features(cleaned, requirements_text, llm_model)

    # Parse LLM output
    try:
        result = json.loads(features_json)
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="LLM output parsing failed")

    # Save the successful extraction to SQLite Database
    record = ExtractionRecord(
        repo_name=req.repo_name,
        project_name=result.get("features", {}).get("project_name"),
        description=result.get("features", {}).get("description"),
    )
    db.add(record)
    db.commit()
    db.refresh(record)

    return {"status": "success", "extraction_id": record.id, **result}


@router.post("/save_feedback")
def api_save_feedback(req: Dict[str, Any], db: Session = Depends(get_db)):
    """
    Endpoint to save user edits and feedback to the database.
    """
    print("User Feedback Received!")
    
    # Import the globally connected MongoDB from main.py dynamically
    from src.main import mongo_db

    # Save to MongoDB (if available)
    if mongo_db is not None:
        try:
            mongo_db["user_feedbacks"].insert_one({
                "timestamp": datetime.datetime.utcnow(),
                "raw_feedback_data": req,
                "status": "reviewed_by_user",
            })
            print("Saved to MongoDB successfully.")
        except Exception as e:
            print(f"MongoDB save failed: {e}")

    # Save to SQLite Database
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
        print("Saved to SQLite successfully.")
    except Exception as e:
        db.rollback()
        print(f"SQLite save failed: {e}")

    return {"status": "success", "message": "Feedback saved!"}