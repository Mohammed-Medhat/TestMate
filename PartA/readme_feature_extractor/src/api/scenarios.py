# src/api/scenarios.py
from __future__ import annotations
import logging
import json
from typing import Any, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

# 1. Import tools from our newly organized folders
from src.services.pipeline import run_pipeline
from src.services.project_extractor import extract_features
from src.utils.github_reader import fetch_repository_data, fetch_readme
from src.utils.cleaner import clean_readme

logger = logging.getLogger(__name__)

# 2. Create the router for scenario endpoints
router = APIRouter(prefix="/scenarios", tags=["Scenarios Generation"])

# 3. Define Request Schemas
class UserInputPayload(BaseModel):
    description: Optional[str] = None
    problems:    Optional[str] = None
    expected:    Optional[str] = None
    edge_cases:  Optional[str] = None

class ScenarioRequest(BaseModel):
    repo_name:     Optional[str] = None
    extraction_id: Optional[int] = None
    user_input:    Optional[UserInputPayload] = None

# 4. Define the Scenario Generation Endpoint
@router.post("/generate_scenarios")
def api_generate_scenarios(req: ScenarioRequest):
    try:
        print("\n[1] Endpoint started: Generate Scenarios")
        
        # Dynamically import the globally loaded LLM model from main.py 
        # (This prevents circular imports during server startup)
        from src.main import llm_model

        cleaned_readme: Any = None

        # ── Step 1: GitHub Fetch ────────────────────────────────────────────
        repo = (req.repo_name or "").strip()
        if repo and repo != "Uploaded File":
            if repo.startswith("http"):
                repo_data = fetch_repository_data(repo)
                raw_readme = repo_data.get("readme", "")
            else:
                raw_readme = fetch_readme(repo)

            if raw_readme:
                cleaned_readme = clean_readme(raw_readme)
                print("[2] README fetched and cleaned")

        # ── Step 2: Feature Extraction ──────────────────────────────────────
        # Ensure we pass a valid dictionary (or empty-dict fallback), never a bare string
        readme_for_extraction = cleaned_readme if isinstance(cleaned_readme, dict) else {
            "prose": "", "code_blocks": [], "tables": []
        }
        
        print("[3] Extracting features...")
        features_json = extract_features(
            readme_for_extraction, 
            "", 
            llm_model
        )

        # Parse the extracted features safely
        try:
            features_data = json.loads(features_json)
            actual_features = features_data.get("features", {})
            if isinstance(actual_features, str):
                actual_features = json.loads(actual_features)
        except Exception:
            actual_features = {}

        # ── Step 3: Parse User Input ────────────────────────────────────────
        user_input = req.user_input.model_dump(exclude_none=True) if req.user_input else {}

        # ── Step 4: Scenario Generation Pipeline ────────────────────────────
        print("[4] Generating scenarios...")
        # Use the dictionary directly; pipeline.py handles both dict and str correctly
        pipeline_input = cleaned_readme if isinstance(cleaned_readme, dict) else {
            "sections": {"description": user_input.get("description", "")},
            "prose": user_input.get("description", ""),
            "code_blocks": [],
        }
        
        # Run the core scenario pipeline
        scenario_result = run_pipeline(
            cleaned_readme=pipeline_input, 
            user_input=user_input, 
            model=llm_model
        )

        # Return the final aggregated response
        return {
            "status": "success",
            "extraction_id": req.extraction_id or 1,
            "features": actual_features,
            "test_scenarios": scenario_result.to_api_dict().get("test_scenarios", []),
        }

    except Exception as exc:
        logger.exception("generate_scenarios failed")
        raise HTTPException(status_code=500, detail=str(exc))