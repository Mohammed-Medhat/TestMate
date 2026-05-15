# src/main.py
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pymongo import MongoClient

# 1. Import our database init and model loader
from src.models.database import init_db
from src.utils.model_loader import load_model

# 2. Import our newly created API routers
from src.api.extraction import router as extraction_router
from src.api.scenarios import router as scenarios_router

# ---------------------------------------------------------------------------
# Global State (Accessible by endpoints)
# ---------------------------------------------------------------------------
llm_model = None
mongo_client = None
mongo_db = None

# ---------------------------------------------------------------------------
# Lifespan Event: Runs once when the server starts
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global llm_model, mongo_client, mongo_db

    # Initialize SQLite tables
    init_db()

    # Load LLM Engine
    print("Initializing LLM Engine...")
    model_path = os.getenv(
        "MODEL_PATH",
        r"C:\Users\dell\readme_feature_extractor\models\qwen2.5-coder-7b-instruct-q4_k_m.gguf"
    )
    llm_model = load_model(model_path)

    # Connect to MongoDB
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
        print(f"[WARNING] MongoDB unavailable: {e} - feedback will be SQLite-only")
        mongo_client = None
        mongo_db = None

    yield  # Server is running...

    # Cleanup when server shuts down
    if mongo_client:
        mongo_client.close()

# ---------------------------------------------------------------------------
# FastAPI App Initialization
# ---------------------------------------------------------------------------
app = FastAPI(title="TestMate API", version="2.0.0", lifespan=lifespan)

# Add CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Register our API routers so the server knows about them
app.include_router(extraction_router, prefix="/api/v1")
app.include_router(scenarios_router, prefix="/api/v1")

# Health check endpoint
@app.get("/")
def health_check():
    return {"status": "TestMate API is running smoothly! 🚀"}