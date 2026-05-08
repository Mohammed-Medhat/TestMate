from pathlib import Path

BASE_DIR     = Path(__file__).parent
PIPELINE_DIR = BASE_DIR.parent / "srs2testcases"
ALIGNED_DS   = PIPELINE_DIR / "data" / "aligned" / "pure_train_aligned.json"
MODEL_DIR    = PIPELINE_DIR / "models" / "bert-classifier"
UPLOAD_DIR   = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

ALLOWED_EXT  = {".pdf", ".docx", ".doc"}
THRESHOLD    = 0.85
