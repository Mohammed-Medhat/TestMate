"""
Thin wrapper around RequirementExtractionPipeline.

Handles sys.path setup and working-directory management so the pipeline's
relative paths ("data/raw", "data/processed") resolve correctly regardless
of where Flask is launched from.
"""

import os
import sys
import logging
from pathlib import Path

from config import PIPELINE_DIR, ALIGNED_DS, MODEL_DIR, THRESHOLD

# Make srs2testcases importable
sys.path.insert(0, str(PIPELINE_DIR))
sys.path.insert(0, str(PIPELINE_DIR / "src"))

logger = logging.getLogger("pipeline_runner")


def init_pipeline():
    """
    Import and initialise RequirementExtractionPipeline once at server startup.
    Loading the BERT model + sentence embeddings is expensive (~20-40 s the first
    time); doing it once means every subsequent request is fast.
    """
    logger.info("Initialising pipeline (loading models)…")
    from run_pipeline import RequirementExtractionPipeline  # heavy import

    original_cwd = os.getcwd()
    try:
        os.chdir(PIPELINE_DIR)
        pipeline = RequirementExtractionPipeline(
            aligned_dataset_path=str(ALIGNED_DS) if ALIGNED_DS.exists() else None,
            model_dir=str(MODEL_DIR) if MODEL_DIR.exists() else None,
        )
    finally:
        os.chdir(original_cwd)

    logger.info("Pipeline ready.")
    return pipeline


def run_pipeline(pipeline, doc_path: str, output_path: str) -> dict:
    """
    Run process_document() inside srs2testcases/ as CWD so relative paths work.

    Returns the stats dict produced by process_document().
    Raises on any error so the caller can set job status to 'error'.
    """
    original_cwd = os.getcwd()
    try:
        os.chdir(PIPELINE_DIR)
        stats = pipeline.process_document(
            doc_path=doc_path,
            output_path=output_path,
            fuzzy_threshold=THRESHOLD,
        )
        return stats
    finally:
        os.chdir(original_cwd)
