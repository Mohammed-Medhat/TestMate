#!/usr/bin/env python
# scripts/generate_negative_samples.py
"""
Generate negative samples (non-requirements) for the PURE dataset.

The PURE dataset CSVs only contain positive examples (Requirements).
To train a classifier, we need negative examples (Non-Requirements).
This script:
1. Loads the aligned requirements (pure_train_aligned.json).
2. Finds the source PDF for each document.
3. Extracts all sentences from the PDF.
4. Filters out sentences that match known requirements.
5. Labels the remaining sentences as 0 (Non-Requirement).
6. Saves the combined dataset.
"""

import json
import os
import re
import spacy
import fitz  # PyMuPDF
from pathlib import Path
from difflib import SequenceMatcher
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("gen_negatives")

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def normalize_text(text):
    """Normalize text for matching."""
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()

def fuzzy_match(text1, text2, threshold=0.85):
    """Check if two texts are similar."""
    return SequenceMatcher(None, normalize_text(text1), normalize_text(text2)).ratio() >= threshold

def find_pdf_file(doc_name, req_dir):
    """Find the PDF file corresponding to doc_name in req_dir."""
    # Handle the '0000 - ' prefix or similar in req/ folder
    # doc_name is likely 'cctns.pdf'
    # we want to find '0000 - cctns.pdf' or just 'cctns.pdf'
    
    # Direct match
    if (req_dir / doc_name).exists():
        return req_dir / doc_name
    
    # Search for suffix match
    for f in req_dir.glob("*.pdf"):
        if f.name.endswith(doc_name) or doc_name in f.name:
            return f
            
    return None

def extract_sentences_from_pdf(pdf_path, nlp):
    """Extract sentences from a PDF file."""
    text = ""
    try:
        doc = fitz.open(pdf_path)
        for page in doc:
            text += page.get_text()
    except Exception as e:
        logger.error(f"Error reading PDF {pdf_path}: {e}")
        return []
    
    # Cleaning
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Sentence splitting
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 10] # Filter very short fragments
    return sentences

def main():
    base_dir = Path(__file__).parent.parent
    aligned_path = base_dir / "data" / "aligned" / "pure_train_aligned.json"
    req_dir = base_dir.parent / "req" # Assuming req is sibling to TestMate or in parent
    # Wait, user context said active workspace is 'd:\Omar\Uni\GP\TestMate'
    # And 'req' is in 'd:\Omar\Uni\GP\TestMate\req'
    # The script is in 'd:\Omar\Uni\GP\TestMate\srs2testcases\scripts'
    # So relative to script: ../../req is WRONG.
    # Relative to project root (srs2testcases): ../req
    
    # Better: find req_dir relative to repo root
    repo_root = base_dir.parent # d:\Omar\Uni\GP\TestMate
    req_dir = repo_root / "req"
    
    output_path = base_dir / "data" / "aligned" / "pure_train_aligned_full.json"
    
    if not aligned_path.exists():
        logger.error(f"Aligned dataset not found: {aligned_path}")
        return

    logger.info("Loading spaCy model...")
    try:
        nlp = spacy.load("en_core_web_sm")
        # Increase max length for large PDFs
        nlp.max_length = 2000000 
    except Exception as e:
        logger.error(f"Failed to load spaCy: {e}")
        return

    logger.info(f"Loading positive samples from {aligned_path}...")
    pos_data = load_json(aligned_path)
    
    # Group requirements by document
    doc_reqs = {}
    for item in pos_data:
        doc_name = item['doc_name']
        if doc_name not in doc_reqs:
            doc_reqs[doc_name] = []
        doc_reqs[doc_name].append(item['text'])
        
    full_dataset = []
    # Add all positive samples first
    full_dataset.extend(pos_data)
    
    logger.info(f"Processing {len(doc_reqs)} documents to find negative samples...")
    
    for doc_name, req_texts in tqdm(doc_reqs.items()):
        # Find PDF
        pdf_path = find_pdf_file(doc_name, req_dir)
        if not pdf_path:
            logger.warning(f"Could not find PDF for {doc_name} in {req_dir}")
            continue
            
        logger.info(f"Processing {pdf_path.name}...")
        
        # Extract all sentences
        all_sentences = extract_sentences_from_pdf(pdf_path, nlp)
        
        # Identify negatives
        neg_count = 0
        for sent in all_sentences:
            # Check if this sentence is a requirement
            is_req = False
            for req_text in req_texts:
                if fuzzy_match(sent, req_text):
                    is_req = True
                    break
            
            if not is_req:
                # Add as negative sample
                full_dataset.append({
                    "text": sent,
                    "label": 0,
                    "doc_name": doc_name,
                    "match_type": "negative_generated"
                })
                neg_count += 1
        
        logger.info(f"  Added {neg_count} negative samples for {doc_name}")

    logger.info(f"Saving full dataset with {len(full_dataset)} samples to {output_path}...")
    save_json(full_dataset, output_path)
    
    # Stats
    pos_count = sum(1 for x in full_dataset if x['label'] == 1)
    neg_count = sum(1 for x in full_dataset if x['label'] == 0)
    logger.info(f"Final Counts: Requirements={pos_count}, Non-Requirements={neg_count}")

if __name__ == "__main__":
    main()
