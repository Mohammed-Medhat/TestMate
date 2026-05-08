# src/preprocessing/align_pure_datasets.py
"""
Align PURE dataset sentences with PURE_Annotated labels.

This module provides functionality to match sentences extracted from SRS documents
with their corresponding labels from the PURE_Annotated dataset using fuzzy text matching.

Output format:
{
  "text": "Full sentence text from PURE",
  "label": 0 or 1,
  "doc_id": "pure_doc_001",
  "block_id": "b78",
  "sentence_id": "b78_0",
  "annotated_text": "Matched text from PURE_Annotated",
  "match_type": "exact" | "fuzzy" | "unmatched",
  "score": 0.0 to 1.0
}
"""

import json
import csv
import os
import re
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from difflib import SequenceMatcher
import argparse

try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    _SEMANTIC_AVAILABLE = True
except ImportError:
    _SEMANTIC_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("align_pure_datasets")


def normalize_text(text: str) -> str:
    """
    Normalize text for better matching by removing extra whitespace and standardizing.
    
    Args:
        text: Input text string
        
    Returns:
        Normalized text string
    """
    if not text:
        return ""
    
    # Remove extra whitespace
    text = " ".join(text.split())
    
    # Remove special characters that might differ between datasets
    text = text.replace('\xa0', ' ')
    text = text.replace("'", "'").replace(""", '"').replace(""", '"')
    text = text.strip()
    
    return text


def fuzzy_match_score(text1: str, text2: str) -> float:
    """
    Calculate fuzzy match score between two text strings.
    
    Args:
        text1: First text string
        text2: Second text string
        
    Returns:
        Similarity score between 0.0 and 1.0
    """
    norm_text1 = normalize_text(text1).lower()
    norm_text2 = normalize_text(text2).lower()
    
    if not norm_text1 or not norm_text2:
        return 0.0
    
    return SequenceMatcher(None, norm_text1, norm_text2).ratio()


class SemanticMatcher:
    """
    Semantic similarity matcher using sentence embeddings.

    Pre-computes embeddings for all candidates in batch so that per-sentence
    matching is a single dot-product (cosine similarity) instead of an O(N*L)
    character-level comparison.

    Usage:
        matcher = SemanticMatcher()
        matcher.index(aligned_dataset)          # once, at startup
        match, match_type, score = matcher.find_best(text, doc_indices, threshold)
    """

    MODEL_NAME = "all-MiniLM-L6-v2"  # 80MB, fast on CPU, strong semantic quality

    def __init__(self):
        if not _SEMANTIC_AVAILABLE:
            raise ImportError(
                "sentence-transformers is required for semantic matching. "
                "Install it with: pip install sentence-transformers"
            )
        logger.info(f"Loading sentence embedding model '{self.MODEL_NAME}'...")
        self._model = SentenceTransformer(self.MODEL_NAME)
        self._candidates: List[Dict] = []
        self._candidate_texts: List[str] = []
        self._embeddings: Optional["np.ndarray"] = None  # shape (N, D), L2-normalised

    def index(self, candidates: List[Dict]) -> None:
        """
        Pre-compute and store L2-normalised embeddings for all candidates.
        Call this once after loading the aligned dataset.
        """
        self._candidates = candidates
        self._candidate_texts = [
            normalize_text(c.get("text", "")).lower() for c in candidates
        ]
        logger.info(f"Encoding {len(candidates)} candidates (this runs once)...")
        self._embeddings = self._model.encode(
            self._candidate_texts,
            batch_size=64,
            show_progress_bar=len(candidates) > 500,
            normalize_embeddings=True,  # L2-normalise so cosine sim == dot product
        )
        logger.info("Candidate embeddings ready.")

    def find_best(
        self,
        query: str,
        indices: List[int],
        threshold: float = 0.85,
    ) -> Tuple[Optional[Dict], str, float]:
        """
        Find the best matching candidate from a pre-selected index subset.

        Args:
            query:     The sentence to match.
            indices:   Indices into the full candidate list to search within.
                       Pass an empty list to return 'unmatched' immediately
                       (no global fallback).
            threshold: Minimum cosine similarity to accept as a match.

        Returns:
            (best_candidate_dict | None, match_type, score)
            match_type is 'exact', 'semantic', or 'unmatched'.
        """
        if self._embeddings is None:
            raise RuntimeError("Call index() before find_best().")

        if not indices or not query.strip():
            return None, "unmatched", 0.0

        norm_q = normalize_text(query).lower()

        # 1. Exact string match (free, catches 100% overlaps)
        for i in indices:
            if self._candidate_texts[i] == norm_q:
                return self._candidates[i], "exact", 1.0

        # 2. Semantic similarity via dot product on L2-normalised embeddings
        q_emb = self._model.encode(norm_q, normalize_embeddings=True)
        sub_emb = self._embeddings[indices]       # (M, D)
        sims = sub_emb @ q_emb                    # (M,) cosine similarities
        best_local = int(np.argmax(sims))
        best_score = float(sims[best_local])

        if best_score >= threshold:
            return self._candidates[indices[best_local]], "semantic", best_score

        return None, "unmatched", best_score


def load_pure_annotated(csv_path: str) -> List[Dict]:
    """
    Load and parse PURE_Annotated CSV file.
    
    Args:
        csv_path: Path to PURE_Annotated CSV file
        
    Returns:
        List of dictionaries containing annotated requirements
    """
    logger.info(f"Loading PURE_Annotated from {csv_path}")
    
    annotated_data = []
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            # Parse the label - handle both numeric and text labels
            label_str = row.get('Req/Not Req', '').strip().lower()
            if label_str in ['1', 'yes', 'req']:
                label = 1
            elif label_str in ['0', 'no', 'not req']:
                label = 0
            else:
                logger.warning(f"Unknown label '{label_str}' in row, skipping")
                continue
            
            annotated_data.append({
                'text': row.get('Requirement', '').strip(),
                'doc_name': row.get('Name of Doc', '').strip(),
                'label': label,
                'original_label': label_str
            })
    
    logger.info(f"Loaded {len(annotated_data)} annotated requirements")
    return annotated_data


def generate_doc_id(doc_name: str) -> str:
    """
    Generate a standardized document ID from document name.
    
    Args:
        doc_name: Original document name
        
    Returns:
        Standardized document ID
    """
    # Remove file extension and special characters
    doc_id = re.sub(r'\.(pdf|doc|docx|htm|html)$', '', doc_name, flags=re.IGNORECASE)
    doc_id = re.sub(r'[^a-zA-Z0-9_-]', '_', doc_id)
    doc_id = doc_id.lower().strip('_')
    
    return f"pure_doc_{doc_id}"


def generate_block_id(index: int) -> str:
    """
    Generate a block ID for a given index.
    
    Args:
        index: Sequential index
        
    Returns:
        Block ID string
    """
    return f"b{index}"


def generate_sentence_id(block_id: str, sentence_index: int) -> str:
    """
    Generate a sentence ID within a block.
    
    Args:
        block_id: Block identifier
        sentence_index: Index of sentence within block
        
    Returns:
        Sentence ID string
    """
    return f"{block_id}_{sentence_index}"


def find_best_match(
    target_text: str,
    candidates: List[Dict],
    threshold: float = 0.85
) -> Tuple[Optional[Dict], str, float]:
    """
    Find the best matching candidate for a target text using fuzzy matching.
    
    Args:
        target_text: Text to match
        candidates: List of candidate dictionaries with 'text' field
        threshold: Minimum similarity threshold for fuzzy match
        
    Returns:
        Tuple of (best_match_dict, match_type, score)
        match_type can be 'exact', 'fuzzy', or 'unmatched'
    """
    if not target_text or not candidates:
        return None, 'unmatched', 0.0
    
    norm_target = normalize_text(target_text)
    best_match = None
    best_score = 0.0
    
    # First try exact match
    for candidate in candidates:
        if normalize_text(candidate['text']) == norm_target:
            return candidate, 'exact', 1.0
    
    # Try fuzzy match
    for candidate in candidates:
        score = fuzzy_match_score(target_text, candidate['text'])
        if score > best_score:
            best_score = score
            best_match = candidate
    
    if best_score >= threshold:
        return best_match, 'fuzzy', best_score
    
    return None, 'unmatched', best_score


def align_datasets(
    pure_annotated_data: List[Dict],
    fuzzy_threshold: float = 0.85
) -> List[Dict]:
    """
    Align PURE_Annotated data with generated IDs and metadata.
    
    Args:
        pure_annotated_data: List of annotated requirements
        fuzzy_threshold: Threshold for fuzzy matching
        
    Returns:
        List of aligned data dictionaries with required output format
    """
    logger.info("Aligning datasets...")
    
    aligned_data = []
    doc_groups = {}
    
    # Group by document
    for item in pure_annotated_data:
        doc_name = item['doc_name']
        if doc_name not in doc_groups:
            doc_groups[doc_name] = []
        doc_groups[doc_name].append(item)
    
    # Process each document
    for doc_name, items in doc_groups.items():
        doc_id = generate_doc_id(doc_name)
        
        for idx, item in enumerate(items):
            block_id = generate_block_id(idx)
            sentence_id = generate_sentence_id(block_id, 0)
            
            aligned_item = {
                'text': item['text'],
                'label': item['label'],
                'doc_id': doc_id,
                'block_id': block_id,
                'sentence_id': sentence_id,
                'annotated_text': item['text'],  # Same as text in this case
                'match_type': 'exact',  # Since we're using the annotated data directly
                'score': 1.0,
                'doc_name': doc_name
            }
            
            aligned_data.append(aligned_item)
    
    logger.info(f"Aligned {len(aligned_data)} requirements")
    
    # Log statistics
    label_counts = {0: 0, 1: 0}
    for item in aligned_data:
        label_counts[item['label']] += 1
    
    logger.info(f"Label distribution - Requirements: {label_counts[1]}, Non-requirements: {label_counts[0]}")
    
    return aligned_data


def save_aligned_dataset(output_path: str, aligned_data: List[Dict]) -> None:
    """
    Save aligned dataset to JSON file.
    
    Args:
        output_path: Path to output JSON file
        aligned_data: List of aligned data dictionaries
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(aligned_data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Saved aligned dataset to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Align PURE dataset with PURE_Annotated labels"
    )
    parser.add_argument(
        '--pure-csv',
        required=True,
        help='Path to PURE_Annotated CSV file'
    )
    parser.add_argument(
        '--output',
        required=True,
        help='Path to output aligned JSON file'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.85,
        help='Fuzzy match threshold (default: 0.85)'
    )
    
    args = parser.parse_args()
    
    # Load PURE_Annotated data
    annotated_data = load_pure_annotated(args.pure_csv)
    
    # Align datasets
    aligned_data = align_datasets(annotated_data, args.threshold)
    
    # Save results
    save_aligned_dataset(args.output, aligned_data)
    
    logger.info("Alignment complete!")
    
    # Print summary
    print("\n" + "="*60)
    print("ALIGNMENT SUMMARY")
    print("="*60)
    print(f"Total aligned items: {len(aligned_data)}")
    print(f"Requirements (label=1): {sum(1 for x in aligned_data if x['label'] == 1)}")
    print(f"Non-requirements (label=0): {sum(1 for x in aligned_data if x['label'] == 0)}")
    print(f"Output saved to: {args.output}")
    print("="*60)


if __name__ == "__main__":
    main()
