#!/usr/bin/env python
# run_pipeline.py
"""
End-to-end pipeline for TestMate Stage 1: Requirement Extraction and Alignment

This script orchestrates the complete workflow:
1. Extract sentences from SRS documents (PDF/DOCX)
2. Align extracted sentences with PURE_Annotated dataset
3. Generate structured output with labels and metadata

Usage:
    python run_pipeline.py --input <srs_file> --output <output_json>
    python run_pipeline.py --batch --input-dir <srs_dir> --output-dir <output_dir>
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Dict, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from preprocessing.reader_pdf import extract_text_from_pdf
from preprocessing.reader_docx import extract_text_from_docx
from preprocessing.segment_sentences import segment_document
from preprocessing.align_pure_datasets import (
    normalize_text,
    fuzzy_match_score,
    generate_doc_id,
    generate_block_id,
    generate_sentence_id
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("run_pipeline")


class RequirementExtractionPipeline:
    """
    End-to-end pipeline for extracting and aligning requirements from SRS documents.
    """
    
    def __init__(self, aligned_dataset_path: Optional[str] = None):
        """
        Initialize the pipeline.
        
        Args:
            aligned_dataset_path: Path to aligned PURE dataset for reference matching
        """
        self.aligned_dataset = None
        if aligned_dataset_path and Path(aligned_dataset_path).exists():
            self.load_aligned_dataset(aligned_dataset_path)
    
    def load_aligned_dataset(self, path: str):
        """Load aligned PURE dataset for reference."""
        logger.info(f"Loading aligned dataset from {path}")
        with open(path, 'r', encoding='utf-8') as f:
            self.aligned_dataset = json.load(f)
        logger.info(f"Loaded {len(self.aligned_dataset)} aligned requirements")
    
    def extract_from_document(self, doc_path: str, output_dir: str = "data/raw") -> str:
        """
        Extract text and structure from SRS document.
        
        Args:
            doc_path: Path to SRS document (PDF or DOCX)
            output_dir: Directory to save extracted JSON
            
        Returns:
            Path to extracted JSON file
        """
        doc_path = Path(doc_path)
        
        if not doc_path.exists():
            raise FileNotFoundError(f"Document not found: {doc_path}")
        
        logger.info(f"Extracting from {doc_path.name}")
        
        # Determine file type and extract
        if doc_path.suffix.lower() == '.pdf':
            json_path = extract_text_from_pdf(str(doc_path), output_dir)
        elif doc_path.suffix.lower() in ['.doc', '.docx']:
            json_path = extract_text_from_docx(str(doc_path), output_dir)
        else:
            raise ValueError(f"Unsupported file type: {doc_path.suffix}")
        
        logger.info(f"Extracted to {json_path}")
        return json_path
    
    def segment_sentences(self, raw_json_path: str, output_dir: str = "data/processed") -> str:
        """
        Segment extracted text into sentences.
        
        Args:
            raw_json_path: Path to raw extracted JSON
            output_dir: Directory to save segmented sentences
            
        Returns:
            Path to segmented sentences JSON
        """
        logger.info(f"Segmenting sentences from {raw_json_path}")
        
        sentences_path = segment_document(
            raw_json_path,
            out_dir=output_dir,
            model_name="en_core_web_sm",
            include_context=False
        )
        
        logger.info(f"Segmented to {sentences_path}")
        return sentences_path
    
    def align_and_label(
        self,
        sentences_path: str,
        doc_name: str,
        fuzzy_threshold: float = 0.85
    ) -> List[Dict]:
        """
        Align extracted sentences with PURE dataset and assign labels.
        
        Args:
            sentences_path: Path to segmented sentences JSON
            doc_name: Original document name
            fuzzy_threshold: Threshold for fuzzy matching
            
        Returns:
            List of aligned and labeled sentences
        """
        logger.info(f"Aligning sentences from {sentences_path}")
        
        # Load segmented sentences
        with open(sentences_path, 'r', encoding='utf-8') as f:
            sentences = json.load(f)
        
        doc_id = generate_doc_id(doc_name)
        aligned_results = []
        
        # If we have aligned dataset, try to match against it
        if self.aligned_dataset:
            logger.info("Using aligned dataset for reference matching")
            aligned_results = self._match_with_aligned_dataset(
                sentences, doc_id, doc_name, fuzzy_threshold
            )
        else:
            # No aligned dataset - create basic structure without labels
            logger.info("No aligned dataset - creating unlabeled output")
            aligned_results = self._create_unlabeled_output(
                sentences, doc_id, doc_name
            )
        
        logger.info(f"Aligned {len(aligned_results)} sentences")
        return aligned_results
    
    def _match_with_aligned_dataset(
        self,
        sentences: List[Dict],
        doc_id: str,
        doc_name: str,
        threshold: float
    ) -> List[Dict]:
        """Match sentences against aligned PURE dataset."""
        results = []
        
        # Filter aligned dataset by document if possible
        doc_aligned = [
            item for item in self.aligned_dataset
            if item.get('doc_name', '').lower() in doc_name.lower() or
               doc_name.lower() in item.get('doc_name', '').lower()
        ]
        
        if not doc_aligned:
            logger.warning(f"No matching document found in aligned dataset for {doc_name}")
            doc_aligned = self.aligned_dataset
        
        logger.info(f"Matching against {len(doc_aligned)} reference sentences")
        
        for sent in sentences:
            text = sent.get('text', '')
            
            # Find best match in aligned dataset
            best_match = None
            best_score = 0.0
            
            for aligned_item in doc_aligned:
                score = fuzzy_match_score(text, aligned_item['text'])
                if score > best_score:
                    best_score = score
                    best_match = aligned_item
            
            # Determine match type and label
            if best_score >= threshold:
                match_type = 'exact' if best_score == 1.0 else 'fuzzy'
                label = best_match['label']
                annotated_text = best_match['text']
            else:
                match_type = 'unmatched'
                label = -1  # Unknown label
                annotated_text = text
            
            result = {
                'text': text,
                'label': label,
                'doc_id': doc_id,
                'block_id': sent.get('block_id', ''),
                'sentence_id': sent.get('sentence_id', ''),
                'annotated_text': annotated_text,
                'match_type': match_type,
                'score': best_score,
                'doc_name': doc_name,
                'paragraph_index': sent.get('paragraph_index'),
                'page': sent.get('page'),
                'token_count': sent.get('token_count', 0)
            }
            
            results.append(result)
        
        return results
    
    def _create_unlabeled_output(
        self,
        sentences: List[Dict],
        doc_id: str,
        doc_name: str
    ) -> List[Dict]:
        """Create output structure without labels."""
        results = []
        
        for sent in sentences:
            result = {
                'text': sent.get('text', ''),
                'label': -1,  # Unknown
                'doc_id': doc_id,
                'block_id': sent.get('block_id', ''),
                'sentence_id': sent.get('sentence_id', ''),
                'annotated_text': sent.get('text', ''),
                'match_type': 'unmatched',
                'score': 0.0,
                'doc_name': doc_name,
                'paragraph_index': sent.get('paragraph_index'),
                'page': sent.get('page'),
                'token_count': sent.get('token_count', 0)
            }
            results.append(result)
        
        return results
    
    def save_results(self, results: List[Dict], output_path: str):
        """Save aligned results to JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved {len(results)} results to {output_path}")
    
    def process_document(
        self,
        doc_path: str,
        output_path: str,
        fuzzy_threshold: float = 0.85
    ) -> Dict:
        """
        Process a single SRS document through the complete pipeline.
        
        Args:
            doc_path: Path to SRS document
            output_path: Path to save final output
            fuzzy_threshold: Threshold for fuzzy matching
            
        Returns:
            Dictionary with processing statistics
        """
        doc_path = Path(doc_path)
        
        logger.info("="*60)
        logger.info(f"Processing: {doc_path.name}")
        logger.info("="*60)
        
        # Step 1: Extract text
        raw_json = self.extract_from_document(str(doc_path))
        
        # Step 2: Segment sentences
        sentences_json = self.segment_sentences(raw_json)
        
        # Step 3: Align and label
        results = self.align_and_label(
            sentences_json,
            doc_path.name,
            fuzzy_threshold
        )
        
        # Step 4: Save results
        self.save_results(results, output_path)
        
        # Generate statistics
        stats = {
            'document': doc_path.name,
            'total_sentences': len(results),
            'labeled_sentences': sum(1 for r in results if r['label'] != -1),
            'requirements': sum(1 for r in results if r['label'] == 1),
            'non_requirements': sum(1 for r in results if r['label'] == 0),
            'unlabeled': sum(1 for r in results if r['label'] == -1),
            'match_types': {
                'exact': sum(1 for r in results if r['match_type'] == 'exact'),
                'fuzzy': sum(1 for r in results if r['match_type'] == 'fuzzy'),
                'unmatched': sum(1 for r in results if r['match_type'] == 'unmatched')
            },
            'output_path': str(output_path)
        }
        
        return stats


def main():
    parser = argparse.ArgumentParser(
        description="TestMate Stage 1: Requirement Extraction Pipeline"
    )
    
    # Input options
    parser.add_argument(
        '--input',
        help='Path to input SRS document (PDF or DOCX)'
    )
    parser.add_argument(
        '--output',
        help='Path to output JSON file'
    )
    
    # Batch processing
    parser.add_argument(
        '--batch',
        action='store_true',
        help='Process multiple documents from a directory'
    )
    parser.add_argument(
        '--input-dir',
        help='Directory containing SRS documents (for batch mode)'
    )
    parser.add_argument(
        '--output-dir',
        help='Directory to save output files (for batch mode)'
    )
    
    # Optional parameters
    parser.add_argument(
        '--aligned-dataset',
        help='Path to aligned PURE dataset for reference matching'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.85,
        help='Fuzzy match threshold (default: 0.85)'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.batch and (not args.input or not args.output):
        parser.error("--input and --output are required for single document mode")
    
    if args.batch and (not args.input_dir or not args.output_dir):
        parser.error("--input-dir and --output-dir are required for batch mode")
    
    # Initialize pipeline
    pipeline = RequirementExtractionPipeline(args.aligned_dataset)
    
    if args.batch:
        # Batch processing
        input_dir = Path(args.input_dir)
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all SRS documents
        doc_files = list(input_dir.glob('*.pdf')) + list(input_dir.glob('*.docx'))
        
        logger.info(f"Found {len(doc_files)} documents to process")
        
        all_stats = []
        for doc_file in doc_files:
            output_file = output_dir / f"{doc_file.stem}_requirements.json"
            
            try:
                stats = pipeline.process_document(
                    str(doc_file),
                    str(output_file),
                    args.threshold
                )
                all_stats.append(stats)
            except Exception as e:
                logger.error(f"Error processing {doc_file.name}: {e}")
                import traceback
                traceback.print_exc()
        
        # Print summary
        print("\n" + "="*60)
        print("BATCH PROCESSING SUMMARY")
        print("="*60)
        for stats in all_stats:
            print(f"\n{stats['document']}:")
            print(f"  Total sentences: {stats['total_sentences']}")
            print(f"  Requirements: {stats['requirements']}")
            print(f"  Non-requirements: {stats['non_requirements']}")
            print(f"  Unlabeled: {stats['unlabeled']}")
        
    else:
        # Single document processing
        stats = pipeline.process_document(
            args.input,
            args.output,
            args.threshold
        )
        
        # Print summary
        print("\n" + "="*60)
        print("PROCESSING SUMMARY")
        print("="*60)
        print(f"Document: {stats['document']}")
        print(f"Total sentences: {stats['total_sentences']}")
        print(f"Requirements: {stats['requirements']}")
        print(f"Non-requirements: {stats['non_requirements']}")
        print(f"Unlabeled: {stats['unlabeled']}")
        print(f"\nMatch types:")
        print(f"  Exact: {stats['match_types']['exact']}")
        print(f"  Fuzzy: {stats['match_types']['fuzzy']}")
        print(f"  Unmatched: {stats['match_types']['unmatched']}")
        print(f"\nOutput saved to: {stats['output_path']}")
        print("="*60)


if __name__ == "__main__":
    main()
