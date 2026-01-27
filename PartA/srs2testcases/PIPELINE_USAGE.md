# Pipeline Usage Guide

## Quick Start

### Single Document Processing

Process a single SRS document:

```bash
cd "E:\Omar uni\Semester 7\GP\TestMate\srs2testcases"

# Without aligned dataset (unlabeled output)
python run_pipeline.py --input "E:\Omar uni\Semester 7\GP\TestMate\req\2011 - opensg 0.1.docx" --output results\opensg_requirements.json

# With aligned dataset (labeled output via fuzzy matching)
python run_pipeline.py --input "E:\Omar uni\Semester 7\GP\TestMate\req\2011 - opensg 0.1.docx" --output results\opensg_requirements.json --aligned-dataset data\aligned\pure_train_aligned.json
```

### Batch Processing

Process all documents in a directory:

```bash
# Process all PDFs and DOCX files in req directory
python run_pipeline.py --batch --input-dir "E:\Omar uni\Semester 7\GP\TestMate\req" --output-dir results --aligned-dataset data\aligned\pure_train_aligned.json
```

### Custom Fuzzy Threshold

Adjust the fuzzy matching threshold (default: 0.85):

```bash
python run_pipeline.py --input document.pdf --output output.json --aligned-dataset data\aligned\pure_train_aligned.json --threshold 0.90
```

---

## Pipeline Workflow

The pipeline performs these steps automatically:

1. **Extract** - Reads PDF/DOCX and extracts text with structure
2. **Segment** - Splits text into sentences using spaCy
3. **Align** - Matches sentences against PURE dataset (if provided)
4. **Label** - Assigns requirement labels based on matches
5. **Save** - Outputs structured JSON with all metadata

---

## Output Format

Each processed sentence includes:

```json
{
  "text": "The system shall provide...",
  "label": 1,
  "doc_id": "pure_doc_opensg_0_1",
  "block_id": "b0",
  "sentence_id": "b0_0",
  "annotated_text": "The system shall provide...",
  "match_type": "exact",
  "score": 1.0,
  "doc_name": "opensg 0.1.docx",
  "paragraph_index": 0,
  "page": 1,
  "token_count": 15
}
```

**Label Values:**
- `1` = Requirement
- `0` = Non-requirement
- `-1` = Unknown (no aligned dataset or no match found)

**Match Types:**
- `exact` = Perfect text match (score = 1.0)
- `fuzzy` = Similar text match (score >= threshold)
- `unmatched` = No good match found (score < threshold)

---

## Examples

### Example 1: Process with Labeling

```bash
python run_pipeline.py \
  --input "E:\Omar uni\Semester 7\GP\TestMate\req\0000 - cctns.pdf" \
  --output results\cctns_requirements.json \
  --aligned-dataset data\aligned\pure_train_aligned.json \
  --threshold 0.85
```

**Expected Output:**
```
Processing: 0000 - cctns.pdf
Extracting from 0000 - cctns.pdf
Segmenting sentences...
Aligning sentences...
Saved 150 results to results\cctns_requirements.json

PROCESSING SUMMARY
Document: 0000 - cctns.pdf
Total sentences: 150
Requirements: 85
Non-requirements: 45
Unlabeled: 20
```

### Example 2: Batch Processing

```bash
python run_pipeline.py \
  --batch \
  --input-dir "E:\Omar uni\Semester 7\GP\TestMate\req" \
  --output-dir results\batch_output \
  --aligned-dataset data\aligned\pure_train_aligned.json
```

---

## Tips

1. **First Time Setup**: Run alignment first to create the aligned dataset:
   ```bash
   python scripts\align_all_datasets.py
   ```

2. **Check Dependencies**: Ensure spaCy model is installed:
   ```bash
   python -m spacy download en_core_web_sm
   ```

3. **Large Documents**: Processing may take 1-2 minutes for large PDFs

4. **Threshold Tuning**: 
   - Lower threshold (0.75-0.80) = More matches, less precise
   - Higher threshold (0.90-0.95) = Fewer matches, more precise

---

## Troubleshooting

**Issue**: "No module named 'spacy'"
```bash
pip install spacy
python -m spacy download en_core_web_sm
```

**Issue**: "FileNotFoundError: aligned dataset"
```bash
# Run alignment first
python scripts\align_all_datasets.py
```

**Issue**: "Unsupported file type"
- Only PDF and DOCX files are supported
- Check file extension is correct

---

## Next Steps

After processing documents:

1. **Review Output**: Check `results/` directory for JSON files
2. **Analyze Labels**: Use verification script to see label distribution
3. **Train Classifier**: Use aligned data to train ML model
4. **Iterate**: Adjust threshold and reprocess if needed
