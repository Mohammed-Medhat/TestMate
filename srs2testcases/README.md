# TestMate Stage 1: Setup Instructions

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

## Installation Steps

### 1. Install Dependencies

```bash
cd "E:\Omar uni\Semester 7\GP\TestMate\srs2testcases"
pip install -r requirements.txt
```

### 2. Download spaCy Language Model

```bash
python -m spacy download en_core_web_sm
```

### 3. Verify Installation

```bash
python scripts\test_alignment.py
```

Expected output: All tests should pass ✓

---

## Quick Start

### Step 1: Align PURE Datasets

First, create aligned datasets from PURE_Annotated CSV files:

```bash
python scripts\align_all_datasets.py
```

This creates:
- `data/aligned/pure_train_aligned.json`
- `data/aligned/pure_valid_aligned.json`
- `data/aligned/pure_test_aligned.json`

### Step 2: Process SRS Documents

Process a single document:

```bash
python run_pipeline.py --input "E:\Omar uni\Semester 7\GP\TestMate\req\2011 - opensg 0.1.docx" --output results\opensg_requirements.json --aligned-dataset data\aligned\pure_train_aligned.json
```

Or batch process all documents:

```bash
python run_pipeline.py --batch --input-dir "E:\Omar uni\Semester 7\GP\TestMate\req" --output-dir results --aligned-dataset data\aligned\pure_train_aligned.json
```

### Step 3: Verify Results

```bash
python scripts\verify_alignment.py
```

---

## Directory Structure

```
srs2testcases/
├── run_pipeline.py              # Main pipeline script
├── requirements.txt             # Python dependencies
├── PIPELINE_USAGE.md           # Detailed usage guide
├── README.md                   # This file
│
├── src/
│   ├── preprocessing/
│   │   ├── align_pure_datasets.py    # Dataset alignment module
│   │   ├── segment_sentences.py      # Sentence segmentation
│   │   ├── reader_pdf.py            # PDF extraction
│   │   └── reader_docx.py           # DOCX extraction
│   │
│   ├── models/                  # (Future: ML models)
│   ├── generation/              # (Future: Test case generation)
│   └── validation/              # (Future: Validation)
│
├── scripts/
│   ├── align_all_datasets.py   # Batch alignment script
│   ├── test_alignment.py       # Alignment tests
│   ├── verify_alignment.py     # Verification tool
│   └── test_pipeline.py        # Pipeline test
│
├── data/
│   ├── raw/                    # Extracted text from SRS
│   ├── processed/              # Segmented sentences
│   └── aligned/                # Aligned datasets
│
└── results/                    # Pipeline output
```

---

## Troubleshooting

### Issue: ModuleNotFoundError

```bash
# Install missing package
pip install <package-name>

# Or reinstall all requirements
pip install -r requirements.txt
```

### Issue: spaCy model not found

```bash
python -m spacy download en_core_web_sm
```

### Issue: Permission denied

Run terminal as administrator or use:

```bash
pip install --user -r requirements.txt
```

---

## What's Next?

After completing Stage 1 (alignment), you can:

1. **Train Classification Model** - Use aligned data to train requirement classifier
2. **Functional/Non-Functional Classification** - Extend to classify requirement types
3. **Test Case Generation** - Generate test cases from classified requirements

See `implementation_plan.md` for details on Stage 2.

---

## Support

For issues or questions:
1. Check `PIPELINE_USAGE.md` for detailed usage examples
2. Run verification scripts to diagnose problems
3. Review log output for error details
