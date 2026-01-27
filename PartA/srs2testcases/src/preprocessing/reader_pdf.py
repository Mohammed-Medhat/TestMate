# src/preprocessing/reader_pdf.py
"""
PDF reader that extracts text blocks from PDF pages and writes a unified JSON
schema compatible with reader_docx outputs.

Outputs (default): data/raw/<doc_id>.json

Dependencies:
  pip install PyMuPDF pillow pytesseract regex

On Windows, make sure Tesseract OCR is installed if you want OCR fallback.
"""

import fitz  # PyMuPDF
import json
import os
import re
from pathlib import Path
from PIL import Image
import io

try:
    import pytesseract
    OCR_AVAILABLE = True
except Exception:
    OCR_AVAILABLE = False

_whitespace_re = re.compile(r"\s+")
_hyphen_re = re.compile(r"(\w+)-\s*\n\s*(\w+)")

def clean_text(text: str) -> str:
    if text is None:
        return ""
    # fix hyphenation across lines, collapse whitespace
    text = _hyphen_re.sub(r"\1\2", text)
    text = _whitespace_re.sub(" ", text)
    return text.strip()

def page_image_to_text(page, zoom=2):
    """Render page to image and run pytesseract OCR. Returns text or empty string."""
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
    mode = "RGB" if pix.alpha == 0 else "RGBA"
    img = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
    if not OCR_AVAILABLE:
        return ""
    try:
        text = pytesseract.image_to_string(img)
    except Exception:
        text = ""
    return text

def extract_blocks_from_pdf(path, ocr_if_empty=True, max_pages=None):
    """
    Returns dict:
    {
      "file_path": str,
      "doc_id": str,
      "num_pages": int,
      "blocks": [ {block_id, type, text, raw_text, page, bbox, source, paragraph_index } ]
    }
    """
    doc = fitz.open(path)
    doc_id = Path(path).stem
    blocks_all = []
    block_counter = 0

    for page_index in range(len(doc)):
        if max_pages is not None and page_index >= max_pages:
            break
        page = doc[page_index]
        page_num = page_index + 1

        # get structured text (blocks->lines->spans)
        page_dict = page.get_text("dict")
        page_blocks = page_dict.get("blocks", [])

        # determine if this page has any text blocks
        has_text = any(pb.get("type", 0) == 0 and any(span.get("text","").strip() for line in pb.get("lines",[]) for span in line.get("spans",[])) for pb in page_blocks)

        if not has_text and ocr_if_empty:
            ocr_text = page_image_to_text(page)
            raw = ocr_text.strip()
            if raw:
                text = clean_text(raw)
                blocks_all.append({
                    "block_id": f"p{page_num}_b{block_counter}",
                    "type": "paragraph",
                    "text": text,
                    "raw_text": raw,
                    "style": None,
                    "source": str(path),
                    "paragraph_index": block_counter,
                    "page": page_num,
                    "bbox": None
                })
                block_counter += 1
            continue

        for pb in page_blocks:
            # Only handle text blocks (type == 0)
            if pb.get("type", 0) != 0:
                continue
            spans = []
            for line in pb.get("lines", []):
                line_text = "".join(span.get("text","") for span in line.get("spans", []))
                spans.append(line_text)
            raw = "\n".join(spans).strip()
            if not raw:
                continue
            text = clean_text(raw)
            bbox = pb.get("bbox", None)
            # heuristics for block type
            block_type = "paragraph"
            if re.match(r"^\s*(\d+[\.\)]|\-|\u2022|\*)\s+", text):
                block_type = "list_item"
            elif len(text.split()) <= 6 and text == text.title():
                block_type = "heading"
            blocks_all.append({
                "block_id": f"p{page_num}_b{block_counter}",
                "type": block_type,
                "text": text,
                "raw_text": raw,
                "style": None,
                "source": str(path),
                "paragraph_index": block_counter,
                "page": page_num,
                "bbox": bbox
            })
            block_counter += 1

    return {
        "file_path": str(path),
        "doc_id": doc_id,
        "num_pages": len(doc),
        "blocks": blocks_all
    }

def save_json(data, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract text blocks from PDF and save JSON")
    parser.add_argument("pdf", help="path to pdf file")
    parser.add_argument("--out", help="output json path (default=data/raw/<doc_id>.json)", default=None)
    parser.add_argument("--no-ocr", action="store_true", help="disable OCR fallback")
    parser.add_argument("--max-pages", type=int, default=None, help="limit pages (for debug)")
    args = parser.parse_args()

    pdf_path = args.pdf
    doc = extract_blocks_from_pdf(pdf_path, ocr_if_empty=not args.no_ocr, max_pages=args.max_pages)
    if args.out:
        out_path = args.out
    else:
        base = Path(pdf_path).stem
        out_path = os.path.join("data", "raw", f"{base}.json")
    save_json(doc, out_path)
    print(f"Saved {len(doc['blocks'])} blocks to {out_path}")
