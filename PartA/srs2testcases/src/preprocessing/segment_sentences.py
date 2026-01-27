# src/preprocessing/segment_sentences.py
"""
Segment blocks (from reader_docx/reader_pdf output) into sentences with noise filtering.

Input: data/raw/<doc>.json (same schema produced by reader_docx/reader_pdf)
Output: data/processed/<doc>_sentences.json

Each sentence object:
{
  "doc_id": "...",
  "block_id": "b12",
  "sentence_id": "b12_0",
  "paragraph_index": 12 (optional),
  "text": "...",
  "clean_text": "...",
  "token_count": N,
  "char_start": int,
  "char_end": int,
  "block_text": "...",
  "block_type": "paragraph"|"table_row"|...
}
"""

import json
import os
from pathlib import Path
import argparse
import logging
import re

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("segmenter")

# Toggle this if you want to save raw (pre-filter) sentences for debugging
SAVE_RAW = True

# Try to import spaCy
SPACY_AVAILABLE = True
try:
    import spacy
except Exception:
    SPACY_AVAILABLE = False

# fallback sentence splitter (simple)
_sent_split_re = re.compile(r'(?<=[\.\?\!])\s+')

# Filters: compiled regexes to detect noise
_ONLY_SYMBOLS_RE = re.compile(r'^[\W_]+$')            # only punctuation/symbols
_HAS_LETTER_RE = re.compile(r'[A-Za-z\u00C0-\u024F]') # any letter (includes many unicode letters)
_TABLE_SEP_RE = re.compile(r'^[\|\-–—\:\;]+$')        # lines that are only separators
_TOKEN_NUMERIC_RE = re.compile(r'^[\d\W_]+$')         # tokens that are only digits/punct
_SHORT_TOKEN_RE = re.compile(r'^[A-Za-z]{1,2}$')      # one- or two-letter words (tweakable)
_DATE_RE = re.compile(r'^(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[\w\d\-/., ]*$', re.I)
_SINGLE_CHAR_RE = re.compile(r'^\W?\w\W?$')           # single character tokens (with possible punctuation)
_LEADING_PIPE_RE = re.compile(r'^\|+')                # starts with pipe

def normalize_whitespace(s: str) -> str:
    return " ".join(s.split()).strip()

def clean_text(s: str) -> str:
    if s is None:
        return ""
    s = s.replace('\xa0',' ')
    s = s.replace("’","'").replace("“", '"').replace("”", '"')
    s = normalize_whitespace(s)
    # remove stray leading/trailing pipes/spaces
    s = s.strip(" \t\n\r\x0b\x0c|")
    return s

def is_valid_sentence(s: str, token_count: int) -> bool:
    """
    Return True if sentence s should be kept for training/inference.
    Heuristics:
     - Must contain at least one alphabetic character (letters)
     - Must not be only punctuation/symbols
     - Must not be only table separators like '|' or '---'
     - Must not be a single short token (numeric id, single char, etc.)
     - Must not be mostly numeric/dates or page markers
     - Must have at least 2 tokens (by default) unless it passes letter-based checks
    """
    if s is None:
        return False
    s = s.strip()
    if not s:
        return False

    # if starts/ends with pipe or composed of pipes => junk
    if _LEADING_PIPE_RE.match(s) or _TABLE_SEP_RE.match(s):
        return False

    # if contains no letters at all => junk (dates like "2011" or "v0.1")
    if not _HAS_LETTER_RE.search(s):
        return False

    # only symbols?
    if _ONLY_SYMBOLS_RE.match(s):
        return False

    # token count rule: require >= 2 tokens with letters, but allow short phrases like "Log in"
    if token_count < 2:
        # accept some short meaningful tokens (e.g., "Logout", "Login") if they contain letters and are longer than 2 chars
        if len(s) <= 3:
            return False
        # otherwise allow short but letter-heavy tokens
    # filter single-letter tokens
    if _SHORT_TOKEN_RE.match(s) and token_count <= 2:
        return False

    # filter tokens that are mostly numeric / punct
    # example: "January252011" might be considered noise if it's a date stuck to a token
    tokens = s.split()
    numeric_like = sum(1 for t in tokens if _TOKEN_NUMERIC_RE.match(t))
    if numeric_like == len(tokens):
        return False

    # if sentence is just a month name or date-like short fragment (common in tables)
    if len(tokens) <= 3 and _DATE_RE.match(s):
        return False

    # catch patterns like single char (e.g., "N", "Y") or "| N"
    if len(s) <= 3 and _SINGLE_CHAR_RE.match(s):
        # allow some short acronyms like "API" (3 letters) -> this won't be filtered (HAS_LETTER + len>3)
        return False

    return True

def segment_block_with_spacy(nlp, block_text):
    doc = nlp(block_text)
    sents = []
    for sent in doc.sents:
        sents.append((sent.start_char, sent.end_char, sent.text.strip(), len(list(sent))))
    return sents

def segment_block_fallback(block_text):
    pieces = _sent_split_re.split(block_text)
    sents = []
    pos = 0
    for p in pieces:
        start = block_text.find(p, pos)
        if start == -1:
            start = pos
        end = start + len(p)
        text = p.strip()
        if text:
            tok_count = len(text.split())
            sents.append((start, end, text, tok_count))
        pos = end
    return sents

def load_doc_json(path: Path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)

def segment_document(raw_json_path: str, out_dir: str = "data/processed", model_name: str = "en_core_web_sm", include_context=False):
    p = Path(raw_json_path)
    doc = load_doc_json(p)
    doc_id = doc.get("doc_id") or p.stem
    blocks = doc.get("blocks", [])
    sentences = []
    raw_sentences = []  # keep for optional saving/debug

    # prepare spaCy
    nlp = None
    if SPACY_AVAILABLE:
        try:
            nlp = spacy.load(model_name, disable=["ner", "textcat"])
            # ensure sentence boundary detection active
            if "parser" not in nlp.pipe_names and "senter" not in nlp.pipe_names:
                try:
                    nlp.add_pipe("senter")
                except Exception:
                    pass
        except Exception as e:
            logger.warning("Could not load spaCy model '%s' (%s). Falling back to regex.", model_name, e)
            nlp = None

    for block in blocks:
        block_id = block.get("block_id")
        block_text = block.get("text") or ""
        paragraph_index = block.get("paragraph_index", None)
        block_type = block.get("type") or block.get("block_type") or None
        page = block.get("page") or block.get("pageno") or None

        if not block_text:
            continue

        if nlp:
            segs = segment_block_with_spacy(nlp, block_text)
        else:
            segs = segment_block_fallback(block_text)

        for idx, (char_start, char_end, sent_text, token_count) in enumerate(segs):
            sent_text_raw = sent_text
            sent_text = clean_text(sent_text_raw)
            if not sent_text:
                continue

            sentence_id = f"{block_id}_{idx}"
            sent_obj = {
                "doc_id": doc_id,
                "block_id": block_id,
                "sentence_id": sentence_id,
                "paragraph_index": paragraph_index,
                "page": page,
                "text": sent_text,
                "clean_text": sent_text,
                "token_count": token_count,
                "char_start": int(char_start),
                "char_end": int(char_end),
                "block_text": block_text,
                "block_type": block_type
            }

            raw_sentences.append(sent_obj)

            # apply filtering heuristic
            if not is_valid_sentence(sent_text, token_count):
                # skip noisy sentence
                continue

            sentences.append(sent_obj)

    # optional: include prev/next context if requested
    if include_context:
        for i, s in enumerate(sentences):
            s["prev_sentence"] = sentences[i - 1]["text"] if i > 0 else None
            s["next_sentence"] = sentences[i + 1]["text"] if i < len(sentences) - 1 else None

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{doc_id}_sentences.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(sentences, fh, ensure_ascii=False, indent=2)

    if SAVE_RAW:
        raw_out = out_dir / f"{doc_id}_sentences_raw.json"
        with open(raw_out, "w", encoding="utf-8") as fh:
            json.dump(raw_sentences, fh, ensure_ascii=False, indent=2)
        logger.info("Saved raw (pre-filter) sentences to %s", raw_out)

    logger.info("Saved %d filtered sentences to %s (raw: %d)", len(sentences), out_path, len(raw_sentences))
    return str(out_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="path to raw JSON produced by reader_docx/reader_pdf")
    parser.add_argument("--outdir", default="data/processed", help="output directory")
    parser.add_argument("--model", default="en_core_web_sm", help="spaCy model name (if installed)")
    parser.add_argument("--context", action="store_true", help="include prev/next sentence")
    args = parser.parse_args()
    segment_document(args.input, args.outdir, args.model, args.context)

if __name__ == "__main__":
    main()
