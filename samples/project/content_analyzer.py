"""
content_analyzer.py
====================
NLP-style text analysis utilities for the content moderation
and search indexing pipeline.
"""
from typing import Dict


def analyze_text(text: str) -> Dict[str, int]:
    """
    Extract linguistic statistics from a text document.

    Computes the following metrics in a single pass:
        "words"      : number of whitespace-separated tokens
        "sentences"  : sentence count (delimited by '.', '!', '?')
        "vowels"     : count of vowel characters (a e i o u, case-insensitive)
        "consonants" : count of consonant letter characters

    Args:
        text: Input string. May be empty or whitespace-only.

    Returns:
        Dictionary with integer values for each metric.
        All values are 0 for empty / whitespace-only input.

    >>> analyze_text("")
    {'words': 0, 'sentences': 0, 'vowels': 0, 'consonants': 0}
    >>> analyze_text("Hello world.")
    {'words': 2, 'sentences': 1, 'vowels': 3, 'consonants': 7}
    >>> analyze_text("Wow! Great job. Really?")
    {'words': 4, 'sentences': 3, 'vowels': 7, 'consonants': 9}
    """
    if not text.strip():
        return {"words": 0, "sentences": 0, "vowels": 0, "consonants": 0}

    words = len(text.split())
    sentences = sum(text.count(p) for p in ".!?")
    lower = text.lower()
    vowels = sum(1 for c in lower if c in "aeiou")
    consonants = sum(1 for c in lower if c.isalpha() and c not in "aeiou")

    return {
        "words":      words,
        "sentences":  sentences,
        "vowels":     vowels,
        "consonants": consonants,
    }


def readability_grade(text: str) -> str:
    """
    Estimate the readability grade level of a text using a simplified
    Flesch-Kincaid-inspired heuristic based on average sentence and word length.

    Grade bands (based on avg characters per word):
        avg_chars_per_word >= 7 : "advanced"
        avg_chars_per_word >= 5 : "intermediate"
        avg_chars_per_word >= 3 : "basic"
        avg_chars_per_word <  3 : "elementary"

    Args:
        text: Input text with at least one word.

    Returns:
        One of: "elementary", "basic", "intermediate", "advanced".
        Returns "elementary" for empty or whitespace-only input.

    >>> readability_grade("Hi!")
    'elementary'
    >>> readability_grade("The cat sat on a mat.")
    'basic'
    >>> readability_grade("Software engineering requires systematic thinking.")
    'intermediate'
    >>> readability_grade("Electroencephalography demonstrates neurological complexity.")
    'advanced'
    """
    words = text.split()
    if not words:
        return "elementary"

    avg_chars = sum(len(w.strip(".,!?;:")) for w in words) / len(words)

    if avg_chars >= 7:
        return "advanced"
    elif avg_chars >= 5:
        return "intermediate"
    elif avg_chars >= 3:
        return "basic"
    else:
        return "elementary"
