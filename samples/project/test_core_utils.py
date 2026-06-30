"""
test_core_utils.py — test suite for core_utils.py
Used for the Part C SBFL bug-detection demo.
"""
import pytest
from core_utils import password_strength_score, apply_discount, classify_text_length


# ── password_strength_score ───────────────────────────────────────────

def test_score_empty():
    assert password_strength_score("") == 0

def test_score_lowercase_only():
    assert password_strength_score("abc") == 1

def test_score_short_mixed():
    assert password_strength_score("Abc1!") == 4

def test_score_full():
    assert password_strength_score("Secure1!") == 5

def test_score_length_boundary():
    # exactly 8 chars, lowercase only — length point awarded
    assert password_strength_score("abcdefgh") == 2


# ── apply_discount ────────────────────────────────────────────────────

def test_no_discount():
    assert apply_discount(10.0, 5) == 50.0

def test_five_percent_tier():
    # quantity=10 → 5% off 100.0 = 95.0
    assert apply_discount(10.0, 10) == 95.0

def test_ten_percent_tier():
    # quantity=20 → 10% off 200.0 = 180.0
    assert apply_discount(10.0, 20) == 180.0

def test_fifteen_percent_tier():
    # quantity=50 → 15% off 500.0 = 425.0
    assert apply_discount(10.0, 50) == 425.0

def test_twenty_percent_tier():
    # quantity=100 → 20% off 1000.0 = 800.0
    assert apply_discount(10.0, 100) == 800.0

def test_boundary_below_ten():
    assert apply_discount(5.0, 9) == 45.0

def test_boundary_exactly_twenty():
    assert apply_discount(10.0, 20) == 180.0

def test_fractional_price():
    assert apply_discount(9.99, 1) == 9.99


# ── classify_text_length ──────────────────────────────────────────────

def test_empty_string():
    assert classify_text_length("") == "empty"

def test_single_word():
    assert classify_text_length("hello") == "short"

def test_ten_words():
    assert classify_text_length(" ".join(["x"] * 10)) == "short"

def test_eleven_words():
    assert classify_text_length(" ".join(["x"] * 11)) == "medium"

def test_fifty_words():
    assert classify_text_length(" ".join(["x"] * 50)) == "medium"

def test_fifty_one_words():
    assert classify_text_length(" ".join(["x"] * 51)) == "long"

def test_long_text():
    assert classify_text_length(" ".join(["word"] * 100)) == "long"
