"""
verify_calculator.py — Bug-exposing pytest suite for buggy_calculator.

Named with a 'verify_' prefix so PartB's autogen won't overwrite it
(PartB writes generated tests as test_<source>.py).

Use this file as the "Test File" in the Bug Fixer (PartC) tab.
"""
import pytest
from buggy_calculator import (
    calculate_discount,
    clamp,
    count_vowels,
    fahrenheit_to_celsius,
    find_second_largest,
)


# ── Bug 1: calculate_discount adds instead of subtracts ──────────────
def test_discount_twenty_percent():
    assert calculate_discount(100, 20) == 80.0

def test_discount_ten_percent():
    assert calculate_discount(50, 10) == 45.0

def test_discount_full():
    assert calculate_discount(100, 100) == 0.0


# ── Bug 2: clamp returns wrong bound ─────────────────────────────────
def test_clamp_below_min():
    assert clamp(5, 10, 20) == 10

def test_clamp_above_max():
    assert clamp(25, 10, 20) == 20

def test_clamp_in_range():
    assert clamp(15, 10, 20) == 15


# ── Bug 3: count_vowels misses uppercase ─────────────────────────────
def test_count_vowels_mixed_case():
    assert count_vowels("Hello World") == 3

def test_count_vowels_all_upper():
    assert count_vowels("AEIOU") == 5

def test_count_vowels_no_vowels():
    assert count_vowels("rhythm") == 0


# ── Bug 4: fahrenheit_to_celsius wrong sign ──────────────────────────
def test_freezing_point():
    assert fahrenheit_to_celsius(32) == pytest.approx(0.0, abs=1e-6)

def test_boiling_point():
    assert fahrenheit_to_celsius(212) == pytest.approx(100.0, abs=1e-6)


# ── Bug 5: find_second_largest returns largest ───────────────────────
def test_second_largest_basic():
    assert find_second_largest([3, 1, 4, 1, 5, 9]) == 5

def test_second_largest_two_elements():
    assert find_second_largest([10, 20]) == 10
