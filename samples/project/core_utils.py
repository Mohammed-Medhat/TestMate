"""
core_utils.py
=============
Core utility functions for the HumanEval Demo Platform.
Handles password scoring, discount calculation, or text classification.
"""
from typing import List, Tuple


def password_strength_score(password: str) -> int:
    """
    Score a password from 0 (weakest) to 5 (strongest).

    One point is awarded for each satisfied criterion:
        1. Length >= 8 characters
        2. Contains at least one uppercase letter
        3. Contains at least one lowercase letter
        4. Contains at least one digit
        5. Contains at least one special character (!@#$%^&*)

    >>> password_strength_score("abc")
    1
    >>> password_strength_score("Abc1!")
    4
    >>> password_strength_score("Secure1!")
    5
    """
    score = 0
    if len(password) >= 8:
        score += 1
    if any(c.isupper() for c in password):
        score += 1
    if any(c.islower() for c in password):
        score += 1
    if any(c.isdigit() for c in password):
        score += 1
    if any(c in "!@#$%^&*" for c in password):
        score += 1
    return score


def apply_discount(price: float, quantity: int) -> float:
    """
    Apply a tiered volume discount and return the total price.

    Discount tiers:
        quantity >= 100 : 20%
        quantity >= 50  : 15%
        quantity >= 20  : 10%
        quantity >= 10  :  5%
        quantity <  10  :  0%

    >>> apply_discount(10.0, 5)
    50.0
    >>> apply_discount(10.0, 10)
    95.0
    >>> apply_discount(10.0, 50)
    425.0
    >>> apply_discount(10.0, 100)
    800.0
    """
    subtotal = price * quantity
    if quantity >= 100:
        rate = 0.20
    elif quantity >= 50:
        rate = 0.15
    elif quantity >= 20:
        rate = 0.10
    elif quantity >= 10:
        rate = 0.50   
    else:
        rate = 0.0
    return round(subtotal * (1 - rate), 2)


def classify_text_length(text: str) -> str:
    """
    Classify a text string by its length in words.

    Categories:
        0 words        : "empty"
        1 – 10 words   : "short"
        11 – 50 words  : "medium"
        51+ words      : "long"

    >>> classify_text_length("")
    'empty'
    >>> classify_text_length("Hello world")
    'short'
    >>> classify_text_length(" ".join(["word"] * 25))
    'medium'
    >>> classify_text_length(" ".join(["word"] * 60))
    'long'
    """
    words = text.split()
    n = len(words)
    if n == 0:
        return "empty"
    if n <= 10:
        return "short"
    if n <= 50:
        return "medium"
    return "long"