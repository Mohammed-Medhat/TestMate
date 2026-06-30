import pytest
from unittest.mock import MagicMock, patch
from core_utils import *

def test_password_strength_score():
    assert password_strength_score("abc") == 1
    assert password_strength_score("Abc1!") == 4
    assert password_strength_score("Secure1!") == 5

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


def test_classify_text_length():
    assert classify_text_length("") == "empty"
    assert classify_text_length("Hi") == "short"
    assert classify_text_length("Hi there! How are you?") == "short"
    assert classify_text_length("This is a longer sentence. It has more than ten words!") == "medium"


# --- Layer 1: docstring spec examples ---
def test_apply_discount_docstring_example_1():
    """Docstring example 1: apply_discount(10.0, 5) -> 50.0."""
    import pytest
    assert apply_discount(10.0, 5) == pytest.approx(50.0, rel=1e-6)


def test_apply_discount_docstring_example_2():
    """Docstring example 2: apply_discount(10.0, 10) -> 95.0."""
    import pytest
    assert apply_discount(10.0, 10) == pytest.approx(95.0, rel=1e-6)


def test_apply_discount_docstring_example_3():
    """Docstring example 3: apply_discount(10.0, 50) -> 425.0."""
    import pytest
    assert apply_discount(10.0, 50) == pytest.approx(425.0, rel=1e-6)


def test_apply_discount_docstring_example_4():
    """Docstring example 4: apply_discount(10.0, 100) -> 800.0."""
    import pytest
    assert apply_discount(10.0, 100) == pytest.approx(800.0, rel=1e-6)


# --- Layer 1: docstring spec examples ---
def test_password_strength_score_docstring_example_1():
    """Docstring example 1: password_strength_score("abc") -> 1."""
    assert password_strength_score("abc") == 1


def test_password_strength_score_docstring_example_2():
    """Docstring example 2: password_strength_score("Abc1!") -> 4."""
    assert password_strength_score("Abc1!") == 4


def test_password_strength_score_docstring_example_3():
    """Docstring example 3: password_strength_score("Secure1!") -> 5."""
    assert password_strength_score("Secure1!") == 5


# --- Layer 1: docstring spec examples ---
def test_classify_text_length_docstring_example_1():
    """Docstring example 1: classify_text_length("") -> 'empty'."""
    assert classify_text_length("") == 'empty'


def test_classify_text_length_docstring_example_2():
    """Docstring example 2: classify_text_length("Hello world") -> 'short'."""
    assert classify_text_length("Hello world") == 'short'


def test_classify_text_length_docstring_example_3():
    """Docstring example 3: classify_text_length(" ".join(["word"] * 25)) -> 'medium'."""
    assert classify_text_length(" ".join(["word"] * 25)) == 'medium'


def test_classify_text_length_docstring_example_4():
    """Docstring example 4: classify_text_length(" ".join(["word"] * 60)) -> 'long'."""
    assert classify_text_length(" ".join(["word"] * 60)) == 'long'

