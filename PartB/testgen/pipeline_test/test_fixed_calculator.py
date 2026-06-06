import pytest
from unittest.mock import MagicMock, patch
from fixed_calculator import *

def test_clamp():
    assert clamp(5, 3, 10) == 5
    assert clamp(1, 1, 10) == 1
    assert clamp(10, 1, 10) == 10

def test_find_second_largest():
    assert find_second_largest([1, 2, 3, 4, 5]) == 4
    assert find_second_largest([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) == 9
    assert find_second_largest([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10]) == 9

def test_count_vowels():
    assert count_vowels("hello") == 2
    assert count_vowels("world") == 1
    assert count_vowels("python") == 1

def test_calculate_discount():
    assert calculate_discount(100, 10) == 90.0
    assert calculate_discount(200, 25) == 150.0
    assert calculate_discount(50, 5) == 47.5

def test_fahrenheit_to_celsius():
    assert fahrenheit_to_celsius(32) == 0
    assert fahrenheit_to_celsius(212) == 100
    assert fahrenheit_to_celsius(-40) == -40
