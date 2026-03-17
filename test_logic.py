"""
test_logic.py — Comprehensive test suite for buggy_code.py
Covers edge cases so the APR pipeline can detect a wide range of bugs.
"""
from buggy_code import add, divide, max_in_list, find_first_in_sorted


# ── add ───────────────────────────────────────────────────────────────

def test_add_positive():
    assert add(2, 3) == 5

def test_add_negative():
    assert add(-1, 1) == 0

def test_add_zeros():
    assert add(0, 0) == 0

def test_add_both_negative():
    assert add(-3, -4) == -7

def test_add_large():
    assert add(1000, 2000) == 3000


# ── divide ────────────────────────────────────────────────────────────

def test_divide_normal():
    assert divide(10, 2) == 5.0

def test_divide_by_zero():
    assert divide(10, 0) is None

def test_divide_negative():
    assert divide(-9, 3) == -3.0

def test_divide_fraction():
    assert divide(7, 2) == 3.5

def test_divide_zero_numerator():
    assert divide(0, 5) == 0.0


# ── max_in_list ───────────────────────────────────────────────────────

def test_max_normal():
    assert max_in_list([5, 6, 4]) == 6

def test_max_single():
    assert max_in_list([10]) == 10

def test_max_empty():
    assert max_in_list([]) is None

def test_max_all_same():
    assert max_in_list([3, 3, 3]) == 3

def test_max_negatives():
    assert max_in_list([-1, -5, -2]) == -1

def test_max_first_is_largest():
    # Catches bugs where initialisation skips index 0
    assert max_in_list([99, 1, 2, 3]) == 99

def test_max_last_is_largest():
    assert max_in_list([1, 2, 3, 99]) == 99

def test_max_with_zero():
    assert max_in_list([0, -1, -2]) == 0

def test_max_two_elements():
    assert max_in_list([7, 3]) == 7
    assert max_in_list([3, 7]) == 7


# ── find_first_in_sorted ─────────────────────────────────────────────

def test_find_middle():
    assert find_first_in_sorted([1, 2, 3, 4, 5], 3) == 2

def test_find_not_found():
    assert find_first_in_sorted([1, 2, 3, 4, 5], 6) == -1

def test_find_first_element():
    # Catches off-by-one bugs at lo boundary
    assert find_first_in_sorted([1, 2, 3, 4, 5], 1) == 0

def test_find_last_element():
    # Catches off-by-one bugs at hi boundary
    assert find_first_in_sorted([1, 2, 3, 4, 5], 5) == 4

def test_find_single_element_found():
    assert find_first_in_sorted([7], 7) == 0

def test_find_single_element_not_found():
    assert find_first_in_sorted([7], 3) == -1

def test_find_empty():
    assert find_first_in_sorted([], 1) == -1

def test_find_two_elements_first():
    assert find_first_in_sorted([4, 8], 4) == 0

def test_find_two_elements_second():
    assert find_first_in_sorted([4, 8], 8) == 1

def test_find_larger_list():
    arr = list(range(0, 100, 2))   # [0, 2, 4, ..., 98]
    assert find_first_in_sorted(arr, 50) == 25
    assert find_first_in_sorted(arr, 99) == -1