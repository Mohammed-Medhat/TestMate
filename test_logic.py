from buggy_code import add, divide, max_in_list

def test_add():
    assert add(2, 3) == 5
    assert add(-1, 1) == 0

def test_divide():
    assert divide(10, 2) == 5
    assert divide(10, 0) is None

def test_max_in_list():
    assert max_in_list([5, 6, 4]) == 6  # This will FAIL due to bug
    assert max_in_list([10]) == 10

from buggy_code import find_first_in_sorted

def test_find_first_in_sorted():
    assert find_first_in_sorted([1, 2, 3, 4, 5], 3) == 2
    
    assert find_first_in_sorted([1, 2, 3, 4, 5], 6) == -1