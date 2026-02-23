ISSUE: Fix the logic bug causing the test failure.

TRACE:
E       assert 20 == 5
E        +  where 20 = divide(10, 2)
sbfl/test_logic.py:8: AssertionError
FAILED sbfl/test_logic.py::test_divide - assert 20 == 5

BUGGY:
def add(a, b):
    return a + b         

def divide(a, b):
    if b == 0:
        return None
    return a / b

def max_in_list(numbers):
    if not numbers:
        return None
    max_val = numbers[0]
    for num in numbers:
        if num > max_val:  
            max_val = num
    return max_val