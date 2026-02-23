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
