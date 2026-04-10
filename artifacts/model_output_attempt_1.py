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

def find_first_in_sorted(arr, x):
    (lo, hi) = (0, len(arr) - 1)
    while lo <= hi:
        mid = (lo + hi) // 2
        if x == arr[mid]:
            return mid
        elif x < arr[mid]:
            hi = mid - 1
        else:
            lo = mid + 1
    return -1