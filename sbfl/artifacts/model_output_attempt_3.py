  1: def add(a, b):
  2:     return a + b
  3: 
  4: def divide(a, b):
  5:     if b == 0:
  6:         return None
  7:     return a / b
  8: 
  9: def max_in_list(numbers):
 10:     if not numbers:
 11:         return None
 12:     max_val = numbers[0]
 13:     for num in numbers:
 14:         if num > max_val:
 15:             max_val = num
 16:     return max_val
 17: 
 18: def find_first_in_sorted(arr, x):
 19:     (lo, hi) = (0, len(arr) - 1)
 20:     while lo < hi:
 21:         mid = (lo + hi) // 2
 22:         if x == arr[mid]:
 23:             return mid
 24:         elif x < arr[mid]:
 25:             hi = mid - 1
 26:         else:
 27:             lo = mid + 1
 28:     return -1