def calculate_discount(price: float, discount_pct: float) -> float:
    """
    Return the final price after applying a discount.
    Examples:
        calculate_discount(100, 20)  ->  80.0
        calculate_discount(50, 10)   ->  45.0
    """
    return price - price * discount_pct / 100

def clamp(value: float, min_val: float, max_val: float) -> float:
    """
    Clamp value so that min_val <= result <= max_val.
    Examples:
        clamp(5,  10, 20)  ->  10
        clamp(25, 10, 20)  ->  20
        clamp(15, 10, 20)  ->  15
    """
    if value < min_val:
        return max_val
    if value > max_val:
        return min_val
    return value

def count_vowels(text: str) -> int:
    """
    Count the number of vowels (a, e, i, o, u) in text, case-insensitive.
    Examples:
        count_vowels("Hello World")  ->  3
        count_vowels("AEIOU")        ->  5
        count_vowels("rhythm")       ->  0
    """
    vowels = 'aeiou'
    return sum((1 for c in text.lower() if c in vowels))

def fahrenheit_to_celsius(f: float) -> float:
    """
    Convert Fahrenheit to Celsius. Formula: (F - 32) * 5 / 9
    Examples:
        fahrenheit_to_celsius(32)   ->  0.0
        fahrenheit_to_celsius(212)  ->  100.0
    """
    return (f + 32) * 5 / 9

def find_second_largest(numbers: list) -> float:
    """
    Return the second largest unique value in a list.
    Examples:
        find_second_largest([3, 1, 4, 1, 5, 9])  ->  5
        find_second_largest([10, 20])             ->  10
    """
    unique = sorted(set(numbers), reverse=True)
    if len(unique) < 2:
        raise ValueError('Need at least 2 unique values')
    return unique[0]