"""fixed_calculator.py — Correct implementations (used to verify tests are valid)."""


def calculate_discount(price: float, discount_pct: float) -> float:
    """Return the final price after applying a discount percentage (0-100)."""
    return price - (price * discount_pct / 100)


def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp value so that min_val <= result <= max_val."""
    if value < min_val:
        return min_val
    if value > max_val:
        return max_val
    return value


def count_vowels(text: str) -> int:
    """Count vowels (a,e,i,o,u) in text, case-insensitive."""
    vowels = "aeiou"
    return sum(1 for c in text.lower() if c in vowels)


def fahrenheit_to_celsius(f: float) -> float:
    """Convert Fahrenheit to Celsius: (F - 32) * 5 / 9."""
    return (f - 32) * 5 / 9


def find_second_largest(numbers: list) -> float:
    """Return the second largest unique value in a list."""
    unique = sorted(set(numbers), reverse=True)
    if len(unique) < 2:
        raise ValueError("Need at least 2 unique values")
    return unique[1]
