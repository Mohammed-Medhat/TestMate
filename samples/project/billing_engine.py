"""
billing_engine.py
=================
Core pricing and billing logic for the e-commerce platform.
Handles volume discounts, order totals, and revenue statistics.
"""
from typing import List, Dict


def calculate_order_total(unit_price: float, quantity: int) -> float:
    """
    Compute the final order total after applying tiered volume discounts.

    Discount schedule based on order quantity:
        quantity >= 100 : 20% discount
        quantity >= 50  : 15% discount
        quantity >= 20  : 10% discount
        quantity >= 10  :  5% discount
        quantity <  10  :  no discount

    Args:
        unit_price: Price per unit in USD (must be >= 0).
        quantity:   Number of units ordered (must be >= 1).

    Returns:
        Total order cost after discount, rounded to 2 decimal places.

    Raises:
        ValueError: If unit_price < 0 or quantity < 1.

    >>> calculate_order_total(20.0, 5)
    100.0
    >>> calculate_order_total(20.0, 10)
    190.0
    >>> calculate_order_total(20.0, 50)
    850.0
    >>> calculate_order_total(20.0, 100)
    1600.0
    """
    if unit_price < 0:
        raise ValueError(f"unit_price must be non-negative, got {unit_price}")
    if quantity < 1:
        raise ValueError(f"quantity must be at least 1, got {quantity}")

    subtotal = unit_price * quantity

    if quantity >= 100:
        discount_rate = 0.20
    elif quantity >= 50:
        discount_rate = 0.15
    elif quantity >= 20:
        discount_rate = 0.10
    elif quantity >= 10:
        discount_rate = 0.05
    else:
        discount_rate = 0.0

    return round(subtotal * (1 - discount_rate), 2)


def summarize_revenue(daily_totals: List[float]) -> Dict[str, float]:
    """
    Compute descriptive revenue statistics from a list of daily sales totals.

    Args:
        daily_totals: Non-empty list of daily revenue figures (USD).

    Returns:
        Dictionary with keys:
            "min"    : lowest daily revenue
            "max"    : highest daily revenue
            "mean"   : average daily revenue (rounded to 4 dp)
            "range"  : difference between max and min
            "median" : median daily revenue

    Raises:
        ValueError: If daily_totals is empty.

    >>> summarize_revenue([120.0, 340.0, 85.0, 210.0, 95.0])
    {'min': 85.0, 'max': 340.0, 'mean': 170.0, 'range': 255.0, 'median': 120.0}
    >>> summarize_revenue([500.0])
    {'min': 500.0, 'max': 500.0, 'mean': 500.0, 'range': 0.0, 'median': 500.0}
    """
    if not daily_totals:
        raise ValueError("daily_totals must not be empty")

    sorted_vals = sorted(daily_totals)
    n = len(sorted_vals)
    total = sum(sorted_vals)

    median = (
        (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2
        if n % 2 == 0
        else sorted_vals[n // 2]
    )

    return {
        "min":    sorted_vals[0],
        "max":    sorted_vals[-1],
        "mean":   round(total / n, 4),
        "range":  sorted_vals[-1] - sorted_vals[0],
        "median": median,
    }
