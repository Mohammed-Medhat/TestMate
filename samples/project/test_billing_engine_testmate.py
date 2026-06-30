import pytest
from unittest.mock import MagicMock, patch
from billing_engine import *

def test_calculate_order_total():
    assert calculate_order_total(20.0, 5) == 100.0
    assert calculate_order_total(20.0, 10) == 190.0
    assert calculate_order_total(20.0, 50) == 850.0
    assert calculate_order_total(20.0, 100) == 1600.0

def test_summarize_revenue():
    assert summarize_revenue([120.0, 340.0, 85.0, 210.0, 95.0]) == {'min': 85.0, 'max': 340.0, 'mean': 170.0, 'range': 255.0, 'median': 120.0}
    assert summarize_revenue([500.0]) == {'min': 500.0, 'max': 500.0, 'mean': 500.0, 'range': 0.0, 'median': 500.0}
    assert summarize_revenue([1000.0, 1000.0, 1000.0]) == {'min': 1000.0, 'max': 1000.0, 'mean': 1000.0, 'range': 0.0, 'median': 1000.0}
    assert summarize_revenue([1.0, 2.0, 3.0, 4.0, 5.0]) == {'min': 1.0, 'max': 5.0, 'mean': 3.0, 'range': 4.0, 'median': 3.0}
    assert summarize_revenue([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) == {'min': 1.0, 'max': 6.0, 'mean': 3.5, 'range': 5.0, 'median': 3.5}
    assert summarize_revenue([1.0]) == {'min': 1.0, 'max': 1.0, 'mean': 1.0, 'range': 0.0, 'median': 1.0}
    assert summarize_revenue([0.0]) == {'min': 0.0, 'max': 0.0, 'mean': 0.0, 'range': 0.0, 'median': 0.0}
    assert summarize_revenue([-1.0]) == {'min': -1.0, 'max': -1.0, 'mean': -1.0, 'range': 0.0, 'median': -1.0}
    assert summarize_revenue([1.0, -1.0, 0.0]) == {'min': -1.0, 'max': 1.0, 'mean': 0.0, 'range': 2.0, 'median': 0.0}


# --- Layer 1: docstring spec examples ---
def test_calculate_order_total_docstring_example_1():
    """Docstring example 1: calculate_order_total(20.0, 5) -> 100.0."""
    import pytest
    assert calculate_order_total(20.0, 5) == pytest.approx(100.0, rel=1e-6)


def test_calculate_order_total_docstring_example_2():
    """Docstring example 2: calculate_order_total(20.0, 10) -> 190.0."""
    import pytest
    assert calculate_order_total(20.0, 10) == pytest.approx(190.0, rel=1e-6)


def test_calculate_order_total_docstring_example_3():
    """Docstring example 3: calculate_order_total(20.0, 50) -> 850.0."""
    import pytest
    assert calculate_order_total(20.0, 50) == pytest.approx(850.0, rel=1e-6)


def test_calculate_order_total_docstring_example_4():
    """Docstring example 4: calculate_order_total(20.0, 100) -> 1600.0."""
    import pytest
    assert calculate_order_total(20.0, 100) == pytest.approx(1600.0, rel=1e-6)


# --- Layer 1: docstring spec examples ---
def test_summarize_revenue_docstring_example_1():
    """Docstring example 1: summarize_revenue([120.0, 340.0, 85.0, 210.0, 95.0]) -> {'min': 85.0, 'max': 340.0, 'mean': 170.0, 'range': 255.0, 'median': 120.0}."""
    assert summarize_revenue([120.0, 340.0, 85.0, 210.0, 95.0]) == {'min': 85.0, 'max': 340.0, 'mean': 170.0, 'range': 255.0, 'median': 120.0}


def test_summarize_revenue_docstring_example_2():
    """Docstring example 2: summarize_revenue([500.0]) -> {'min': 500.0, 'max': 500.0, 'mean': 500.0, 'range': 0.0, 'median': 500.0}."""
    assert summarize_revenue([500.0]) == {'min': 500.0, 'max': 500.0, 'mean': 500.0, 'range': 0.0, 'median': 500.0}

