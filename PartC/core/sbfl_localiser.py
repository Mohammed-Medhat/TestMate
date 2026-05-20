"""
sbfl_localiser.py — Ochiai-based spectrum fault localisation.
Ranks suspicious lines by their correlation with test failures.
"""
import math


def ochiai_score(passed: int, failed: int, total_failed: int) -> float:
    if failed == 0 or total_failed == 0:
        return 0.0
    return failed / math.sqrt(total_failed * (passed + failed))


def rank_suspicious_lines(spectrum: dict) -> list:
    """
    spectrum = { line_number: [passed_count, failed_count] }
    Returns a sorted list of (line, score) tuples, highest score first.
    """
    total_failed = sum(f for _, f in spectrum.values())

    scored = [
        (line, ochiai_score(p, f, total_failed))
        for line, (p, f) in spectrum.items()
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored
