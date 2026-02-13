import math

def ochiai_score(passed, failed, total_failed):
    if failed == 0:
        return 0.0
    return failed / math.sqrt(total_failed * (passed + failed))

def rank_suspicious_lines(spectrum):
    """
    spectrum = { line: (passed_count, failed_count) }
    """
    total_failed = sum(f for _, f in spectrum.values())

    scored = []
    for line, (p, f) in spectrum.items():
        score = ochiai_score(p, f, total_failed)
        scored.append((line, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored
