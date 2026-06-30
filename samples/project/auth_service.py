"""
auth_service.py
===============
User authentication and security utilities.
Handles password validation and account access rules.
"""
from typing import Tuple, List


def validate_password(password: str) -> Tuple[bool, List[str]]:
    """
    Validate a password against the platform's security policy.

    All five rules must pass for the password to be accepted:
        1. Minimum length of 8 characters.
        2. At least one uppercase letter (A-Z).
        3. At least one lowercase letter (a-z).
        4. At least one digit (0-9).
        5. At least one special character from: ! @ # $ % ^ & *

    Args:
        password: The candidate password string.

    Returns:
        Tuple of (is_valid: bool, violations: List[str]).
        violations is empty when the password is valid.

    >>> validate_password("Secure1!")
    (True, [])
    >>> validate_password("short")
    (False, ['At least 8 characters required', 'Missing uppercase letter', 'Missing digit', 'Missing special character (!@#$%^&*)'])
    >>> validate_password("NoSpecial123")
    (False, ['Missing special character (!@#$%^&*)'])
    """
    violations: List[str] = []

    if len(password) < 8:
        violations.append("At least 8 characters required")
    if not any(c.isupper() for c in password):
        violations.append("Missing uppercase letter")
    if not any(c.islower() for c in password):
        violations.append("Missing lowercase letter")
    if not any(c.isdigit() for c in password):
        violations.append("Missing digit")
    if not any(c in "!@#$%^&*" for c in password):
        violations.append("Missing special character (!@#$%^&*)")

    return (len(violations) == 0, violations)


def classify_account_risk(failed_attempts: int, days_since_activity: int) -> str:
    """
    Classify an account's security risk level based on login behaviour.

    Risk levels:
        "critical" : failed_attempts >= 10  OR  days_since_activity >= 365
        "high"     : failed_attempts >= 5   OR  days_since_activity >= 180
        "medium"   : failed_attempts >= 3   OR  days_since_activity >= 90
        "low"      : all other cases

    Args:
        failed_attempts:      Number of consecutive failed login attempts.
        days_since_activity:  Days elapsed since the last successful login.

    Returns:
        One of: "critical", "high", "medium", "low".

    >>> classify_account_risk(0, 0)
    'low'
    >>> classify_account_risk(3, 0)
    'medium'
    >>> classify_account_risk(0, 200)
    'high'
    >>> classify_account_risk(10, 30)
    'critical'
    >>> classify_account_risk(2, 400)
    'critical'
    """
    if failed_attempts >= 10 or days_since_activity >= 365:
        return "critical"
    if failed_attempts >= 5 or days_since_activity >= 180:
        return "high"
    if failed_attempts >= 3 or days_since_activity >= 90:
        return "medium"
    return "low"
