import pytest
from unittest.mock import MagicMock, patch
from auth_service import *


# FAILING TEST (exhausted retries): Potential Bug
def test_validate_password():
    assert validate_password('ab') == (False, ['At least 8 characters required', 'Missing uppercase letter', 'Missing digit', 'Missing special character (!@#$%^&*)'])
    assert validate_password('ABC1') == (False, ['At least 8 characters required', 'Missing lowercase letter', 'Missing special character (!@#$%^&*)'])
    assert validate_password('abc1') == (False, ['At least 8 characters required', 'Missing uppercase letter', 'Missing special character (!@#$%^&*)'])
    assert validate_password('1!@#') == (False, ['At least 8 characters required', 'Missing uppercase letter', 'Missing lowercase letter'])
    assert validate_password('abcde!@#') == (False, ['At least 8 characters required', 'Missing uppercase letter'])
    assert validate_password('ABC1!@#') == (False, ['At least 8 characters required', 'Missing lowercase letter'])
    assert validate_password('123abc!@#') == (False, ['At least 8 characters required', 'Missing uppercase letter'])
    assert validate_password('AB1!@#') == (False, ['At least 8 characters required', 'Missing lowercase letter'])
