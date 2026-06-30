import pytest
from unittest.mock import MagicMock, patch
from content_analyzer import *


# FAILING TEST (exhausted retries): Potential Bug
def test_readability_grade():
    assert readability_grade("Hi!") == 'elementary'
    assert readability_grade("The cat sat on a mat.") == 'basic'
    assert readability_grade("Software engineering requires systematic thinking.") == 'intermediate'
    assert readability_grade("Electroencephalography demonstrates neurological complexity.") == 'advanced'
