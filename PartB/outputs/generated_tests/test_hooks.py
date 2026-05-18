import pytest
from unittest.mock import MagicMock, patch
from requests.hooks import *

def test_default_hooks():
    assert default_hooks() == {'response': [], 'response': [], 'response': []}
