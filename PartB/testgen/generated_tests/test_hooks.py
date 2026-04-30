import pytest
from unittest.mock import MagicMock, patch
from requests.hooks import *

def test_dispatch_hook():
    assert dispatch_hook('a', None, 'b') == 'b'
    assert dispatch_hook('a', {}, 'b') == 'b'

def test_default_hooks():
    assert default_hooks() == {'response': []}
