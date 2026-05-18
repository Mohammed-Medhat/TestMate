import pytest
from unittest.mock import MagicMock, patch
from requests.status_codes import *
from requests.status_codes import _init

def test_init():
    assert _init() == None
