import pytest
from unittest.mock import MagicMock, patch
from requests.hooks_buggy import *

def test_dispatch_hook():
    assert dispatch_hook("response",{"response":[lambda x:x*2]},2) == 4
    assert dispatch_hook("response",{"response":[]},2) == 2
    assert dispatch_hook(None,{},3) == 3
