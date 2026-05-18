import pytest
from unittest.mock import MagicMock, patch
from requests.exceptions import *

def test_invalidjsonerror_init():
    assert InvalidJSONError("foo").args == ("foo",)

def test_httperror_init():
    assert HTTPError('message').args[0] == 'message'

def test_connectionerror_init():
    assert ConnectionError("message").args[0] == "message"

def test_urlrequired_init():
    assert URLRequired("message").args[0] == "message"

def test_invalidurl_init():
    assert InvalidURL('foo').args == ('foo',)

def test_chunkedencodingerror_init():
    assert ChunkedEncodingError("message").args == ("message",)

def test_contentdecodingerror_init():
    assert ContentDecodingError('message').args[0] == 'message'

def test_streamconsumederror_init():
    assert isinstance(StreamConsumedError(), StreamConsumedError)
    assert isinstance(StreamConsumedError(), RequestException)
    assert isinstance(StreamConsumedError(), TypeError)
    assert StreamConsumedError().__module__ == 'requests.exceptions'

def test_retryerror_init():
    assert RetryError("RetryError").args[0] == "RetryError"
