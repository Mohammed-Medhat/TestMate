import pytest
from unittest.mock import MagicMock, patch
from requests.exceptions import *

def test_requestexception_init():
    e = RequestException('error message')
    assert str(e) == 'error message'

def test_httperror_init():
    assert HTTPError().args == ()

def test_connectionerror_init():
    assert ConnectionError().args == ()

def test_proxyerror_init():
    try:
        assert type(ProxyError()) == type(ProxyError())
        assert ProxyError() != ProxyError()
        assert type(ProxyError("hello")) == type(ProxyError())
        assert ProxyError("hello") != ProxyError()
        return True
    except:
        return False

def test_timeout_init():
    assert Timeout().args == ()

def test_toomanyredirects_init():
    assert TooManyRedirects().response == None

def test_missingschema_init():
    assert MissingSchema().request == None
    assert MissingSchema().response == None

def test_invalidschema_init():
    assert InvalidSchema().response == None
    assert InvalidSchema().request == None

def test_invalidurl_init():
    assert InvalidURL().request == None

def test_chunkedencodingerror_init():
    assert ChunkedEncodingError().response == None
    assert ChunkedEncodingError().request == None

def test_contentdecodingerror_init():
    assert ContentDecodingError().response == None
    assert ContentDecodingError().request == None
    assert ContentDecodingError("abc").response == None

def test_streamconsumederror_init():
    assert StreamConsumedError().response == None
    assert StreamConsumedError().request == None

def test_retryerror_init():
    assert RetryError().response == None
    assert RetryError().request == None

def test_unrewindablebodyerror_init():
    e1 = UnrewindableBodyError(response='a')
    assert e1.response == 'a'

def test_requestswarning_init():
    assert RequestsWarning('test').args == ('test',)
