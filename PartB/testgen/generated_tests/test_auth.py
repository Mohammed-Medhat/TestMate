import pytest
from unittest.mock import MagicMock, patch
from requests.auth import *
from requests.auth import _basic_auth_str

def test_basic_auth_str():
    assert _basic_auth_str('username', 'password') == 'Basic dXNlcm5hbWU6cGFzc3dvcmQ='

def test_httpdigestauth_init_per_thread_state():
    assert HTTPDigestAuth(None, None).init_per_thread_state() == None

def test_httpbasicauth_eq():
    assert HTTPBasicAuth('username', 'password') == HTTPBasicAuth('username', 'password')
    assert HTTPBasicAuth('username', 'password') != HTTPBasicAuth('username', 'passworde')
    assert HTTPBasicAuth('username', 'password') != HTTPBasicAuth('usernamex', 'password')
    assert HTTPBasicAuth('username', 'password') != HTTPBasicAuth('usernamex', 'passworde')

def test_httpdigestauth_eq():
    assert HTTPDigestAuth("user", "pass") == HTTPDigestAuth("user", "pass")

def test_httpbasicauth_init():
    assert HTTPBasicAuth('username', 'password') == HTTPBasicAuth('username', 'password')

def test_httpbasicauth_init_2():
    assert HTTPBasicAuth("username", "password") == HTTPBasicAuth("username", "password")
    assert HTTPBasicAuth(b"username", b"password") == HTTPBasicAuth(b"username", b"password")
    assert HTTPBasicAuth(b"username", "password") == HTTPBasicAuth(b"username", "password")

def test_httpdigestauth_init():
    assert HTTPDigestAuth('username', 'password') == HTTPDigestAuth('username', 'password')
    assert HTTPDigestAuth(b'username', b'password') == HTTPDigestAuth(b'username', b'password')
    assert HTTPDigestAuth('username', b'password') == HTTPDigestAuth('username', b'password')
    assert HTTPDigestAuth(b'username', 'password') == HTTPDigestAuth(b'username', 'password')

def test_httpdigestauth_init_2():
    assert HTTPDigestAuth('username', 'password').username == 'username'
    assert HTTPDigestAuth('username', 'password').password == 'password'

def test_httpdigestauth_init_3():
    assert HTTPDigestAuth('username', 'password') == HTTPDigestAuth('username', 'password')
