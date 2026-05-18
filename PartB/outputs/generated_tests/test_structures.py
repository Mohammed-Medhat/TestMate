import pytest
from unittest.mock import MagicMock, patch
from requests.structures import *

def test_caseinsensitivedict_eq():
    assert CaseInsensitiveDict({'foo': 'bar'}) == {'FOO': 'bar'}
    assert CaseInsensitiveDict({'FOO': 'bar'}) == {'foo': 'bar'}

def test_caseinsensitivedict_setitem():
    obj = CaseInsensitiveDict()
    obj['KEY'] = 'val'
    assert obj['key'] == 'val'
    assert obj['KEY'] == 'val'
    assert obj['kEy'] == 'val'

def test_lookupdict_getitem():
    obj = LookupDict('foo')
    assert obj['bar'] == None

def test_caseinsensitivedict_getitem():
    obj = CaseInsensitiveDict({'key': 'value'})
    assert obj['key'] == 'value'

def test_caseinsensitivedict_delitem():
    obj = CaseInsensitiveDict({"key": "value"})
    del obj["key"]
    assert "key" not in obj

def test_caseinsensitivedict_iter():
    assert list(iter(CaseInsensitiveDict({'a': 1, 'b': 2}))) == ['a', 'b']
    assert list(iter(CaseInsensitiveDict({'x': 10, 'y': 20, 'z': 30}))) == ['x', 'y', 'z']

def test_caseinsensitivedict_len():
    assert len(CaseInsensitiveDict({'a': 1, 'b': 2})) == 2
    assert len(CaseInsensitiveDict({'a': 1, 'b': 2, 'c': 3})) == 3
    assert len(CaseInsensitiveDict({'a': 1, 'b': 2, 'c': 3, 'd': 4})) == 4

def test_caseinsensitivedict_copy():
    assert CaseInsensitiveDict({'key': 'value'}).copy()['key'] == 'value'
    assert CaseInsensitiveDict({'key': 'value'}).copy()['key'] != 'not_value'

def test_caseinsensitivedict_repr():
    assert repr(CaseInsensitiveDict({'a': 1})) == "{'a': 1}"

def test_lookupdict_repr():
    assert repr(LookupDict()) == "<lookup 'None'>"
    assert repr(LookupDict("test_name")) == "<lookup 'test_name'>"

def test_lookupdict_get():
    assert LookupDict().get('x', 1) == 1

def test_caseinsensitivedict_init():
    assert CaseInsensitiveDict() == {}
    assert CaseInsensitiveDict({}) == {}
    assert CaseInsensitiveDict({"a": 1}) == {"a": 1}
    assert CaseInsensitiveDict(a=1) == {"a": 1}
    assert CaseInsensitiveDict(a=1, b=2) == {"a": 1, "b": 2}

def test_lookupdict_init():
    assert LookupDict('test_name').name == 'test_name'
