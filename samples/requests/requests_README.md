# Requests: HTTP for Humans™

**Requests** is an elegant and simple HTTP library for Python, built for human beings.

## Overview

The Requests library makes HTTP requests intuitive: no manual URL string construction, no manual encoding of POST data, no manual encoding of headers, no handling of low-level connection details. It removes the friction of HTTP work in Python.

It is designed to be a high-level abstraction over the standard `urllib3` library, exposing a small, consistent surface (`get`, `post`, `put`, `patch`, `delete`, `head`, `options`) that returns rich `Response` objects.

## Installation

```bash
pip install requests
```

## Quick example

```python
import requests

# Simple GET
r = requests.get('https://api.github.com/user', auth=('user', 'pass'))
r.status_code            # 200
r.headers['content-type']  # 'application/json; charset=utf8'
r.encoding               # 'utf-8'
r.text                   # '{"type":"User"...'
r.json()                 # parsed Python dict

# POST with form data
r = requests.post('https://httpbin.org/post', data={'key': 'value'})

# POST with JSON
r = requests.post('https://httpbin.org/post', json={'k': 'v'})

# Sessions for connection re-use + persistent cookies
s = requests.Session()
s.get('https://httpbin.org/cookies/set/sessioncookie/123456789')
r = s.get('https://httpbin.org/cookies')
```

## Features

- Keep-Alive & connection pooling
- International domains and URLs
- Sessions with cookie persistence
- Browser-style TLS/SSL verification
- Basic & Digest authentication (`HTTPBasicAuth`, `HTTPDigestAuth`)
- Dict-like cookie access
- Automatic content decoding
- Automatic decompression of gzip/deflate
- Unicode response bodies
- HTTP(S) proxy support
- Multipart file uploads
- Streaming downloads (`stream=True`, `iter_content`)
- Connection / read timeouts
- Chunked HTTP requests
- `.netrc` support
- Custom redirect handling (`allow_redirects`)
- Mountable transport adapters (`HTTPAdapter`)

## What I want tests to cover

**Main problems we care about:**
1. Network errors should raise the correct, specific exceptions (`ConnectionError`, `Timeout`, `TooManyRedirects`, `SSLError`, `HTTPError`)
2. URL handling should accept Unicode, IDN domains, and URLs with query parameters dict
3. Header dicts should be case-insensitive when accessed
4. Response decoding (`r.text`, `r.json()`, `r.content`) should work on UTF-8, latin-1, and binary payloads
5. Session-level state (cookies, headers, auth) should persist across requests in that session

**Expected behaviour:**
- Successful 2xx responses return a `Response` object with `.ok == True` and `raise_for_status()` does nothing
- 4xx/5xx responses have `.ok == False` and `raise_for_status()` raises `HTTPError`
- Timeouts respect both connect and read components when a tuple is passed
- `auth` parameter accepts both a (user, pass) tuple AND a callable auth handler

**Known edge cases:**
- Empty response bodies (`b''`) — `.text` should be `''`, `.json()` should raise `JSONDecodeError`
- Redirect loops should raise `TooManyRedirects` (not hang)
- Servers with `Content-Encoding: gzip` but actually-plain bodies — should not crash
- Mixed-case header names should compare equal: `r.headers['content-type']` == `r.headers['CONTENT-TYPE']`
- Sending bytes vs str in `data` should both work
- `params={}` with empty dict should not modify the URL

## Tech stack

- Python 3.7+
- `urllib3` (transport)
- `charset-normalizer` (encoding detection)
- `idna` (international domain encoding)
- `certifi` (CA bundle)

## License
Apache License 2.0
