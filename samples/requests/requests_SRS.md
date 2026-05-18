# Software Requirements Specification

## Project: Requests — HTTP Library for Python

**Version:** 2.31
**Document type:** Software Requirements Specification (SRS)
**Target system:** `psf/requests` — a Python HTTP client library

---

## 1. Introduction

### 1.1 Purpose
This document specifies the functional and non-functional requirements for the Requests Python library. Requests provides a high-level, ergonomic HTTP client API that abstracts the complexity of `urllib3` and the standard library while preserving full HTTP semantics.

### 1.2 Scope
The Requests library shall provide a Python API for performing HTTP and HTTPS requests, handling responses, managing sessions, encoding and decoding payloads, and performing authentication. It shall not include a server-side component.

### 1.3 Definitions
- **HTTP request:** A message sent from a client to a server using the HTTP protocol.
- **HTTP response:** A message returned from a server in response to an HTTP request.
- **Session:** A persistent client object that maintains state across multiple requests.
- **Adapter:** A pluggable transport layer responsible for issuing requests for a given URL scheme.

---

## 2. Overall Description

### 2.1 Product perspective
Requests shall be a self-contained Python package distributed via PyPI. It shall depend on `urllib3` for low-level transport, `charset-normalizer` for encoding detection, `idna` for international domain encoding, and `certifi` for CA bundle data.

### 2.2 Product functions
The system shall expose seven module-level convenience functions: `get`, `post`, `put`, `patch`, `delete`, `head`, and `options`. The system shall also expose a `Session` class for managing persistent state.

### 2.3 User characteristics
The intended users are Python developers who require a simple, reliable HTTP client for scripts, applications, and tooling.

### 2.4 Constraints
The library shall support Python 3.7 and later. The library shall not modify the global state of the Python interpreter. The library shall be thread-safe at the Session level.

---

## 3. Functional Requirements

### 3.1 HTTP method support
**REQ-3.1.1** The system shall support the GET HTTP method.
**REQ-3.1.2** The system shall support the POST HTTP method.
**REQ-3.1.3** The system shall support the PUT HTTP method.
**REQ-3.1.4** The system shall support the PATCH HTTP method.
**REQ-3.1.5** The system shall support the DELETE HTTP method.
**REQ-3.1.6** The system shall support the HEAD HTTP method.
**REQ-3.1.7** The system shall support the OPTIONS HTTP method.

### 3.2 URL handling
**REQ-3.2.1** The system shall accept fully-qualified URLs as the first positional argument to all request functions.
**REQ-3.2.2** The system shall accept Unicode characters in URLs and encode them using IDN (Internationalized Domain Names) where appropriate.
**REQ-3.2.3** The system shall accept a `params` keyword argument as a dict, list of tuples, or string to be appended as a URL query string.
**REQ-3.2.4** The system shall not modify the URL when an empty `params` dict is supplied.
**REQ-3.2.5** The system shall reject malformed URLs by raising a `MissingSchema` or `InvalidURL` exception.

### 3.3 Headers
**REQ-3.3.1** The system shall accept a `headers` keyword argument as a dict of HTTP headers to send with the request.
**REQ-3.3.2** The system shall return response headers via the `Response.headers` attribute as a case-insensitive dict.
**REQ-3.3.3** Header dict lookups shall succeed regardless of the case of the requested key.
**REQ-3.3.4** The system shall set a default `User-Agent` header if the caller does not provide one.

### 3.4 Request body
**REQ-3.4.1** The system shall accept a `data` keyword argument for form-encoded request bodies (dict, list of tuples, bytes, or file-like).
**REQ-3.4.2** The system shall accept a `json` keyword argument and automatically serialise the value as JSON, setting the `Content-Type` header to `application/json`.
**REQ-3.4.3** The system shall accept a `files` keyword argument for multipart/form-data uploads.
**REQ-3.4.4** The system shall send bytes payloads as-is without re-encoding.

### 3.5 Response handling
**REQ-3.5.1** The system shall return a `Response` object from every successful request call.
**REQ-3.5.2** The `Response.status_code` attribute shall contain the integer HTTP status code returned by the server.
**REQ-3.5.3** The `Response.text` attribute shall contain the response body decoded as a string using the encoding declared in the response headers.
**REQ-3.5.4** The `Response.content` attribute shall contain the raw response body as bytes.
**REQ-3.5.5** The `Response.json()` method shall parse the response body as JSON and return the resulting Python object.
**REQ-3.5.6** The `Response.json()` method shall raise `requests.exceptions.JSONDecodeError` when the body is not valid JSON.
**REQ-3.5.7** The `Response.ok` attribute shall be `True` if the status code is less than 400, and `False` otherwise.
**REQ-3.5.8** The `Response.raise_for_status()` method shall raise `HTTPError` for 4xx and 5xx responses and do nothing for 2xx responses.

### 3.6 Authentication
**REQ-3.6.1** The system shall accept an `auth` keyword argument as a tuple of `(username, password)` for HTTP Basic Authentication.
**REQ-3.6.2** The system shall accept an `auth` keyword argument as an `HTTPDigestAuth` instance for HTTP Digest Authentication.
**REQ-3.6.3** The system shall accept any callable as an `auth` argument; the callable shall receive the `PreparedRequest` and shall return it modified.
**REQ-3.6.4** The system shall read credentials from `.netrc` if no explicit `auth` is provided and a matching host entry exists.

### 3.7 Sessions
**REQ-3.7.1** The system shall expose a `Session` class that persists state across requests.
**REQ-3.7.2** A `Session` shall persist cookies set by any prior request in that session.
**REQ-3.7.3** A `Session` shall expose `headers`, `auth`, and `params` attributes whose values are sent on every request issued through that session.
**REQ-3.7.4** A `Session` shall reuse TCP connections to the same host using HTTP keep-alive where the server permits.
**REQ-3.7.5** A `Session` shall be usable as a context manager and shall close its connections on exit.

### 3.8 Cookies
**REQ-3.8.1** The system shall accept a `cookies` keyword argument as a dict or `RequestsCookieJar`.
**REQ-3.8.2** The system shall expose response cookies via the `Response.cookies` attribute.
**REQ-3.8.3** Cookie persistence within a `Session` shall be automatic — the caller is not required to extract and re-send cookies between calls.

### 3.9 Redirects
**REQ-3.9.1** The system shall follow HTTP 3xx redirects automatically by default.
**REQ-3.9.2** The system shall accept an `allow_redirects` keyword argument to disable automatic redirect following.
**REQ-3.9.3** The system shall raise `TooManyRedirects` if the number of consecutive redirects exceeds the configured maximum (default 30).
**REQ-3.9.4** The system shall record each intermediate response in the `Response.history` list.

### 3.10 Timeouts
**REQ-3.10.1** The system shall accept a `timeout` keyword argument as a number or `(connect, read)` tuple.
**REQ-3.10.2** When the connect timeout elapses, the system shall raise `requests.exceptions.ConnectTimeout`.
**REQ-3.10.3** When the read timeout elapses, the system shall raise `requests.exceptions.ReadTimeout`.
**REQ-3.10.4** A `timeout` value of `None` shall mean "wait forever" (no timeout).

### 3.11 Proxies
**REQ-3.11.1** The system shall accept a `proxies` keyword argument as a dict mapping URL scheme to proxy URL.
**REQ-3.11.2** The system shall respect the standard `HTTP_PROXY`, `HTTPS_PROXY`, and `NO_PROXY` environment variables.

### 3.12 TLS/SSL
**REQ-3.12.1** The system shall verify HTTPS server certificates against a trusted CA bundle by default.
**REQ-3.12.2** The system shall accept a `verify` keyword argument; setting it to `False` shall disable certificate verification.
**REQ-3.12.3** Setting `verify` to a filesystem path shall use that path as the CA bundle.
**REQ-3.12.4** When server certificate verification fails, the system shall raise `requests.exceptions.SSLError`.

### 3.13 Streaming
**REQ-3.13.1** The system shall accept a `stream=True` keyword argument; when set, the response body shall not be downloaded immediately.
**REQ-3.13.2** The system shall expose `Response.iter_content(chunk_size)` to iterate over the body in fixed-size chunks.
**REQ-3.13.3** The system shall expose `Response.iter_lines()` to iterate over the body line by line.

### 3.14 Encoding
**REQ-3.14.1** The system shall detect the response encoding from the `Content-Type` header where available.
**REQ-3.14.2** When no encoding is declared, the system shall fall back to `charset-normalizer` for detection.
**REQ-3.14.3** The caller shall be able to override the detected encoding by assigning to `Response.encoding`.

### 3.15 Exceptions
**REQ-3.15.1** All exceptions raised by the library shall inherit from `requests.exceptions.RequestException`.
**REQ-3.15.2** Network unreachability shall raise `ConnectionError`.
**REQ-3.15.3** Operations on a closed response shall raise an explicit error rather than silently returning empty data.

---

## 4. Non-Functional Requirements

### 4.1 Performance
**REQ-4.1.1** Session-level connection pooling shall provide measurable throughput gains when issuing more than 5 requests to the same host.
**REQ-4.1.2** The library shall not load the entire response body into memory when `stream=True` is used.

### 4.2 Reliability
**REQ-4.2.1** A failure in one request shall not leave the Session in an unusable state.
**REQ-4.2.2** All allocated sockets shall be released when the Session is closed.

### 4.3 Security
**REQ-4.3.1** The library shall not log credentials, cookies, or auth headers at any default log level.
**REQ-4.3.2** Credentials in `.netrc` shall only be used when the request host matches a `.netrc` entry exactly.

### 4.4 Usability
**REQ-4.4.1** The public API surface shall be importable from the top-level `requests` package.
**REQ-4.4.2** Exception messages shall identify the failing URL.

---

## 5. Out of Scope
- WebSocket support
- HTTP/2 protocol support
- An asynchronous (`async`/`await`) variant of the API
- Server-side functionality
