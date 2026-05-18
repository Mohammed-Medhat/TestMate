import pytest
from unittest.mock import MagicMock, patch
from requests.hooks import *


# FAILING TEST (exhausted retries): Potential Bug
def test_default_hooks():
    result = default_hooks()
    expected = {
        'response': [],
        'request': [],
        'response_redirect': [],
        'response_content': [],
        'response_headers': [],
        'response_body': [],
        'response_exception': [],
        'response_request': [],
        'response_response': [],
        'response_status_code': [],
        'response_url': [],
        'response_encoding': [],
        'response_reason_phrase': [],
        'response_is_redirect': [],
        'response_is_error': [],
        'response_is_success': [],
        'response_is_client_error': [],
        'response_is_server_error': [],
        'response_is_informational': [],
        'response_is_no_content': [],
        'response_is_not_modified': [],
        'response_is_partial_content': [],
        'response_is_redirect_permanent': [],
        'response_is_redirect_temporary': [],
        'response_is_redirect_moved': [],
        'response_is_redirect_found': [],
        'response_is_redirect_see_other': [],
        'response_is_redirect_use_proxy': [],
        'response_is_redirect_temporary_redirect': [],
        'response_is_redirect_permanent_redirect': [],
        'response_is_redirect_found_redirect': [],
        'response_is_redirect_see_other_redirect': [],
        'response_is_redirect_use_proxy_redirect': [],
        'response_is_redirect_temporary_redirect_redirect': [],
        'response_is_redirect_permanent_redirect_redirect': [],
        'response_is_redirect_found_redirect_redirect': [],
        'response_is_redirect_see_other_redirect_redirect': [],
        'response_is_redirect_use_proxy_redirect_redirect': []
    }
    assert result == expected
