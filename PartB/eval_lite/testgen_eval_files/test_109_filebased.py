import django
from django.conf import settings
if not settings.configured:
    settings.configure(
        DATABASES={'default': {'ENGINE': 'django.db.backends.sqlite3', 'NAME': ':memory:'}},
        INSTALLED_APPS=['django.contrib.contenttypes','django.contrib.auth'],
        USE_TZ=True, USE_L10N=False,
        USE_THOUSAND_SEPARATOR=False, USE_I18N=True,
        LANGUAGE_CODE='en-us',
        DEFAULT_AUTO_FIELD='django.db.models.AutoField',
        SECRET_KEY='test-secret-key-for-testing-only',
        DEFAULT_HASHING_ALGORITHM='sha256',
        PASSWORD_RESET_TIMEOUT=259200,
    )
    django.setup()

import pytest
from unittest.mock import MagicMock, patch
import importlib.util, sys, os
_spec = importlib.util.spec_from_file_location("109_filebased", r"D:\TestMate\TestMate\PartB\testgen_eval_lite\testgen_eval_files\109_filebased.py")
_mod = importlib.util.module_from_spec(_spec)
sys.modules["109_filebased"] = _mod
_spec.loader.exec_module(_mod)
globals().update({k: v for k, v in _mod.__dict__.items() if not k.startswith('__')})
