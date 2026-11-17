"""Check backend inspection without importing optional compute engines."""
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from klygo.models.runtime import keras as runtime


class KerasEnvironmentTest(unittest.TestCase):
    def test_reads_actual_backend_without_changing_environment(self):
        fake = SimpleNamespace(backend=SimpleNamespace(backend=lambda: 'jax'))
        with patch.dict(os.environ, {'KERAS_BACKEND': 'torch'}), \
             patch.object(runtime, '_require_keras', return_value=fake):
            self.assertEqual(runtime.get_active_backend(), 'jax')
            self.assertEqual(os.environ['KERAS_BACKEND'], 'torch')
        self.assertFalse(hasattr(runtime, 'ensure_backend'))
